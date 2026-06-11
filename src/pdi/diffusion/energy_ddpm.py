from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import tqdm
from torch_geometric.data import Data

from pdi.diffusion.ddpm import DDPM


def _safe_torch_load(path: Path, *, weights_only: Optional[bool] = None) -> Any:
    """Torch load helper compatible with older versions lacking weights_only."""
    try:
        if weights_only is None:
            return torch.load(path, map_location="cpu")
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass
class _NetworkChannelData:
    """Cached per-network channel/system metadata for energy computation."""

    tx_assoc: torch.Tensor  # [N], receiver -> paired transmitter
    h_ls_gains: torch.Tensor  # [N, N], fallback gains from large-scale channels
    h_timeslot_paths: List[Path]  # precomputed instantaneous channel slots
    p_max: float
    noise_var: float
    r_min: float


@dataclass
class _EnergySamplerContext:
    """Per-batch tensors needed for analytical energy score estimation."""

    gains: torch.Tensor  # [B, R, N, N], R channel realizations
    p_max: torch.Tensor  # [B, 1, 1]
    noise_var: torch.Tensor  # [B, 1, 1]
    r_min: torch.Tensor  # [B, 1, 1]


class _EnergyScoreMixin:
    """Shared helpers for Energy-DDPM and Energy-DDIM analytical score sampling."""

    def _init_energy_params(
        self,
        *,
        energy_mc_samples: int,
        energy_num_channel_realizations: int,
        dual_num_channel_realizations: Optional[int] = None,
        inverse_beta: float,
        inverse_beta_schedule: str,
        inverse_beta_start: Optional[float],
        inverse_beta_end: Optional[float],
        dual_update_mode: str,
        dual_step_size: float,
        dual_step_size_schedule: str,
        dual_step_size_start: Optional[float],
        dual_step_size_end: Optional[float],
        dual_num_outer_iterations: int,
        dual_lambda_init: float,
        dual_lambda_max: Optional[float],
        dual_lambda_decay: float,
        dual_lambda_mode: str,
        langevin_step_size: float,
        langevin_noise_scale: float,
        dataset_root: Optional[str],
        default_p_max: float,
        default_noise_var: float,
        default_r_min: float,
        r_min_override: Optional[float],
        min_energy_sigma: float,
        use_precomputed_channels: bool,
        allow_sparse_graph_fallback: bool,
    ) -> None:
        if energy_mc_samples <= 0:
            raise ValueError(f"energy_mc_samples must be > 0, got {energy_mc_samples}")
        if energy_num_channel_realizations <= 0:
            raise ValueError(
                "energy_num_channel_realizations must be > 0, "
                f"got {energy_num_channel_realizations}"
            )
        if inverse_beta <= 0.0:
            raise ValueError(f"inverse_beta must be > 0, got {inverse_beta}")
        if inverse_beta_start is not None and inverse_beta_start <= 0.0:
            raise ValueError(
                f"inverse_beta_start must be > 0, got {inverse_beta_start}"
            )
        if inverse_beta_end is not None and inverse_beta_end <= 0.0:
            raise ValueError(f"inverse_beta_end must be > 0, got {inverse_beta_end}")
        if min_energy_sigma <= 0.0:
            raise ValueError(f"min_energy_sigma must be > 0, got {min_energy_sigma}")
        if dual_step_size <= 0.0:
            raise ValueError(f"dual_step_size must be > 0, got {dual_step_size}")
        if dual_step_size_start is not None and dual_step_size_start <= 0.0:
            raise ValueError(
                f"dual_step_size_start must be > 0, got {dual_step_size_start}"
            )
        if dual_step_size_end is not None and dual_step_size_end <= 0.0:
            raise ValueError(
                f"dual_step_size_end must be > 0, got {dual_step_size_end}"
            )
        if dual_num_outer_iterations <= 0:
            raise ValueError(
                "dual_num_outer_iterations must be > 0, "
                f"got {dual_num_outer_iterations}"
            )
        if dual_lambda_max is not None and dual_lambda_max <= 0.0:
            raise ValueError(f"dual_lambda_max must be > 0, got {dual_lambda_max}")
        if not (0.0 <= dual_lambda_decay < 1.0):
            raise ValueError(f"dual_lambda_decay must be in [0, 1), got {dual_lambda_decay}")
        if langevin_step_size <= 0.0:
            raise ValueError(
                f"langevin_step_size must be > 0, got {langevin_step_size}"
            )
        if langevin_noise_scale < 0.0:
            raise ValueError(
                f"langevin_noise_scale must be >= 0, got {langevin_noise_scale}"
            )
        if r_min_override is not None and not math.isfinite(float(r_min_override)):
            raise ValueError(
                f"r_min_override must be finite when provided, got {r_min_override}."
            )

        self.energy_mc_samples = int(energy_mc_samples)
        self.energy_num_channel_realizations = int(energy_num_channel_realizations)
        self.dual_num_channel_realizations = (
            None if dual_num_channel_realizations is None
            else int(dual_num_channel_realizations)
        )
        self.inverse_beta = float(inverse_beta)
        self.inverse_beta_schedule = str(inverse_beta_schedule).strip().lower()
        if self.inverse_beta_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(
                "inverse_beta_schedule must be one of {'constant', 'linear', 'cosine'}, "
                f"got '{inverse_beta_schedule}'."
            )
        self.inverse_beta_start = float(
            inverse_beta if inverse_beta_start is None else inverse_beta_start
        )
        self.inverse_beta_end = float(
            inverse_beta if inverse_beta_end is None else inverse_beta_end
        )
        if self.inverse_beta_schedule == "constant":
            inverse_beta_by_t = torch.full(
                (self.num_timesteps,), self.inverse_beta_start, dtype=torch.float32,
            )
        elif self.inverse_beta_schedule == "linear":
            inverse_beta_by_t = torch.linspace(
                self.inverse_beta_start,
                self.inverse_beta_end,
                steps=self.num_timesteps,
                dtype=torch.float32,
            )
        else:
            ratio = torch.linspace(0.0, 1.0, steps=self.num_timesteps, dtype=torch.float32)
            weight = 0.5 * (1.0 - torch.cos(math.pi * ratio))
            inverse_beta_by_t = (
                self.inverse_beta_start
                + (self.inverse_beta_end - self.inverse_beta_start) * weight
            )
        self.register_buffer("inverse_beta_by_t", inverse_beta_by_t.clamp_min(1e-12))
        dual_mode = str(dual_update_mode).strip().lower()
        if dual_mode not in ("x0_pred", "full_backward", "hybrid", "langevin"):
            raise ValueError(
                "dual_update_mode must be one of "
                "{'x0_pred', 'full_backward', 'hybrid', 'langevin'}, "
                f"got '{dual_update_mode}'."
            )
        self.dual_update_mode = dual_mode
        self.dual_step_size = float(dual_step_size)
        self.dual_step_size_schedule = str(dual_step_size_schedule).strip().lower()
        if self.dual_step_size_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(
                "dual_step_size_schedule must be one of {'constant', 'linear', 'cosine'}, "
                f"got '{dual_step_size_schedule}'."
            )
        self.dual_step_size_start = float(
            dual_step_size if dual_step_size_start is None else dual_step_size_start
        )
        self.dual_step_size_end = float(
            dual_step_size if dual_step_size_end is None else dual_step_size_end
        )
        if self.dual_step_size_schedule == "constant":
            dual_step_size_by_t = torch.full(
                (self.num_timesteps,), self.dual_step_size_start, dtype=torch.float32,
            )
        elif self.dual_step_size_schedule == "linear":
            dual_step_size_by_t = torch.linspace(
                self.dual_step_size_start,
                self.dual_step_size_end,
                steps=self.num_timesteps,
                dtype=torch.float32,
            )
        else:
            ratio = torch.linspace(0.0, 1.0, steps=self.num_timesteps, dtype=torch.float32)
            weight = 0.5 * (1.0 - torch.cos(math.pi * ratio))
            dual_step_size_by_t = (
                self.dual_step_size_start
                + (self.dual_step_size_end - self.dual_step_size_start) * weight
            )
        self.register_buffer("dual_step_size_by_t", dual_step_size_by_t.clamp_min(1e-12))
        self.dual_num_outer_iterations = int(dual_num_outer_iterations)
        self.dual_lambda_init = float(dual_lambda_init)
        self.dual_lambda_max = (
            None if dual_lambda_max is None else float(dual_lambda_max)
        )
        self.dual_lambda_decay = float(dual_lambda_decay)
        dual_lambda_mode_norm = str(dual_lambda_mode).strip().lower()
        if dual_lambda_mode_norm not in ("per_sample", "shared_per_network"):
            raise ValueError(
                "dual_lambda_mode must be one of "
                "{'per_sample', 'shared_per_network'}, "
                f"got '{dual_lambda_mode}'."
            )
        self.dual_lambda_mode = dual_lambda_mode_norm
        self.langevin_step_size = float(langevin_step_size)
        self.langevin_noise_scale = float(langevin_noise_scale)
        self.dataset_root = Path(dataset_root).expanduser() if dataset_root else None
        self.default_p_max = float(default_p_max)
        self.default_noise_var = float(default_noise_var)
        self.default_r_min = float(default_r_min)
        self.r_min_override = (
            None if r_min_override is None else float(r_min_override)
        )
        self.min_energy_sigma = float(min_energy_sigma)
        self.use_precomputed_channels = bool(use_precomputed_channels)
        self.allow_sparse_graph_fallback = bool(allow_sparse_graph_fallback)

        self._dataset_meta_cache: Dict[str, Dict[str, float]] = {}
        self._network_channel_cache: Dict[Tuple[str, int], _NetworkChannelData] = {}

    @staticmethod
    def _init_dual_lambda(
        batch_size: int,
        num_nodes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        init_value: float,
    ) -> torch.Tensor:
        return torch.full(
            (batch_size, num_nodes),
            float(init_value),
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Dataset + network metadata loading/caching
    # ------------------------------------------------------------------

    def _metadata_path(self, dataset_name: str) -> Path:
        if self.dataset_root is None:
            raise ValueError(
                "Energy diffusion needs dataset_root when use_precomputed_channels=True."
            )
        return self.dataset_root / dataset_name / "processed" / "metadata.json"

    def _network_graph_path(self, dataset_name: str, network_id: int) -> Path:
        if self.dataset_root is None:
            raise ValueError(
                "Energy diffusion needs dataset_root when use_precomputed_channels=True."
            )
        return (
            self.dataset_root
            / dataset_name
            / "processed"
            / f"network_{network_id}"
            / "graph.pt"
        )

    def _network_h_timeslot_dir(self, dataset_name: str, network_id: int) -> Path:
        if self.dataset_root is None:
            raise ValueError(
                "Energy diffusion needs dataset_root when use_precomputed_channels=True."
            )
        return (
            self.dataset_root
            / dataset_name
            / "processed"
            / f"network_{network_id}"
            / "H_instantaneous"
        )

    def _get_dataset_system_params(self, dataset_name: str) -> Dict[str, float]:
        cached = self._dataset_meta_cache.get(dataset_name, None)
        if cached is not None:
            return cached

        params = {
            "p_max": self.default_p_max,
            "noise_var": self.default_noise_var,
            "r_min": self.default_r_min,
        }
        meta_path = self._metadata_path(dataset_name)
        if not meta_path.exists():
            if self.use_precomputed_channels:
                required_text = "P_max, noise_var"
                if self.r_min_override is None:
                    required_text += ", and r_min"
                raise FileNotFoundError(
                    f"Missing dataset metadata for '{dataset_name}': {meta_path}. "
                    "Energy sampling with precomputed channels requires metadata.json "
                    f"containing {required_text}."
                )
            if self.r_min_override is not None:
                params["r_min"] = float(self.r_min_override)
            self._dataset_meta_cache[dataset_name] = params
            return params

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:
            if self.use_precomputed_channels:
                raise RuntimeError(
                    f"Failed to parse dataset metadata for '{dataset_name}': {meta_path}"
                ) from exc
            if self.r_min_override is not None:
                params["r_min"] = float(self.r_min_override)
            self._dataset_meta_cache[dataset_name] = params
            return params

        required = ["P_max", "noise_var"]
        if self.r_min_override is None:
            required.append("r_min")
        missing = [k for k in required if k not in metadata]
        if missing and self.use_precomputed_channels:
            required_text = "P_max, noise_var"
            if self.r_min_override is None:
                required_text += ", and r_min"
            raise KeyError(
                f"Dataset metadata for '{dataset_name}' is missing required keys {missing}. "
                f"Energy sampling with precomputed channels needs {required_text}."
            )

        if "P_max" in metadata:
            params["p_max"] = float(metadata["P_max"])
        if "noise_var" in metadata:
            params["noise_var"] = float(metadata["noise_var"])
        if "r_min" in metadata:
            params["r_min"] = float(metadata["r_min"])
        if self.r_min_override is not None:
            params["r_min"] = float(self.r_min_override)

        self._dataset_meta_cache[dataset_name] = params
        return params

    @staticmethod
    def _list_timeslot_paths(timeslot_dir: Path, num_available: int) -> List[Path]:
        """Resolve timeslot files in new and legacy naming conventions."""
        if num_available > 0:
            out: List[Path] = []
            for idx in range(num_available):
                new_path = timeslot_dir / f"timestep_{idx}.pt"
                legacy_path = timeslot_dir / f"h_timeslot_{idx}.pt"
                if new_path.exists() or not legacy_path.exists():
                    out.append(new_path)
                else:
                    out.append(legacy_path)
            return out

        paths = sorted(timeslot_dir.glob("timestep_*.pt"))
        if len(paths) == 0:
            paths = sorted(timeslot_dir.glob("h_timeslot_*.pt"))
        return paths

    @staticmethod
    def _decode_timeslot_tensor(slot_obj: Any) -> torch.Tensor:
        """Decode loaded timeslot object into a rank-2 tensor [m, n]."""
        if isinstance(slot_obj, dict):
            if "H" in slot_obj:
                slot_obj = slot_obj["H"]
            elif "H_inst" in slot_obj:
                slot_obj = slot_obj["H_inst"]
        if not isinstance(slot_obj, torch.Tensor):
            slot_obj = torch.as_tensor(slot_obj)
        if slot_obj.ndim != 2:
            raise ValueError(
                f"Malformed instantaneous channel slot shape {tuple(slot_obj.shape)}; expected [m, n]."
            )
        return slot_obj.detach().cpu().float()

    def _get_network_channel_data(
        self, dataset_name: str, network_id: int,
    ) -> _NetworkChannelData:
        key = (dataset_name, int(network_id))
        cached = self._network_channel_cache.get(key, None)
        if cached is not None:
            return cached

        graph_path = self._network_graph_path(dataset_name, network_id)
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing graph metadata: {graph_path}")

        graph_data = _safe_torch_load(graph_path, weights_only=True)
        if "H_ls" not in graph_data or "associations" not in graph_data:
            raise KeyError(
                f"graph.pt for {dataset_name}/network_{network_id} lacks H_ls/associations."
            )

        h_ls = graph_data["H_ls"]
        associations = graph_data["associations"]
        if not isinstance(h_ls, torch.Tensor):
            h_ls = torch.tensor(h_ls, dtype=torch.float32)
        if not isinstance(associations, torch.Tensor):
            associations = torch.tensor(associations, dtype=torch.float32)

        h_ls = h_ls.detach().cpu().float()
        associations = associations.detach().cpu().float()
        if h_ls.dim() != 2 or associations.dim() != 2:
            raise ValueError(
                f"Expected 2D H_ls/associations, got {tuple(h_ls.shape)} and {tuple(associations.shape)}."
            )
        if h_ls.shape != associations.shape:
            raise ValueError(
                f"H_ls and associations shape mismatch: {tuple(h_ls.shape)} vs {tuple(associations.shape)}."
            )

        # tx_assoc[j] = transmitter paired with receiver j.
        tx_assoc = associations.argmax(dim=0).to(torch.long)
        h_ls_gains = h_ls.index_select(0, tx_assoc).contiguous()  # [N, N]

        num_available = int(graph_data.get("h_num_timesteps", 0) or 0)
        timeslot_dir = self._network_h_timeslot_dir(dataset_name, network_id)
        timeslot_paths: List[Path] = []
        if timeslot_dir.exists():
            timeslot_paths = self._list_timeslot_paths(timeslot_dir, num_available)

        params = self._get_dataset_system_params(dataset_name)
        info = _NetworkChannelData(
            tx_assoc=tx_assoc,
            h_ls_gains=h_ls_gains,
            h_timeslot_paths=timeslot_paths,
            p_max=float(params["p_max"]),
            noise_var=float(params["noise_var"]),
            r_min=float(params["r_min"]),
        )
        self._network_channel_cache[key] = info
        return info

    # ------------------------------------------------------------------
    # Batch context construction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_batched_values(values: List[Any], batch_size: int) -> List[Any]:
        if len(values) == batch_size:
            return values
        if len(values) == 1 and batch_size > 1:
            return values * batch_size
        raise ValueError(f"Expected {batch_size} values, got {len(values)}.")

    def _resolve_batch_keys(
        self, data: Data, batch_size: int,
    ) -> Tuple[Optional[List[str]], Optional[List[int]]]:
        dataset_names = self._to_python_list(getattr(data, "dataset_name", None))
        network_ids = self._to_python_list(getattr(data, "network_id", None))
        if len(dataset_names) == 0 or len(network_ids) == 0:
            return None, None

        ds_values = [
            str(v) for v in self._normalize_batched_values(dataset_names, batch_size)
        ]
        net_values = [
            int(v) for v in self._normalize_batched_values(network_ids, batch_size)
        ]
        return ds_values, net_values

    def _load_instantaneous_gains_for_network(
        self,
        net_data: _NetworkChannelData,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        num_realizations_override: Optional[int] = None,
    ) -> torch.Tensor:
        """Load selected H_instantaneous slots and map to receiver-indexed TX gains."""
        available = len(net_data.h_timeslot_paths)
        required = num_realizations_override if num_realizations_override is not None else self.energy_num_channel_realizations
        if available < required:
            raise ValueError(
                f"Insufficient H_instantaneous slots: available={available}, required={required}."
            )

        if not hasattr(self, "_gains_cache"):
            self._gains_cache: Dict[str, torch.Tensor] = {}

        cache_key = str(net_data.h_timeslot_paths[0].parent)
        if cache_key not in self._gains_cache:
            all_gains: List[torch.Tensor] = []
            for idx in range(available):
                slot_path = net_data.h_timeslot_paths[idx]
                if not slot_path.exists():
                    raise FileNotFoundError(f"Missing H_instantaneous timeslot file: {slot_path}")
                slot = _safe_torch_load(slot_path, weights_only=False)
                h_t = self._decode_timeslot_tensor(slot)  # [m, n]
                gains_t = h_t.index_select(0, net_data.tx_assoc).contiguous()  # [N, N]
                if gains_t.shape[0] != num_nodes or gains_t.shape[1] != num_nodes:
                    raise ValueError(
                        "Instantaneous channel slot size mismatch: "
                        f"got {tuple(gains_t.shape)}, expected ({num_nodes}, {num_nodes})."
                    )
                all_gains.append(gains_t.to(device=device, dtype=dtype))
            self._gains_cache[cache_key] = torch.stack(all_gains, dim=0)  # [R_all, N, N]

        cached = self._gains_cache[cache_key]
        selected = torch.randperm(available, device=cached.device)[:required]
        return cached[selected]  # [R, N, N]

    def _build_context_from_precomputed_channels(
        self,
        data: Data,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        num_realizations_override: Optional[int] = None,
    ) -> _EnergySamplerContext:
        keys = self._resolve_batch_keys(data, batch_size)
        if keys[0] is None or keys[1] is None:
            raise ValueError("dataset_name/network_id are required for precomputed channels.")

        dataset_names, network_ids = keys

        # Deduplicate: load gains once per unique (ds_name, net_id)
        unique_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, float, float, float]] = {}
        for ds_name, net_id in zip(dataset_names, network_ids):
            key = (str(ds_name), int(net_id))
            if key in unique_cache:
                continue
            net_data = self._get_network_channel_data(ds_name, net_id)
            if (
                net_data.h_ls_gains.shape[0] != num_nodes
                or net_data.h_ls_gains.shape[1] != num_nodes
            ):
                raise ValueError(
                    f"Network ({ds_name}, {net_id}) has {tuple(net_data.h_ls_gains.shape)} "
                    f"channel matrix but sampler expects ({num_nodes}, {num_nodes})."
                )

            if self.use_precomputed_channels:
                gains_rnn = self._load_instantaneous_gains_for_network(
                    net_data=net_data,
                    num_nodes=num_nodes,
                    device=device,
                    dtype=dtype,
                    num_realizations_override=num_realizations_override,
                )
            else:
                gains_rnn = net_data.h_ls_gains.to(device=device, dtype=dtype).unsqueeze(0)

            unique_cache[key] = (gains_rnn, float(net_data.p_max),
                                 float(net_data.noise_var), float(net_data.r_min))

        gains_list: List[torch.Tensor] = []
        p_max_list: List[float] = []
        noise_var_list: List[float] = []
        r_min_list: List[float] = []
        for ds_name, net_id in zip(dataset_names, network_ids):
            g, pm, nv, rm = unique_cache[(str(ds_name), int(net_id))]
            gains_list.append(g)
            p_max_list.append(pm)
            noise_var_list.append(nv)
            r_min_list.append(rm)

        if len(unique_cache) == 1:
            # All B samples share the same network — store [1, R, N, N]
            gains = gains_list[0].unsqueeze(0)  # [1, R, N, N]
        else:
            gains = torch.stack(gains_list, dim=0)  # [B, R, N, N]
        p_max = torch.tensor(p_max_list, device=device, dtype=dtype).view(-1, 1, 1)
        noise_var = torch.tensor(noise_var_list, device=device, dtype=dtype).view(-1, 1, 1)
        r_min = torch.tensor(r_min_list, device=device, dtype=dtype).view(-1, 1, 1)
        if self.r_min_override is not None:
            r_min.fill_(self.r_min_override)
        return _EnergySamplerContext(
            gains=gains, p_max=p_max, noise_var=noise_var, r_min=r_min,
        )

    def _build_context_from_sparse_graph(
        self,
        data: Data,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _EnergySamplerContext:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError("data.edge_index is required for sparse-graph energy fallback.")

        edge_index = data.edge_index.to(device=device)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")

        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device, dtype=dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=dtype)

        gains = torch.zeros((batch_size, num_nodes, num_nodes), device=device, dtype=dtype)
        src = edge_index[0]
        dst = edge_index[1]

        ptr = getattr(data, "ptr", None)
        if isinstance(ptr, torch.Tensor) and ptr.numel() == batch_size + 1:
            ptr = ptr.to(device=device, dtype=torch.long)
            for b in range(batch_size):
                start = int(ptr[b].item())
                end = int(ptr[b + 1].item())
                if (end - start) != num_nodes:
                    raise ValueError(
                        f"Graph {b} has {end - start} nodes; expected fixed num_nodes={num_nodes}."
                    )
                in_graph = (src >= start) & (src < end) & (dst >= start) & (dst < end)
                if not in_graph.any():
                    continue
                src_local = src[in_graph] - start
                dst_local = dst[in_graph] - start
                gains[b].index_put_((src_local, dst_local), edge_weight[in_graph], accumulate=True)
        else:
            for b in range(batch_size):
                start = b * num_nodes
                end = start + num_nodes
                in_graph = (src >= start) & (src < end) & (dst >= start) & (dst < end)
                if not in_graph.any():
                    continue
                src_local = src[in_graph] - start
                dst_local = dst[in_graph] - start
                gains[b].index_put_((src_local, dst_local), edge_weight[in_graph], accumulate=True)

        p_max = torch.full((batch_size, 1, 1), self.default_p_max, device=device, dtype=dtype)
        noise_var = torch.full(
            (batch_size, 1, 1), self.default_noise_var, device=device, dtype=dtype,
        )
        r_min = torch.full((batch_size, 1, 1), self.default_r_min, device=device, dtype=dtype)

        keys = self._resolve_batch_keys(data, batch_size)
        if keys[0] is not None:
            dataset_names = keys[0]
            for b, ds_name in enumerate(dataset_names):
                params = self._get_dataset_system_params(ds_name)
                p_max[b, 0, 0] = float(params["p_max"])
                noise_var[b, 0, 0] = float(params["noise_var"])
                r_min[b, 0, 0] = float(params["r_min"])

        # Sparse-graph fallback has one static realization.
        return _EnergySamplerContext(
            gains=gains.unsqueeze(1), p_max=p_max, noise_var=noise_var, r_min=r_min,
        )

    def _build_energy_context(
        self,
        data: Data,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        num_realizations_override: Optional[int] = None,
    ) -> _EnergySamplerContext:
        if self.use_precomputed_channels:
            try:
                return self._build_context_from_precomputed_channels(
                    data=data,
                    batch_size=batch_size,
                    num_nodes=num_nodes,
                    device=device,
                    dtype=dtype,
                    num_realizations_override=num_realizations_override,
                )
            except Exception:
                if not self.allow_sparse_graph_fallback:
                    raise

        return self._build_context_from_sparse_graph(
            data=data,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=device,
            dtype=dtype,
        )

    def _build_dual_context(
        self,
        data: Data,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[_EnergySamplerContext]:
        """Build a separate context for dual updates with more channel realizations.

        Returns None when dual_num_channel_realizations is not set (caller
        should fall back to the main context).
        """
        if self.dual_num_channel_realizations is None:
            return None
        return self._build_energy_context(
            data=data,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=device,
            dtype=dtype,
            num_realizations_override=self.dual_num_channel_realizations,
        )

    # ------------------------------------------------------------------
    # Energy + score
    # ------------------------------------------------------------------

    def _rate_tensor_from_samples(
        self,
        x: torch.Tensor,  # [B, T, N, F]
        context: _EnergySamplerContext,
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x to be rank-4 [B,T,N,F], got shape {tuple(x.shape)}")
        if x.size(-1) < 1:
            raise ValueError(f"Expected x feature dimension >=1, got {x.size(-1)}")

        power_norm = x[..., 0]  # [B, T, N]
        p_max = context.p_max  # [B,1,1]
        power = (power_norm + 0.5) * p_max  # [B, T, N]

        # gains is [B, R, N, N] or [1, R, N, N] (shared network).
        gains = context.gains
        if gains.shape[0] == 1 and x.shape[0] > 1:
            # Shared network: use "rij,bti->brtj" to avoid materializing B×R×N×N
            received_power = torch.einsum("rij,bti->brtj", gains[0], power)
            direct_gain = torch.diagonal(gains[0], dim1=1, dim2=2).unsqueeze(0).unsqueeze(2)  # [1,R,1,N]
        else:
            received_power = torch.einsum("brij,bti->brtj", gains, power)
            direct_gain = torch.diagonal(gains, dim1=2, dim2=3).unsqueeze(2)  # [B,R,1,N]
        direct_signal = power.unsqueeze(1) * direct_gain  # [B,R,T,N]
        interference = (received_power - direct_signal).clamp_min(0.0)

        noise = context.noise_var.clamp_min(1e-20).unsqueeze(1)  # [B,1,1,1]
        sinr = direct_signal.clamp_min(0.0) / (interference + noise)
        return torch.log2(1.0 + sinr)  # [B,R,T,N]

    def _ergodic_rates_from_samples(
        self,
        x: torch.Tensor,  # [B, T, N, F]
        context: _EnergySamplerContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rates = self._rate_tensor_from_samples(x=x, context=context)
        # Per-user ergodic rates: average over channel realizations and time axis.
        ergodic_rates = rates.mean(dim=(1, 2))  # [B, N]
        sum_rates = ergodic_rates.sum(dim=1)  # [B]
        return ergodic_rates, sum_rates

    def _time_shared_ergodic_rates_broadcast(
        self,
        x: torch.Tensor,  # [B, T, N, F]
        context: _EnergySamplerContext,
        n_samples_per_input: int,
    ) -> torch.Tensor:
        # Uniform time-sharing across the K policies of each network, then
        # broadcast the per-network rate back to [B, N] so the shared dual
        # update produces block-constant λ via the existing ascent step.
        per_sample_rates, _ = self._ergodic_rates_from_samples(x=x, context=context)
        B, N = per_sample_rates.shape
        K = max(1, int(n_samples_per_input))
        if K == 1 or B % K != 0:
            return per_sample_rates
        n_nets = B // K
        shared = per_sample_rates.view(n_nets, K, N).mean(dim=1)  # [n_nets, N]
        return shared.unsqueeze(1).expand(n_nets, K, N).reshape(B, N).contiguous()

    def _dual_ascent_step(
        self,
        dual_lambda: torch.Tensor,  # [B, N]
        ergodic_rates: torch.Tensor,  # [B, N]
        context: _EnergySamplerContext,
        t: Optional[torch.Tensor] = None,  # [B], DDPM timestep indices
    ) -> torch.Tensor:
        if dual_lambda.shape != ergodic_rates.shape:
            raise ValueError(
                "dual_lambda / ergodic_rates shape mismatch: "
                f"{tuple(dual_lambda.shape)} vs {tuple(ergodic_rates.shape)}."
            )
        if t is None:
            dual_step_size_t = self.dual_step_size_by_t[0].to(
                device=dual_lambda.device, dtype=dual_lambda.dtype,
            ).expand(dual_lambda.shape[0])
        else:
            t_safe = t.to(device=dual_lambda.device, dtype=torch.long).clamp(
                0, self.num_timesteps - 1,
            )
            dual_step_size_t = self._gather(
                self.dual_step_size_by_t.to(device=dual_lambda.device), t_safe,
            ).to(dtype=dual_lambda.dtype)
        violation = context.r_min.view(-1, 1) - ergodic_rates  # [B, N]
        lam_decayed = (1.0 - self.dual_lambda_decay) * dual_lambda
        updated = (lam_decayed + dual_step_size_t.view(-1, 1) * violation).clamp_min(0.0)
        if self.dual_lambda_max is not None:
            updated = updated.clamp_max(float(self.dual_lambda_max))
        return updated

    @staticmethod
    def _per_network_dual_lambda_lists(
        dual_lambda: torch.Tensor,  # [B, N]
        n_samples_per_input: int,
    ) -> List[List[float]]:
        B, _N = dual_lambda.shape
        K = max(1, int(n_samples_per_input))
        n_nets = max(1, B // K)
        out: List[List[float]] = []
        for n in range(n_nets):
            lo, hi = n * K, min((n + 1) * K, B)
            lam_n = dual_lambda[lo:hi].mean(dim=0)  # [N]
            out.append([float(v) for v in lam_n.detach().cpu().tolist()])
        return out

    @staticmethod
    def _per_network_dual_lambda_stats(
        dual_lambda: torch.Tensor,  # [B, N]
        n_samples_per_input: int,
    ) -> List[Dict[str, List[float]]]:
        """Return per-network statistics over the K samples for each user.

        Each list element is a dict with keys ``mean``, ``std``, ``min``,
        ``max`` -- each a length-N list of floats. When K==1 the std is
        reported as zeros.
        """
        B, _N = dual_lambda.shape
        K = max(1, int(n_samples_per_input))
        n_nets = max(1, B // K)
        out: List[Dict[str, List[float]]] = []
        for n in range(n_nets):
            lo, hi = n * K, min((n + 1) * K, B)
            block = dual_lambda[lo:hi].detach()  # [K, N]
            mean = block.mean(dim=0)
            if block.shape[0] > 1:
                std = block.std(dim=0, unbiased=False)
            else:
                std = torch.zeros_like(mean)
            mn = block.min(dim=0).values
            mx = block.max(dim=0).values
            out.append(
                {
                    "mean": [float(v) for v in mean.cpu().tolist()],
                    "std": [float(v) for v in std.cpu().tolist()],
                    "min": [float(v) for v in mn.cpu().tolist()],
                    "max": [float(v) for v in mx.cpu().tolist()],
                }
            )
        return out

    def _energy_from_samples(
        self,
        x: torch.Tensor,  # [B, T, N, F]
        context: _EnergySamplerContext,
        dual_lambda: Optional[torch.Tensor] = None,  # [B, N]
        t: Optional[torch.Tensor] = None,  # [B], DDPM timestep indices
        inverse_beta_override: Optional[torch.Tensor] = None,  # [B] or scalar
    ) -> torch.Tensor:
        ergodic_rates, sum_rates = self._ergodic_rates_from_samples(x=x, context=context)

        if dual_lambda is None:
            dual_lambda = torch.zeros_like(ergodic_rates)
        if dual_lambda.shape != ergodic_rates.shape:
            raise ValueError(
                "dual_lambda shape mismatch: "
                f"expected {tuple(ergodic_rates.shape)}, got {tuple(dual_lambda.shape)}."
            )

        lagrangian_term = ((context.r_min.view(-1, 1) - ergodic_rates) * dual_lambda).sum(dim=1)  # [B]
        if inverse_beta_override is not None:
            inverse_beta_t = inverse_beta_override.to(device=x.device, dtype=x.dtype).view(-1)
            if inverse_beta_t.numel() == 1:
                inverse_beta_t = inverse_beta_t.expand(x.shape[0])
            elif inverse_beta_t.numel() != x.shape[0]:
                raise ValueError(
                    "inverse_beta_override shape mismatch: "
                    f"expected scalar or [{x.shape[0]}], got {tuple(inverse_beta_t.shape)}."
                )
        elif t is None:
            inverse_beta_t = self.inverse_beta_by_t[0].to(
                device=x.device, dtype=x.dtype,
            ).expand(x.shape[0])
        else:
            t_safe = t.to(device=x.device, dtype=torch.long).clamp(0, self.num_timesteps - 1)
            inverse_beta_t = self._gather(self.inverse_beta_by_t.to(device=x.device), t_safe).to(
                dtype=x.dtype,
            )

        # Lagrangian energy:
        # E = (1/beta) * ( -sum_rates + lambda^T (r_min - ergodic_rates) )
        energy = inverse_beta_t * (-sum_rates + lagrangian_term)
        return energy

    @staticmethod
    def _tile_context(context, K: int):
        """Repeat all batch-dimension tensors in a dataclass context by K.

        Only tiles fields whose leading dimension matches the batch size
        (inferred from ``context.r_min.shape[0]``).  Non-batch fields such
        as MoG parameters ``[K_modes, d]`` are left untouched.

        Tensors with leading dim == 1 (shared across batch, e.g. gains from a
        single network) are left untouched to preserve memory efficiency.
        """
        B = context.r_min.shape[0] if hasattr(context, "r_min") else None
        updates = {}
        for f in fields(context):
            v = getattr(context, f.name)
            if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[0] > 1:
                if B is not None and v.shape[0] != B:
                    continue
                updates[f.name] = v.repeat(K, *([1] * (v.dim() - 1)))
        if not updates:
            return context
        from dataclasses import replace
        return replace(context, **updates)

    def _estimate_score(
        self,
        x_t: torch.Tensor,  # [B, T, N, F]
        t: torch.Tensor,  # [B]
        context: _EnergySamplerContext,
        dual_lambda: Optional[torch.Tensor] = None,  # [B, N]
        inverse_beta_override: Optional[torch.Tensor] = None,  # [B] or scalar
    ) -> torch.Tensor:
        K = self.energy_mc_samples
        B = x_t.shape[0]
        sigma_t = self._gather(self.sqrt_one_minus_alphas_cumprod, t).view(-1, 1, 1, 1)
        sigma_t = sigma_t.clamp_min(self.min_energy_sigma)

        with torch.enable_grad():
            x_t_req = x_t.detach().requires_grad_(True)

            # Tile x_t for all MC samples at once: [K*B, ...]
            x_t_tiled = x_t_req.repeat(K, 1, 1, 1)
            sigma_tiled = sigma_t.repeat(K, 1, 1, 1)
            noise = torch.randn_like(x_t_tiled)
            x0_candidates = x_t_tiled + noise * sigma_tiled
            x0_candidates = self._clip_mc_candidate(x0_candidates)

            # Tile context, dual_lambda, t, ib_override to match K*B batch
            ctx_tiled = self._tile_context(context, K)
            t_tiled = t.repeat(K)
            dl_tiled = dual_lambda.repeat(K, 1) if dual_lambda is not None else None
            ib_tiled = None
            if inverse_beta_override is not None:
                ib_t = inverse_beta_override
                if ib_t.dim() == 0 or ib_t.numel() == 1:
                    ib_tiled = ib_t
                else:
                    ib_tiled = ib_t.repeat(K)

            energies = self._energy_from_samples(
                x0_candidates, ctx_tiled,
                dual_lambda=dl_tiled, t=t_tiled,
                inverse_beta_override=ib_tiled,
            )  # [K*B]

            energy_matrix = energies.view(K, B)
            log_partition = torch.logsumexp(-energy_matrix, dim=0) - math.log(float(K))
            score = torch.autograd.grad(log_partition.sum(), x_t_req)[0]

            # Track IS Neff per step (lightweight, no grad)
            if hasattr(self, '_is_neff_trace'):
                with torch.no_grad():
                    log_w = -energy_matrix - torch.logsumexp(-energy_matrix, dim=0, keepdim=True)
                    is_neff = torch.exp(-torch.logsumexp(2 * log_w, dim=0)).mean().item()
                    self._is_neff_trace.append(is_neff)

        return score.detach()

    def _energy_gradient(
        self,
        x: torch.Tensor,  # [B, T, N, F]
        t: torch.Tensor,  # [B]
        context: _EnergySamplerContext,
        dual_lambda: Optional[torch.Tensor] = None,  # [B, N]
        inverse_beta_override: Optional[torch.Tensor] = None,  # [B] or scalar
    ) -> torch.Tensor:
        """Compute ∇_x E(x; t) for Langevin dynamics."""
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            energy = self._energy_from_samples(
                x_req,
                context,
                dual_lambda=dual_lambda,
                t=t,
                inverse_beta_override=inverse_beta_override,
            )
            grad = torch.autograd.grad(energy.sum(), x_req)[0]
        return grad.detach()

    def _rates_summary_from_samples(
        self,
        x: torch.Tensor,  # [B, T, N, F]
        context: _EnergySamplerContext,
        n_samples_per_input: int = 1,
        dual_lambda: Optional[torch.Tensor] = None,  # [B, N]
        t: Optional[torch.Tensor] = None,  # [B], DDPM timestep indices
        inverse_beta_override: Optional[torch.Tensor] = None,  # [B] or scalar
    ) -> Dict[str, Any]:
        """Return per-network and global time-sharing ergodic metrics.

        For each network in the batch, K = n_samples_per_input policies are available.
        For each of R channel realizations (virtual time slots), one policy is drawn
        uniformly at random. The ergodic rate per user is averaged over the R slots,
        then p5 is the 5th percentile across users within that network.

        Also computes the Lagrangian energy using the time-sharing rates:
            E_n = (1/beta) * (-sum_rate_n + lambda^T (r_min - ergodic_rates_n))

        Returns a dict with keys:
            sum_rate: float  (mean over networks)
            p5: float        (mean over networks)
            energy: float    (mean over networks)
            per_net_sum_rates: List[float]  (one per network)
            per_net_p5s: List[float]        (one per network)
            per_net_energies: List[float]   (one per network)
        """
        with torch.no_grad():
            B = x.shape[0]
            K = max(1, n_samples_per_input)
            n_nets = max(1, B // K)
            R = context.gains.shape[1]
            N = x.shape[2]
            device = x.device
            dtype = x.dtype

            # Collapse time axis: average power allocation over T diffusion sub-steps.
            power = ((x[..., 0] + 0.5) * context.p_max).clamp_min(0.0).mean(dim=1)  # [B, N]

            if dual_lambda is None:
                dual_lambda_local = torch.zeros((B, N), device=device, dtype=dtype)
            else:
                dual_lambda_local = dual_lambda.to(device=device, dtype=dtype)
                if dual_lambda_local.shape != (B, N):
                    raise ValueError(
                        "dual_lambda shape mismatch in summary: "
                        f"expected {(B, N)}, got {tuple(dual_lambda_local.shape)}."
                    )
            if inverse_beta_override is not None:
                inverse_beta_batch = inverse_beta_override.to(device=device, dtype=dtype).view(-1)
                if inverse_beta_batch.numel() == 1:
                    inverse_beta_batch = inverse_beta_batch.expand(B)
                elif inverse_beta_batch.numel() != B:
                    raise ValueError(
                        "inverse_beta_override shape mismatch in summary: "
                        f"expected scalar or [{B}], got {tuple(inverse_beta_batch.shape)}."
                    )
            elif t is None:
                inverse_beta_batch = self.inverse_beta_by_t[0].to(
                    device=device, dtype=dtype,
                ).expand(B)
            else:
                t_safe = t.to(device=device, dtype=torch.long).clamp(0, self.num_timesteps - 1)
                inverse_beta_batch = self._gather(
                    self.inverse_beta_by_t.to(device=device), t_safe,
                ).to(dtype=dtype)

            sum_rates_list: List[float] = []
            p5_rates_list: List[float] = []
            energy_list: List[float] = []
            for n in range(n_nets):
                lo, hi = n * K, min((n + 1) * K, B)
                k = hi - lo
                # For each channel realization, pick one policy uniformly at random.
                idx = torch.randint(0, k, (R,), device=device)
                power_r = power[lo:hi][idx]  # [R, N]
                gains_n = context.gains[lo]   # [R, N, N]
                noise_n = float(context.noise_var[lo].clamp_min(1e-20).mean().item())
                r_min_n = float(context.r_min[lo].item())

                received = torch.einsum("rij,ri->rj", gains_n, power_r)  # [R, N]
                direct_gain = torch.diagonal(gains_n, dim1=1, dim2=2)    # [R, N]
                direct_signal = power_r * direct_gain                     # [R, N]
                interference = (received - direct_signal).clamp_min(0.0)
                sinr = direct_signal.clamp_min(0.0) / (interference + noise_n)
                rates = torch.log2(1.0 + sinr)   # [R, N]
                ergodic_n = rates.mean(dim=0)     # [N]

                sr = float(ergodic_n.sum().item())
                p5 = float(torch.quantile(ergodic_n, 0.05).item())
                lambda_n = dual_lambda_local[lo:hi].mean(dim=0)  # [N]
                lagrangian = float(((r_min_n - ergodic_n) * lambda_n).sum().item())
                inverse_beta_n = float(inverse_beta_batch[lo:hi].mean().item())
                e_n = float(inverse_beta_n * (-sr + lagrangian))

                sum_rates_list.append(sr)
                p5_rates_list.append(p5)
                energy_list.append(e_n)

            mean_sr = sum(sum_rates_list) / len(sum_rates_list) if sum_rates_list else 0.0
            mean_p5 = sum(p5_rates_list) / len(p5_rates_list) if p5_rates_list else 0.0
            mean_e = sum(energy_list) / len(energy_list) if energy_list else 0.0

        return {
            "sum_rate": mean_sr,
            "p5": mean_p5,
            "energy": mean_e,
            "per_net_sum_rates": sum_rates_list,
            "per_net_p5s": p5_rates_list,
            "per_net_energies": energy_list,
        }

    def _inverse_beta_for_full_backward_pass(
        self,
        *,
        pass_index: int,
        total_passes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if total_passes <= 1:
            t_idx = 0
        else:
            frac = float(pass_index) / float(total_passes - 1)
            t_idx = int(round((self.num_timesteps - 1) * (1.0 - frac)))
            t_idx = max(0, min(self.num_timesteps - 1, t_idx))
        return self.inverse_beta_by_t[t_idx].to(device=device, dtype=dtype)

    def _score_to_x0_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar.clamp_min(1e-12))
        sqrt_one_minus_alpha_bar = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))

        # score(x_t, t) = -eps / sqrt(1 - alpha_bar_t)
        eps_pred = -score * sqrt_one_minus_alpha_bar
        x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar

        if self.clip_denoised:
            x0_pred = x0_pred.clamp(-0.5, 0.5)
        return x0_pred, eps_pred

    def _clip_mc_candidate(self, x0: torch.Tensor) -> torch.Tensor:
        if self.clip_denoised:
            return x0.clamp(-0.5, 0.5)
        return x0


class EnergyDDPM(_EnergyScoreMixin, DDPM):
    """
    Energy-based DDPM sampler with analytical score estimation (Eq. 8 style).

    Training is intentionally not implemented yet; this class supports sampling only.
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        loss_type: str = "l2",
        parameterization: str = "eps",
        clip_denoised: bool = True,
        cosine_s: float = 0.008,
        compile_model: bool = False,
        *,
        energy_mc_samples: int = 8,
        energy_num_channel_realizations: int = 200,
        dual_num_channel_realizations: Optional[int] = None,
        inverse_beta: float = 1.0,
        inverse_beta_schedule: str = "constant",
        inverse_beta_start: Optional[float] = None,
        inverse_beta_end: Optional[float] = None,
        dual_update_mode: str = "x0_pred",
        dual_step_size: float = 1.0,
        dual_step_size_schedule: str = "constant",
        dual_step_size_start: Optional[float] = None,
        dual_step_size_end: Optional[float] = None,
        dual_num_outer_iterations: int = 1,
        dual_lambda_init: float = 0.0,
        dual_lambda_max: Optional[float] = None,
        dual_lambda_decay: float = 0.0,
        dual_lambda_mode: str = "shared_per_network",
        langevin_step_size: float = 1e-4,
        langevin_noise_scale: float = 1.0,
        dataset_root: Optional[str] = None,
        default_p_max: float = 1.0,
        default_noise_var: float = 1e-12,
        default_r_min: float = 0.5,
        r_min_override: Optional[float] = None,
        min_energy_sigma: float = 1e-4,
        use_precomputed_channels: bool = True,
        allow_sparse_graph_fallback: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            loss_type=loss_type,
            parameterization=parameterization,
            clip_denoised=clip_denoised,
            cosine_s=cosine_s,
            compile_model=compile_model,
            **kwargs,
        )
        self._init_energy_params(
            energy_mc_samples=energy_mc_samples,
            energy_num_channel_realizations=energy_num_channel_realizations,
            dual_num_channel_realizations=dual_num_channel_realizations,
            inverse_beta=inverse_beta,
            inverse_beta_schedule=inverse_beta_schedule,
            inverse_beta_start=inverse_beta_start,
            inverse_beta_end=inverse_beta_end,
            dual_update_mode=dual_update_mode,
            dual_step_size=dual_step_size,
            dual_step_size_schedule=dual_step_size_schedule,
            dual_step_size_start=dual_step_size_start,
            dual_step_size_end=dual_step_size_end,
            dual_num_outer_iterations=dual_num_outer_iterations,
            dual_lambda_init=dual_lambda_init,
            dual_lambda_max=dual_lambda_max,
            dual_lambda_decay=dual_lambda_decay,
            dual_lambda_mode=dual_lambda_mode,
            langevin_step_size=langevin_step_size,
            langevin_noise_scale=langevin_noise_scale,
            dataset_root=dataset_root,
            default_p_max=default_p_max,
            default_noise_var=default_noise_var,
            default_r_min=default_r_min,
            r_min_override=r_min_override,
            min_energy_sigma=min_energy_sigma,
            use_precomputed_channels=use_precomputed_channels,
            allow_sparse_graph_fallback=allow_sparse_graph_fallback,
        )

    def training_loss(self, data: Data) -> torch.Tensor:
        raise NotImplementedError(
            "EnergyDDPM training is not implemented yet. This class currently supports sampling only."
        )

    def _run_single_backward_pass(
        self,
        *,
        x_init: torch.Tensor,
        context: _EnergySamplerContext,
        dual_lambda: torch.Tensor,  # [B, N]
        n_samples_per_input: int,
        return_diffusion_rate_trace: bool,
        update_dual_each_timestep: bool,
        show_progress: bool,
        progress_desc: str,
        inverse_beta_override: Optional[torch.Tensor] = None,  # [B] or scalar
        dual_context: Optional[_EnergySamplerContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        batch_size = int(x_init.shape[0])
        x_t = x_init.clone()
        dual_lambda_work = dual_lambda
        dual_ctx = dual_context if dual_context is not None else context

        trace_sum_rates: List[float] = []
        trace_p5_rates: List[float] = []
        trace_energies: List[float] = []
        trace_per_net_sum_rates: List[List[float]] = []
        trace_per_net_p5s: List[List[float]] = []
        trace_per_net_energies: List[List[float]] = []
        trace_per_net_dual_lambdas: List[List[List[float]]] = []
        trace_per_net_dual_lambda_stats: List[List[Dict[str, List[float]]]] = []
        trace_dual_lambda_mean: List[float] = []
        trace_dual_lambda_max: List[float] = []

        iterator: Any = reversed(range(self.num_timesteps))
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc=progress_desc, unit="step")

        for t_int in iterator:
            t = torch.full((batch_size,), t_int, device=x_t.device, dtype=torch.long)
            score = self._estimate_score(
                x_t=x_t,
                t=t,
                context=context,
                dual_lambda=dual_lambda_work,
                inverse_beta_override=inverse_beta_override,
            )
            x0_pred, _ = self._score_to_x0_eps(x_t=x_t, t=t, score=score)

            if update_dual_each_timestep:
                if self.dual_lambda_mode == "shared_per_network":
                    ergodic_rates = self._time_shared_ergodic_rates_broadcast(
                        x=x0_pred,
                        context=dual_ctx,
                        n_samples_per_input=n_samples_per_input,
                    )
                else:
                    ergodic_rates, _ = self._ergodic_rates_from_samples(
                        x=x0_pred, context=dual_ctx,
                    )
                dual_lambda_work = self._dual_ascent_step(
                    dual_lambda=dual_lambda_work,
                    ergodic_rates=ergodic_rates,
                    context=dual_ctx,
                    t=t,
                )

            mean = self._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t)
            if t_int > 0:
                var = self._gather(self.posterior_variance, t).view(-1, 1, 1, 1)
                x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
            else:
                x_t = mean

            if return_diffusion_rate_trace:
                t_summary_int = t_int - 1 if t_int > 0 else 0
                t_summary = torch.full(
                    (batch_size,),
                    t_summary_int,
                    device=x_t.device,
                    dtype=torch.long,
                )
                summary = self._rates_summary_from_samples(
                    x=x_t,
                    context=dual_ctx,
                    n_samples_per_input=n_samples_per_input,
                    dual_lambda=dual_lambda_work,
                    t=t_summary,
                    inverse_beta_override=inverse_beta_override,
                )
                trace_sum_rates.append(summary["sum_rate"])
                trace_p5_rates.append(summary["p5"])
                trace_energies.append(summary["energy"])
                trace_per_net_sum_rates.append(summary["per_net_sum_rates"])
                trace_per_net_p5s.append(summary["per_net_p5s"])
                trace_per_net_energies.append(summary["per_net_energies"])
                trace_per_net_dual_lambdas.append(
                    self._per_network_dual_lambda_lists(
                        dual_lambda=dual_lambda_work,
                        n_samples_per_input=n_samples_per_input,
                    )
                )
                trace_per_net_dual_lambda_stats.append(
                    self._per_network_dual_lambda_stats(
                        dual_lambda=dual_lambda_work,
                        n_samples_per_input=n_samples_per_input,
                    )
                )
                trace_dual_lambda_mean.append(float(dual_lambda_work.mean().item()))
                trace_dual_lambda_max.append(float(dual_lambda_work.max().item()))

        if not return_diffusion_rate_trace:
            return x_t, dual_lambda_work, None

        rate_trace = {
            "sum_rates": trace_sum_rates,
            "p5_rates": trace_p5_rates,
            "energies": trace_energies,
            "per_net_sum_rates": trace_per_net_sum_rates,
            "per_net_p5s": trace_per_net_p5s,
            "per_net_energies": trace_per_net_energies,
            "per_net_dual_lambdas": trace_per_net_dual_lambdas,
            "per_net_dual_lambda_stats": trace_per_net_dual_lambda_stats,
            "dual_lambda_mean": trace_dual_lambda_mean,
            "dual_lambda_max": trace_dual_lambda_max,
        }
        return x_t, dual_lambda_work, rate_trace

    def _run_single_langevin_pass(
        self,
        *,
        x_init: torch.Tensor,
        context: _EnergySamplerContext,
        dual_lambda: torch.Tensor,  # [B, N]
        n_samples_per_input: int,
        return_diffusion_rate_trace: bool,
        update_dual_each_timestep: bool,
        show_progress: bool,
        progress_desc: str,
        inverse_beta_override: Optional[torch.Tensor] = None,  # [B] or scalar
        dual_context: Optional[_EnergySamplerContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        batch_size = int(x_init.shape[0])
        x_t = x_init.clone()
        dual_lambda_work = dual_lambda
        dual_ctx = dual_context if dual_context is not None else context

        trace_sum_rates: List[float] = []
        trace_p5_rates: List[float] = []
        trace_energies: List[float] = []
        trace_per_net_sum_rates: List[List[float]] = []
        trace_per_net_p5s: List[List[float]] = []
        trace_per_net_energies: List[List[float]] = []
        trace_per_net_dual_lambdas: List[List[List[float]]] = []
        trace_dual_lambda_mean: List[float] = []
        trace_dual_lambda_max: List[float] = []

        iterator: Any = reversed(range(self.num_timesteps))
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc=progress_desc, unit="step")

        step_size = float(self.langevin_step_size)
        noise_std = float(self.langevin_noise_scale) * math.sqrt(2.0 * step_size)
        step_size_t = torch.tensor(step_size, device=x_t.device, dtype=x_t.dtype)

        for t_int in iterator:
            t = torch.full((batch_size,), t_int, device=x_t.device, dtype=torch.long)
            grad_e = self._energy_gradient(
                x=x_t,
                t=t,
                context=context,
                dual_lambda=dual_lambda_work,
                inverse_beta_override=inverse_beta_override,
            )

            x_t = x_t - step_size_t * grad_e
            if t_int > 0 and noise_std > 0.0:
                x_t = x_t + noise_std * torch.randn_like(x_t)
            if self.clip_denoised:
                x_t = x_t.clamp(-0.5, 0.5)

            if update_dual_each_timestep:
                if self.dual_lambda_mode == "shared_per_network":
                    ergodic_rates = self._time_shared_ergodic_rates_broadcast(
                        x=x_t,
                        context=dual_ctx,
                        n_samples_per_input=n_samples_per_input,
                    )
                else:
                    ergodic_rates, _ = self._ergodic_rates_from_samples(
                        x=x_t, context=dual_ctx,
                    )
                dual_lambda_work = self._dual_ascent_step(
                    dual_lambda=dual_lambda_work,
                    ergodic_rates=ergodic_rates,
                    context=dual_ctx,
                    t=t,
                )

            if return_diffusion_rate_trace:
                summary = self._rates_summary_from_samples(
                    x=x_t,
                    context=dual_ctx,
                    n_samples_per_input=n_samples_per_input,
                    dual_lambda=dual_lambda_work,
                    t=t,
                    inverse_beta_override=inverse_beta_override,
                )
                trace_sum_rates.append(summary["sum_rate"])
                trace_p5_rates.append(summary["p5"])
                trace_energies.append(summary["energy"])
                trace_per_net_sum_rates.append(summary["per_net_sum_rates"])
                trace_per_net_p5s.append(summary["per_net_p5s"])
                trace_per_net_energies.append(summary["per_net_energies"])
                trace_per_net_dual_lambdas.append(
                    self._per_network_dual_lambda_lists(
                        dual_lambda=dual_lambda_work,
                        n_samples_per_input=n_samples_per_input,
                    )
                )
                trace_dual_lambda_mean.append(float(dual_lambda_work.mean().item()))
                trace_dual_lambda_max.append(float(dual_lambda_work.max().item()))

        if not return_diffusion_rate_trace:
            return x_t, dual_lambda_work, None

        rate_trace = {
            "sum_rates": trace_sum_rates,
            "p5_rates": trace_p5_rates,
            "energies": trace_energies,
            "per_net_sum_rates": trace_per_net_sum_rates,
            "per_net_p5s": trace_per_net_p5s,
            "per_net_energies": trace_per_net_energies,
            "per_net_dual_lambdas": trace_per_net_dual_lambdas,
            "dual_lambda_mean": trace_dual_lambda_mean,
            "dual_lambda_max": trace_dual_lambda_max,
        }
        return x_t, dual_lambda_work, rate_trace

    def sample(
        self,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        data: Optional[Data] = None,
        use_amp: bool = False,
        return_selector_sampling_diagnostics: bool = False,
        selector_probe_timesteps: Optional[List[int]] = None,
        return_diffusion_rate_trace: bool = False,
        n_samples_per_input: int = 1,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, Any]]]]:
        del use_amp, selector_probe_timesteps  # Unused for analytical-score sampling.
        if data is None:
            raise ValueError("EnergyDDPM.sample requires `data` (test batch) for channel metadata.")

        batch_size, num_nodes = int(shape[0]), int(shape[2])
        x_init = torch.randn(shape, device=device, dtype=torch.float32)
        context = self._build_energy_context(
            data=data,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=device,
            dtype=x_init.dtype,
        )
        dual_context = self._build_dual_context(
            data=data,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=device,
            dtype=x_init.dtype,
        )

        dual_lambda = self._init_dual_lambda(
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=device,
            dtype=x_init.dtype,
            init_value=self.dual_lambda_init,
        )

        if self.dual_update_mode == "x0_pred":
            x_t, dual_lambda, rate_trace = self._run_single_backward_pass(
                x_init=x_init,
                context=context,
                dual_lambda=dual_lambda,
                n_samples_per_input=n_samples_per_input,
                return_diffusion_rate_trace=return_diffusion_rate_trace,
                update_dual_each_timestep=True,
                show_progress=True,
                progress_desc="Energy-DDPM Sampling",
                inverse_beta_override=None,
                dual_context=dual_context,
            )
        elif self.dual_update_mode == "langevin":
            x_t, dual_lambda, rate_trace = self._run_single_langevin_pass(
                x_init=x_init,
                context=context,
                dual_lambda=dual_lambda,
                n_samples_per_input=n_samples_per_input,
                return_diffusion_rate_trace=return_diffusion_rate_trace,
                update_dual_each_timestep=True,
                show_progress=True,
                progress_desc="Energy-DDPM Langevin",
                inverse_beta_override=None,
                dual_context=dual_context,
            )
        else:
            hybrid_mode = self.dual_update_mode == "hybrid"
            mode_label = "hybrid" if hybrid_mode else "full_backward"
            inverse_beta_hybrid_const = torch.tensor(
                float(self.inverse_beta), device=x_init.device, dtype=x_init.dtype,
            )
            warmup_trace: Optional[Dict[str, Any]] = None
            if hybrid_mode:
                tqdm.tqdm.write(
                    "[Energy-DDPM hybrid] running x0_pred warm-start for dual variable."
                )
                _, dual_lambda, warmup_trace = self._run_single_backward_pass(
                    x_init=x_init,
                    context=context,
                    dual_lambda=dual_lambda,
                    n_samples_per_input=n_samples_per_input,
                    return_diffusion_rate_trace=return_diffusion_rate_trace,
                    update_dual_each_timestep=True,
                    show_progress=True,
                    progress_desc="Energy-DDPM Hybrid Warmup",
                    inverse_beta_override=inverse_beta_hybrid_const,
                    dual_context=dual_context,
                )
            outer_trace: Dict[str, List[Any]] = {
                "sum_rates": [],
                "p5_rates": [],
                "energies": [],
                "inverse_beta": [],
                "per_net_sum_rates": [],
                "per_net_p5s": [],
                "per_net_energies": [],
                "per_net_dual_lambdas": [],
                "dual_lambda_mean": [],
                "dual_lambda_max": [],
                "per_pass_diffusion_traces": [],
            }
            total_passes = int(self.dual_num_outer_iterations) + 1
            for outer_idx in tqdm.tqdm(
                range(self.dual_num_outer_iterations),
                desc="Energy-DDPM Dual Updates",
                unit="iter",
            ):
                if hybrid_mode:
                    inverse_beta_outer = inverse_beta_hybrid_const
                else:
                    inverse_beta_outer = self._inverse_beta_for_full_backward_pass(
                        pass_index=outer_idx,
                        total_passes=total_passes,
                        device=x_init.device,
                        dtype=x_init.dtype,
                    )
                tqdm.tqdm.write(
                    f"[Energy-DDPM {mode_label}] "
                    f"outer {outer_idx + 1}/{self.dual_num_outer_iterations}: "
                    f"inverse_beta={float(inverse_beta_outer.item()):.6g}"
                )
                x_candidate, _, pass_trace = self._run_single_backward_pass(
                    x_init=x_init,
                    context=context,
                    dual_lambda=dual_lambda,
                    n_samples_per_input=n_samples_per_input,
                    return_diffusion_rate_trace=return_diffusion_rate_trace,
                    update_dual_each_timestep=False,
                    show_progress=False,
                    progress_desc="",
                    inverse_beta_override=inverse_beta_outer,
                    dual_context=dual_context,
                )
                dual_ctx_resolved = dual_context if dual_context is not None else context
                ergodic_rates, sum_rates = self._ergodic_rates_from_samples(
                    x=x_candidate, context=dual_ctx_resolved,
                )
                if self.dual_lambda_mode == "shared_per_network":
                    ergodic_rates_for_lam = self._time_shared_ergodic_rates_broadcast(
                        x=x_candidate,
                        context=dual_ctx_resolved,
                        n_samples_per_input=n_samples_per_input,
                    )
                else:
                    ergodic_rates_for_lam = ergodic_rates
                dual_lambda = self._dual_ascent_step(
                    dual_lambda=dual_lambda,
                    ergodic_rates=ergodic_rates_for_lam,
                    context=dual_ctx_resolved,
                    t=torch.zeros(
                        (batch_size,),
                        device=dual_lambda.device,
                        dtype=torch.long,
                    ),
                )

                if return_diffusion_rate_trace:
                    if pass_trace is not None:
                        outer_trace["per_pass_diffusion_traces"].append(pass_trace)
                    p5_rates = torch.quantile(ergodic_rates, 0.05, dim=1)
                    lagrangian = (
                        (dual_ctx_resolved.r_min.view(-1, 1) - ergodic_rates) * dual_lambda
                    ).sum(dim=1)
                    energies = inverse_beta_outer * (-sum_rates + lagrangian)

                    outer_trace["sum_rates"].append(float(sum_rates.mean().item()))
                    outer_trace["p5_rates"].append(float(p5_rates.mean().item()))
                    outer_trace["energies"].append(float(energies.mean().item()))
                    outer_trace["inverse_beta"].append(float(inverse_beta_outer.item()))
                    outer_trace["per_net_sum_rates"].append(
                        [float(v) for v in sum_rates.detach().cpu().tolist()]
                    )
                    outer_trace["per_net_p5s"].append(
                        [float(v) for v in p5_rates.detach().cpu().tolist()]
                    )
                    outer_trace["per_net_energies"].append(
                        [float(v) for v in energies.detach().cpu().tolist()]
                    )
                    outer_trace["per_net_dual_lambdas"].append(
                        self._per_network_dual_lambda_lists(
                            dual_lambda=dual_lambda,
                            n_samples_per_input=n_samples_per_input,
                        )
                    )
                    outer_trace["dual_lambda_mean"].append(float(dual_lambda.mean().item()))
                    outer_trace["dual_lambda_max"].append(float(dual_lambda.max().item()))

            if hybrid_mode:
                inverse_beta_final = inverse_beta_hybrid_const
            else:
                inverse_beta_final = self._inverse_beta_for_full_backward_pass(
                    pass_index=int(self.dual_num_outer_iterations),
                    total_passes=total_passes,
                    device=x_init.device,
                    dtype=x_init.dtype,
                )
            tqdm.tqdm.write(
                f"[Energy-DDPM {mode_label}] "
                f"final sampling pass inverse_beta={float(inverse_beta_final.item()):.6g}"
            )
            x_t, dual_lambda, rate_trace = self._run_single_backward_pass(
                x_init=x_init,
                context=context,
                dual_lambda=dual_lambda,
                n_samples_per_input=n_samples_per_input,
                return_diffusion_rate_trace=return_diffusion_rate_trace,
                update_dual_each_timestep=False,
                show_progress=True,
                progress_desc="Energy-DDPM Sampling",
                inverse_beta_override=inverse_beta_final,
                dual_context=dual_context,
            )
            if rate_trace is not None:
                rate_trace["outer_dual_updates"] = outer_trace
                rate_trace["full_backward_inverse_beta"] = float(inverse_beta_final.item())
                if hybrid_mode and warmup_trace is not None:
                    rate_trace["hybrid_warmup_trace"] = warmup_trace

        self._last_dual_lambda = dual_lambda.detach()

        if rate_trace is not None:
            rate_trace["final_dual_lambda_mean"] = float(dual_lambda.mean().item())
            rate_trace["final_dual_lambda_max"] = float(dual_lambda.max().item())
            rate_trace["final_dual_lambda"] = dual_lambda.detach().cpu()
            if return_selector_sampling_diagnostics:
                return x_t, None, rate_trace
            return x_t, rate_trace

        if return_selector_sampling_diagnostics:
            return x_t, None
        return x_t
