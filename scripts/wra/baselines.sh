micromamba run -n gdiff python scripts/wra/diffusion/wra_baselines.py \
    --methods pdi,pdl,pdm,dps \
    --dataset wra_medium_outdoor_high_density \
    --pdi-net-checkpoints \
        "outputs/wireless_resource_allocation-wra/score_net_train/2026-05-18_22-08-09 - energy_portgnn_normal/best_model.pt" \
        "outputs/wireless_resource_allocation-wra/score_net_train/2026-05-19_00-18-19 - energy_portgnn_ddim_warmup/best_model.pt" \
    --dt-checkpoints \
        "outputs/wireless_resource_allocation-wra/score_net_train/2026-05-23_12-52-11 - dual_update_per_batch_fast_lam10/checkpoints/outer_01800.pt" \
        "outputs/wireless_resource_allocation-wra/score_net_train/2026-05-23_12-52-11 - dual_update_per_batch_fast_lam10/checkpoints/outer_02200.pt" \
        "outputs/wireless_resource_allocation-wra/score_net_train/2026-05-23_12-52-11 - dual_update_per_batch_fast_lam10/checkpoints/outer_02600.pt" \
        "outputs/wireless_resource_allocation-wra/score_net_train/2026-05-23_12-52-11 - dual_update_per_batch_fast_lam10/checkpoints/outer_03000.pt" \
    --pdi-net-hidden 256 \
    --pdi-net-layers 8 \
    --pdi-net-K 2 \
    --inverse-beta 1.0 \
    --dual-step-size 0.05 \
    --dual-lambda-init 10.0 \
    --energy-mc-samples 8 \
    --num-channel-realizations 50 \
    --n-samples-per-input 200 \
    --max-batches 64 \
    --sub-batch 20 \
    --chunk-size 50 \
    --eval-timeslots 500 \
	--pdl-num-iters 500 \
	--pdl-primal-lr 1e-4 \
	--pdl-dual-lr 1.0 \
	--pdl-lambda-init 0.0 \
    --num-evolution-trials 50 \
    --dual-lambda-decay 0.0 \
    --label pdi_net_baselinesK200