CUDA_VISIBLE_DEVICES=1 micromamba run -n gdiff python scripts/wra/diffusion/wra_baselines.py \
    --methods ddim \
    --dataset wra_medium_outdoor_high_density \
    --ddim-checkpoint-dir "outputs/wireless_resource_allocation-wra/ugnn_wra_v3_ds4-ddim_wra-gdm_wra_medium_outdoor_high_density/00c2b045" \
    --n-samples-per-input 200 \
    --ddim-epoch 4920 \
    --max-batches 64 \
    --eval-timeslots 500 \
    --num-evolution-trials 50 \
    --label st_baselines_K200