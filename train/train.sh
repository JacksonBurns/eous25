# chemprop hpopt \
#     --smiles-columns clean_smiles \
#     --molecule-featurizers morgan_count \
#     -i ../data/training.csv \
#     -o hpopt_output \
#     --from-foundation CheMeleon \
#     --target-columns T340 T450 F340450 F480 \
#     -t classification \
#     --class-balance \
#     -l bce \
#     --metrics roc prc accuracy f1 \
#     --epochs 50 \
#     --patience 5 \
#     --split-sizes 0.10 0.02 0.88 \
#     --data-seed 42 \
#     --pytorch-seed 42 \
#     --search-parameter-keywords ffn_hidden_dim ffn_num_layers batch_size init_lr_ratio max_lr final_lr_ratio warmup_epochs \
#     --hpopt-save-dir /datad/jwburns \
#     --raytune-temp-dir /datad/jwburns \
#     --raytune-num-samples 128 \
#     --raytune-max-concurrent-trials 8 \
#     --raytune-num-gpus 8 \
#     --raytune-use-gpu \
#     --raytune-search-algorithm optuna

chemprop train \
    --smiles-columns clean_smiles \
    -b 256 \
    --molecule-featurizers morgan_count \
    -i ../data/training.csv \
    -o output \
    --from-foundation CheMeleon \
    --ffn-hidden-dim 2048 \
    --ffn-num-layers 1 \
    --target-columns T340 T450 F340450 F480 \
    -t classification \
    --class-balance \
    -l bce \
    --metrics roc prc accuracy f1 \
    --epochs 50 \
    --patience 5 \
    --split-sizes 0.90 0.10 0.00 \
    --data-seed 42 \
    --pytorch-seed 42 \
    --num-replicates 4
