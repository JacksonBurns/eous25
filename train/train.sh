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

# preatrain chemeleon on regression prediction, and then fine tune on classification
chemprop train \
    --smiles-columns clean_smiles \
    -b 256 \
    -i ../data/training.csv \
    -o output_step1_train \
    --from-foundation CheMeleon \
    --ffn-hidden-dim 2048 \
    --ffn-num-layers 0 \
    --target-columns T340_actual T450_actual \
    -t regression \
    -l mse \
    --metrics mse rmse mae r2 \
    --epochs 50 \
    --patience 5 \
    --split-sizes 0.90 0.10 0.00 \
    --data-seed 42 \
    --pytorch-seed 42

chemprop train \
    --smiles-columns clean_smiles \
    -b 256 \
    --molecule-featurizers morgan_count \
    -i ../data/training.csv \
    -o output_step2_train \
    --from-foundation output_step1_train/model_0/best.pt \
    --ffn-hidden-dim 2048 \
    --ffn-num-layers 0 \
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
