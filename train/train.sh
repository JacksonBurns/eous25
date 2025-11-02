# use this to tune towards the optimal model
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

# this is set based on the output of the above
chemprop train \
    --init-lr 0.001536679517849679 \
    --max-lr 0.0015679314580728588 \
    --final-lr 6.819161996304235e-05 \
    -b 16 \
    --ffn-hidden-dim 600 \
    --ffn-num-layers 2 \
    --warmup-epochs 5 \
    --smiles-columns clean_smiles \
    --molecule-featurizers morgan_count \
    -i ../data/training.csv \
    -o output \
    --from-foundation CheMeleon \
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
