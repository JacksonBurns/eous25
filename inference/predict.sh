mkdir -p output

chemprop predict \
    --smiles-columns clean_smiles \
    -b 256 \
    --molecule-featurizers morgan_count \
    -i ../data/testing.csv \
    -o output/pred.csv \
    --model-paths ../train/output_step2_train

python prepare_submission.py
