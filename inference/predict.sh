chemprop predict \
    --test-path ../data/testing.csv \
    --smiles-columns clean_smiles \
    --batch-size 1024 \
    --num-workers 8 \
    --output predictions.csv \
    --model-paths ../train/output_train \
    --molecule-featurizers morgan_count rdkit_2d
