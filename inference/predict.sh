mkdir -p output

chemprop predict \
    --smiles-columns clean_smiles \
    -b 256 \
    -i ../data/testing.csv \
    -o output/pred.csv \
    --model-paths ../train/output/
