mkdir -p output

chemprop predict \
    --smiles-columns clean_smiles \
    -b 256 \
    --molecule-featurizers morgan_count \
    -i ../data/testing.csv \
    -o output/pred_transmittance.csv \
    --model-paths ../train/output_transmittance/model_0/best.pt

chemprop predict \
    --smiles-columns clean_smiles \
    -b 256 \
    --molecule-featurizers morgan_count \
    -i ../data/testing.csv \
    -o output/pred_fluorescence.csv \
    --model-paths ../train/output_fluorescence/model_0/best.pt
