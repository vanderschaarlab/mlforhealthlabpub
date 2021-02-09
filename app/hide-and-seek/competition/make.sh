#!/bin/bash

# Make all the competition related zip archives.

COMP_DIR="$(dirname $(readlink -f $0))"  # Get competition dir.
echo ""; echo "Competition dir: $COMP_DIR"
cd $COMP_DIR

# 1. Scoring program:
echo ""; echo "Making scoring_program.zip..."
rm -f scoring_program.zip
cd ./scoring_program
zip -r $COMP_DIR/scoring_program.zip \
    . \
    -x \*__pycache__\* \
    -x \*data/public_data\*
cd $COMP_DIR

# 2. Public data:
echo ""; echo "Making public_data.zip..."
rm -f public_data.zip
cd ./public_data
zip -r $COMP_DIR/public_data.zip \
    .
cd $COMP_DIR

# 3. Reference data:
echo ""; echo "Making reference_data.zip..."
rm -f reference_data.zip
cd ./reference_data
zip -r $COMP_DIR/reference_data.zip \
    .
cd $COMP_DIR

# 4. Baselines:
echo ""; echo "Making hns_baseline_add_noise.zip..."
rm -f hns_baseline_add_noise.zip
cd ./baselines/hider_add_noise
zip -r $COMP_DIR/hns_baseline_add_noise.zip \
    . \
    -x \*__pycache__\*
cd $COMP_DIR

echo ""; echo "Making hns_baseline_timegan.zip..."
rm -f hns_baseline_timegan.zip
cd ./baselines/hider_timegan
zip -r $COMP_DIR/hns_baseline_timegan.zip \
    . \
    -x \*__pycache__\*
cd $COMP_DIR

echo ""; echo "Making hns_baseline_binary_predictor.zip..."
rm -f hns_baseline_binary_predictor.zip
cd ./baselines/seeker_binary_predictor
zip -r $COMP_DIR/hns_baseline_binary_predictor.zip \
    . \
    -x \*__pycache__\*
cd $COMP_DIR

echo ""; echo "Making hns_baseline_knn.zip..."
rm -f hns_baseline_knn.zip
cd ./baselines/seeker_knn
zip -r $COMP_DIR/hns_baseline_knn.zip \
    . \
    -x \*__pycache__\*
cd $COMP_DIR

# 5. Starting kit:
echo ""; echo "Making starting_kit.zip..."
rm -f starting_kit.zip
cd ../starting_kit
zip -r $COMP_DIR/starting_kit.zip \
    . \
    -x \*__pycache__\* \
    -x \*output\* \
    -x \*.zip \
    -x \*.npz \
    -x \*.csv
cd $COMP_DIR

# 6. Competition bundle:
echo ""; echo "Making bundle.zip..."
rm -f bundle.zip
cd .
zip -r $COMP_DIR/bundle.zip \
    competition.yaml \
    overview.html \
    evaluation.html \
    terms.html \
    data.html \
    logo.jpg \
    scoring_program.zip \
    public_data.zip \
    reference_data.zip \
    starting_kit.zip
cd $COMP_DIR

echo ""; echo "Finished."; echo ""
