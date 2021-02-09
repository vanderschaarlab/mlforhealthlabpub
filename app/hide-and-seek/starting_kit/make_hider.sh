#!/bin/bash

# A helper script for zipping up a hider solution.

# To run this, from the starting_kit directory execute:
# $ bash make_hider.sh

ZIP_FILE_NAME="hider_$USER.zip"  # <-- Change the zip file name here if desired.



# ----------------------------------------------------------------------------------------------------------------------

echo ""; echo "This script will zip up a hider solution."

STARTING_KIT_DIR="$(dirname $(readlink -f $0))"
echo ""; echo "Starting kit directory: $STARTING_KIT_DIR"
cd $STARTING_KIT_DIR

rm -f $ZIP_FILE_NAME
zip -r $ZIP_FILE_NAME \
    hider.py \
    Dockerimage \
    utils \
    examples \
    -x \*__pycache__\* \
    -x \*output\* \
    -x \*.zip \
    -x \*.npz \
    -x \*.csv \
    -x \*.sh

echo ""; echo "Finished."; echo ""
