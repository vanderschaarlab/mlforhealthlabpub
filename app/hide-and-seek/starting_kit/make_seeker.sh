#!/bin/bash

# A helper script for zipping up a seeker solution.

# To run this, from the starting_kit directory execute:
# $ bash make_seeker.sh

ZIP_FILE_NAME="seeker_$USER.zip"  # <-- Change the zip file name here if desired.



# ----------------------------------------------------------------------------------------------------------------------

echo ""; echo "This script will zip up a seeker solution."

STARTING_KIT_DIR="$(dirname $(readlink -f $0))"
echo ""; echo "Starting kit directory: $STARTING_KIT_DIR"
cd $STARTING_KIT_DIR

rm -f $ZIP_FILE_NAME
zip -r $ZIP_FILE_NAME \
    seeker.py \
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
