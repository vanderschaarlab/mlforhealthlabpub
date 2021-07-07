#!/bin/bash

set -e

echo "github workspace $GITHUB_WORKSPACE"
# Python
python3 -m pip install --upgrade pip
python3 -m pip install setuptools wheel twine auditwheel

# Publish
python3 -m pip wheel . -w dist/ --no-deps
auditwheel repair dist/*.whl --plat $AUDITWHEEL_PLAT
twine upload --skip-existing wheelhouse/*
