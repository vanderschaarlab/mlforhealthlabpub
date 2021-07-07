#!/bin/bash

set -e

# Python
python3 -m pip install --upgrade pip
python3 -m pip install setuptools wheel twine auditwheel

# Publish
python3 -m pip wheel . -w dist/ --no-deps
twine upload --verbose --skip-existing dist/*
