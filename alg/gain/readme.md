# Gain

Written by Jinsung Yoon
Date: Jan 29th 2019
Generative Adversarial Imputation Networks (GAIN) Implementation on Spam Dataset
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
[Paper](http://medianetlab.ee.ucla.edu/papers/ICML_GAIN.pdf)
[Appendix](http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf)
Contact: jsyoon0823@g.ucla.edu


## Usage:

```
    python3 test_gain.py  # tests the gain algorithm
    python3 gain.py  # implemention of GAIN, imputes missing data.

    python3 create_missing.py  # creates a csv with missing data.
    python3 gain_ana.py  # analyses the imputed data by calculating the RMSE, be sure to normalize first
```

## example:

```
    python3 create_missing.py --dataset bc -o missing.csv --oref ref.csv --istarget 1 --normalize01 1
    python3 gain.py -i missing.csv -o imputed.csv --target target
    python3 gain_ana.py -i missing.csv --ref ref.csv --imputed imputed.csv -o result.json --target target
```

