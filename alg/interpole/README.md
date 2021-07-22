# Explaining by Imitating: Understanding Decisions by Interpretable Policy Learning
Code author: Alihan Hüyük ([ah2075@cam.ac.uk](mailto:ah2075@cam.ac.uk))

This repository contains the necessary code to replicate the main experimental results in the ICLR 2021 paper '[Explaining by Imitating: Understanding Decision by Interpretable Policy Learning](https://openreview.net/forum?id=unI5ucw_Jk).' Our proposed method, *Interpole*, is implemented in files `adni/main-interpole.py` and `diag-bias/main-interpole.py` for the decision environments considered in the paper, namely ADNI, DIAG, and BIAS.

### Usage
First, install *pomdp-solve v5.4* inside the empty directory `pomdp/` by following the instructions on [pomdp.org](https://www.pomdp.org/code/index.html). Make sure the executable `pomdp-solve` is located at `pomdp/src/pomdp-solve`. Install the required python packages as well by running:
```shell
    python3 -m pip install -r requirements.txt
```

Then, the experiments in the paper can be replicated by running:
```shell
    ./adni/run.sh            # generates the results for ADNI given in Table 2
    ./diag-bias/run.sh       # generates the results for DIAG given in Table 3
    ./diag-bias/run-bias.sh  # generates the results for BIAS given in Table 4
```

Note: in order to run the `adni` experiment, you need to get access to the  [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/) dataset.

### Citing
If you use this software please cite as follows:
```
@inproceedings{huyuk2021explaining,
  author={Alihan Huyuk and Daniel Jarrett and Cem Tekin and Mihaela van der Schaar},
  title={Explaining by imitating: understanding decisions by interpretable policy learning},
  booktitle={Proceedings of the 9th International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
