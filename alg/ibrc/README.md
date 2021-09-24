# Inverse Decision Modeling: Learning Interpretable Representations of Behavior
Code author: Alihan Hüyük ([ah2075@cam.ac.uk](mailto:ah2075@cam.ac.uk))

This repository contains the necessary code to replicate the main experimental results in the ICML 2021 paper '[Inverse Decision Modeling: Learning Interpretable Representations of Behavior](http://proceedings.mlr.press/v139/jarrett21a.html).' *Inverse bounded rational control*, which is given as an example instance of inverse decision modeling in the paper, is implemented in files `diag/main.py` and `adni/main.py` for the decision-making environments considered in the paper, namely DIAG and ADNI.

### Usage

First, install the required python packages by running:
```shell
    python3 -m pip install -r requirements.txt
```
For generating the figures, make sure you have Latex installed
```
sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

Then, the main experimental results in the paper can be replicated by running:
```shell
    ./diag/run.sh
    python3 diag/plot-forward.py  # generates Figure 2
    python3 diag/plot-inverse.py  # generates Figure 3
    python3 diag/eval-irl.py      # computes cost-benefit ratios in Section 5.2

    ./adni/run.sh
    python3 adni/eval.py          # computes estimated values of beta in Section 5.3
```

Note that, in order to run the experiments for ADNI, you need to get access to the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/) dataset.

### Citing
If you use this software please cite as follows:
```
@inproceedings{jarrett2021inverse,
  author={Daniel Jarrett and Alihan H\"uy\"uk and Mihaela van der Schaar},
  title={Inverse decision modeling: learning Interpretable Representations of behavior},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year={2021}
}
```
