# CASTLE (Causal Structure Learning) regularization

This is sample code for running CASTLE regularization (for regression).  This code attempts to learn a causal DAG to improve predictive performance.  
This code has been inspired by the work of [1,2].

## Requirements 

- Python 3.6+
- `tensorflow`
- `numpy`
- `network`
- `scikit-learn`
- `pandas`

## Contents

- `CASTLE.py` - main regularization file
- `main.py` - runs synthetic experiments (arguments below)
- `utils.py` - includes utils for generating DAGs and synthetic data generation
- `synth_nonlinear.csv` - an example toy file to recreate Table 2 in the main manuscript

## Examples

To run the toy example in Table 2 (Fig. 1 DAG) with 1000 samples use
```bash
$ python main.py --csv synth_nonlinear.csv --dataset_sz 1000
```

To run a custom DAG with 1000 samples, 20 nodes, and a branching factor of 5 use:
```
$ python main.py --random_dag --num_nodes 20 --branchf 5 --dataset_sz 1000
```

## References

[1] Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning (NeurIPS 2018). Source code @ https://github.com/xunzheng/notears

[2] Zheng, X., Dan, C., Aragam, B., Ravikumar, P., & Xing, E. P. (2020). Learning sparse nonparametric DAGs (AISTATS 2020). Source code @ https://github.com/xunzheng/notears
