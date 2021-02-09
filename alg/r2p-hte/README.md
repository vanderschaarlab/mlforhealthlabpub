# Robust Recursive Partitioning for Heterogeneous Treatment Effects with Uncertainty Quantification
Python implementation of the robust recursive partitioning method for heterogeneous treatment effects (R2P-HTE).

Authors: Hyun-Suk Lee, Yao Zhang, William Zame, Cong Shen, Jang-Won Lee, and Mihaela van der Schaar

Paper Link: https://arxiv.org/abs/2006.07917

## Requirements
This implementation is based on Python 3.6 (requirements.txt is included). It requires the causal multi-task Gaussian process (CMGP) model in [1,[url](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/causal_multitask_gaussian_processes_ite)]
 and the conformal prediction framework [2,[url](https://github.com/donlnz/nonconformist)].
The data generating function of the synthetic datasets (SYNTH_A and SYNTH_B) is implemented within this implementation.
The exogenous semi-synthetic datasets (i.e., the IHDP and CPP datasets introduced in [3] and [4], respectively) are required. 

## Usage
```
run_experiment.py --data DATA [--file_path FILE_PATH] [--max_depth MAX_DEPTH] [--min_size MIN_SIZE] [--miscoverage MISCOVERAGE] [--weight WEIGHT] [--gamma GAMMA]
```
Required argument:
*  --data: types of dataset {SYNTH_A, SYNTH_B, IHDP, CPP}

Optional arguments:
*  --file_path: file path of dataset (for IHDP and CPP datasets)
*  --max_depth: maximum depth of partition (-1 for no limits)
*  --min_size: minimum number of samples for each subgroup
*  --miscoverage: target miscoverage rate
*  --weight: weight parameter (lambda)
*  --gamma: regularization parameter (gamma)
                        
Example 
```train
python run_experiment.py --data SYNTH_B
```
                        
## Results
This implementation produces the results of R2P in Table 1 and Table 2 in the paper.

## References
1. A. M. Alaa and M. van der Schaar, "Bayesian inference of individualized treatment effects using multi-task Gaussian processes," NIPS, 2017
1. V. Vovk, A. Gammerman, and G. Shafer, "Algorithmic learning in a random world," Springer Science & Business Media, 2005.
1. J. L. Hill, "Bayesian nonparametric modeling for causal inference," Journal of Computational and Graphical Statistics, 20(1), 217-240, 2011.
1. V. Dorie, J. Hill, U. Shalit, M. Scott, and D. Cervone, "Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition," Statistical Science, 34(1), 43-68, 2019.