# Example of the GCIT applied to the Cancer Cell Line Encyclopedia

## References 
- Barretina, J., Caponigro, G., Stransky, N., Venkatesan, K., Margolin, A. A., Kim, S., ... & Reddy, A. (2012). The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity. Nature, 483(7391), 603.
- Tansey, Wesley, et al. "The holdout randomization test: Principled and easy black box feature selection." arXiv preprint arXiv:1811.00645 (2018).

To run the code you will need to download the following data sets: [Response data](https://www.dropbox.com/s/eb60o4cviblzk5k/response.csv?dl=0), [Mutation data](https://www.dropbox.com/s/pyks91zh4zj466j/mutation.txt?dl=0), [Expression data](https://www.dropbox.com/s/jent2ys5tvgar2f/expression.txt?dl=0)

Pre-processing steps follow the code in https://github.com/tansey/hrt.

## Usage
```
python3 ccle_experiment.py
```
