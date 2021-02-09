# based upon R Tutorial for Knockoffs : https://web.stanford.edu/group/candes/knockoffs/software/knockoffs
#
# Creating an artificial problem Let us begin by creating some
# synthetic data. For simplicity, we will use synthetic data
# constructed from a generalized linear model such that the response
# only depends on a small fraction of the variables.

load_install_packages <- function(list.of.packages) {
    new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages,repos="http://cran.cnr.Berkeley.edu/")
    for (i in list.of.packages) {
        library(i,character.only=T)
    }
}

load_install_packages(c("argparse"))

init_arg <- function() {
    parser <- ArgumentParser(description='')
    parser$add_argument('-o', type="character")
    parser$add_argument('--ojson', type="character")
    parser$add_argument('--target', default='label', type="character")
    return (parser)
}

args <- init_arg()$parse_args()
lbl <- args$target
fn_csv <- args$o
fn_json <- args$ojson

set.seed(12345)

                   # parameters generation of synthetic data
n = 3000           # number of observations
p = 100            # number of variables
k = 10             # number of variables with nonzero coefficients
amplitude = 2*7.5  # signal amplitude

# Generate the variables from a multivariate normal distribution
mu = rep(0,p)
rho = 0.10
Sigma = toeplitz(rho^(0:(p-1))) # diagnal const


X = matrix(rnorm(n*p),n) %*% chol(Sigma)  # rnorm(n, mean = , sd = ) is used to generate n normal random numbers with arguments mean and sd ; , %*% is matrix multiplication

# Generate the response from a logistic model and encode it as a factor.
nonzero = sample(p, k)
beta = amplitude * (1:p %in% nonzero) / sqrt(n)
invlogit = function(x) exp(x) / (1+exp(x))
y.sample = function(x) rbinom(n, prob=invlogit(x %*% beta), size=1)
y = factor(y.sample(X), levels=c(0,1), labels=c("A","B"))

df_syn= as.data.frame(X)
df_syn[, lbl] = y
write.csv(df_syn, file=fn_csv, row.names=FALSE)
features <- names(df_syn)[names(df_syn) != lbl]
result_json = sprintf('{ "features_selected": [\"%s\" ], "features": ["%s"]}',
    paste(features[nonzero], collapse='","'),
    paste(features, collapse='","'))

print(sprintf("# samples:%d, # explanatory variables:%d, relevant variables: # %d) : %s",
              NROW(df_syn),
              NCOL(df_syn) - 1,
              length(nonzero),
              paste(features[nonzero], collapse=',')))
write(result_json, file=fn_json)
