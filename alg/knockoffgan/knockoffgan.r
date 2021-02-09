
load_install_packages <- function(list.of.packages) {
    new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages,repos="http://cran.cnr.Berkeley.edu/")
    for (i in list.of.packages) {
        library(i,character.only=T)
    }
}

load_install_packages(c("knockoff", "argparse", "rjson", "ranger", "doMC"))

init_arg <- function() {
    parser <- ArgumentParser(description='knockoff gan')
    parser$add_argument('--it', type="integer", default=2000, help="number of GAN iterations")
    parser$add_argument('--target', type="character", help="column name with target/response values")
    parser$add_argument('-i', type="character", help="input data in csv format")
    parser$add_argument('-o', type="character", help="json output file name")
    parser$add_argument('--fdr', type="double", default=0.1)
    parser$add_argument('--usegan', type="integer", default=1)
    parser$add_argument('--scale', type="integer", default=1)
    parser$add_argument('--verbose', type="integer", default=1)
    parser$add_argument('--exe', type="character", default="python3", help="name of python interpreter, typically python3 or python" )
    parser$add_argument('--projdir', type="character")
    parser$add_argument(
        '--replication', 
        type="integer",
        default=20,
        help="number of evaluations over which the results are collected")
    parser$add_argument('--stat', type="character", default='rf', help="Importance statistics based on .. random forest (rf), glmnet_coefdiff (glm)")
    return (parser)
}


get_selected_features <- function(selected_lst, max_it) {
    selected_a <- matrix(0, nrow = 1, ncol=dim(X)[2])
    for (i in 1:length(selected_lst)) {
        if (use_verbose) {
            print(sprintf("%d) %s", i, paste(selected_lst[[i]], collapse=" ")))
        }
        selected_a[1, selected_lst[[i]]] <- selected_a[1,selected_lst[[i]]] + 1
    }
    if (use_verbose) {
        print("")
    }
    return (which(selected_a[1,] > (max_it/2)))
}


args <- init_arg()$parse_args()
python_exe <- args$exe
niter <- args$it
fn_src <- args$i
lbl <- args$target
fdr_threshold <- args$fdr
use_gan <- args$usegan
use_scale <- args$scale
use_verbose <- args$verbose
num_iterations <- args$replication
stat_type <- args$stat
fn_json <- args$o
proj_dir <- args$projdir

df = read.csv(fn_src)

stopifnot(is.element(lbl, names(df)))

features = names(df)


features = features[features != lbl]
X = as.matrix(df[, features])
y = as.matrix(df[, lbl])

num_categories = length(unique(y))
if (num_categories != 2) {
    print(sprintf("error: target %s is not binary: %d): %s",
                  lbl,
                  num_categories,
                  paste0(unique(y), collapse=" ")))
}

stopifnot(num_categories == 2)

nfeatures = dim(X)[2]

selected_lst <- list()
fn_knockoff <- tempfile()
if (!dir.exists(dirname(fn_knockoff))) {
   dir.create(dirname(fn_knockoff))
}
for (it in 1:num_iterations) {
    if (!use_gan) {
        mu = rep(0, nfeatures)
        rho = 0.10
        Sigma = toeplitz(rho^(0:(nfeatures-1))) # diagnal const
        diag_s = create.solve_asdp(Sigma)
        X_k = create.gaussian(X, mu, Sigma, diag_s=diag_s)
    } else {
        script_bs <- "KnockoffGAN.py"
        if (length(proj_dir)) {
            script <- sprintf("%s/alg/knockoffgan/%s", proj_dir, script_bs)
        } else {
            script <- script_bs
        }
      	cmd <- sprintf("%s %s -i %s -o %s --target %s --it %d --scale %d",
	                python_exe, script, fn_src, fn_knockoff, lbl, niter, use_scale)
        print(cmd)		
        rval <- system(cmd)
        stopifnot(!rval)
        print(sprintf("reading %s", fn_knockoff))
        df_k = read.csv(fn_knockoff)
        X_k = as.matrix(df_k[, names(df_k) != lbl])
    }
    if (stat_type=="rf") {
        W = stat.random_forest(X, X_k, y, family="binomial")
    } else if (stat_type=="glm") {
        W = stat.glmnet_coefdiff(X, X_k, y, family="binomial")
    } else {
        stopifnot(FALSE)
    }
    t = knockoff.threshold(W, fdr=fdr_threshold, offset=1)
    selected = which(W >= t)
    selected_lst[[it]] <- selected
    print(sprintf("%d) selected: %s", it, paste(selected_lst[[it]], collapse=" ")))
    selected_avg <- get_selected_features(selected_lst, it)
    print(sprintf("%d,%d)) average selected: %d) %s",
                  it,
                  num_iterations,
                  length(selected_avg),
                  paste(features[selected_avg], collapse=" ")))
}

selected_final <- get_selected_features(selected_lst, length(selected_lst))

print(
    sprintf(
        "fdr %0.2f #%d,%d,%0.2f%%) (th %0.0f) : %s",
        fdr_threshold,
        length(selected_final), nfeatures, length(selected_final)/nfeatures,
        num_iterations/2,
        paste(features[selected_final], collapse=" ")))

result_json = sprintf('{ "features_selected": [\"%s\" ], "features": ["%s"]}',
    paste(features[selected_final], collapse='","'),
    paste(features, collapse='","'))
if (length(fn_json)) {
    write(result_json, file=fn_json)
}
file.remove(fn_knockoff)
