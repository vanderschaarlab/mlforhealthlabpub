load_install_packages <- function(list.of.packages) {
    new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages,repos="http://cran.cnr.Berkeley.edu/")
    for (i in list.of.packages) {
        print(sprintf("loading %s",i))
        library(i,character.only=T)
    }
}

package_lst <- c(
    "coin",
    "missForest",
    "knockoff",
    "glmnet",
    "ranger",
    "Xmisc",
    "doMC",
    "CoxBoost",
    "randomForestSRC",
    "party",
    "softImpute",
    "ranger",
    'IRkernel'
    )

load_install_packages(package_lst)
