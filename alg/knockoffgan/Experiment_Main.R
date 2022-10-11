## Main Code Implementations for KnockoffGAN Experiments. (9/27/2018, Jinsung Yoon)


# Reset
rm(list = ls())

### Package install
#install.packages("knockoff")
#install.packages("ranger")
#install.packages("doMC")

load_install_packages <- function(list.of.packages) {
    new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages,repos="http://cran.cnr.Berkeley.edu/")
    for (i in list.of.packages) {
        print(sprintf("loading %s",i))
        library(i,character.only=T)
    }
}

load_install_packages(c("knockoff","glmnet","ranger", "Xmisc","doMC"))

parser <- ArgumentParser$new()

parser$add_argument('-i',type='character')
parser$add_argument('-o',type='character')
parser$add_argument('--xname',type='character', default='Uniform')
parser$add_argument('--yname',type='character', default='Logit')

args <- parser$get_args()

idir <- args$i
odir <- args$o

idir <- paste0(idir, "/")


# Necessary packages
# 1. Knockoff
#library("knockoff")
# 2. LASSO
#library("glmnet")
# 3. Random Forest
#library("ranger")

### Paramters
# 1. Models
model_sets = c('KnockoffGAN')

# 2. X Distributions
x_sets = c('Normal','AR_Normal','Uniform','AR_Uniform')
x_name = x_sets[3]
x_name = args$xname

# 3. Y|X Distributions
y_sets = c('Logit','Gauss')
y_name = y_sets[1]
y_name = args$yname

# 4. Coefficient
if (x_name == 'AR_Normal' || x_name == 'AR_Uniform'){
  coef_set = c(0.0,0.1,0.2,0.3,0.4,0.6,0.8)
} else if (x_name == 'Uniform' && y_name == 'Logit'){
  coef_set = c(1.0,2.0,3.0,4.0,5.0,6.0,7.0)
} else if (x_name == 'Uniform' && y_name == 'Gauss'){
  coef_set = c(0.5,1.0,1.5,2.0,2.5,3.0,3.5)
} else if (y_name == 'Logit'){
  coef_set = c(5.0,6.0,7.0,8.0,9.0,10.0,11.0)
} else if (y_name == 'Gauss'){
  coef_set = c(2.0,2.5,3.0,3.5,4.0,4.5,5.0)
}

# 4. FDR threshold
fdr_thresh = 0.1

# 5. Others
Replication = 100
Coef_No = length(coef_set)
Model_No = length(model_sets)

### Output Initialization
# For all models and all coefficients
Output_TPR = matrix(0,Replication,Coef_No*Model_No)
Output_FDR = matrix(0,Replication,Coef_No*Model_No)

### Iterations (no coefficients when new data)
for (i in 1:Coef_No){
  
  # Print the x and y distribution
  diag = paste0("X: ", x_name, ", Y: ", y_name)
  print(diag)
  
  # select the coefficitions
  coef = coef_set[i]
  
  # For the entire iterations (no replicate when new data)
  for (iter in 1:Replication){
    midfix = '_'
    # midfix = '.0_'

    coef_paste0 = sprintf("%0.1f", coef)
    ## Read the generated X, Y, and Ground truth relevant features G

    # use only one branch when new data
    if (round(coef) == coef){
      ## file_nameX = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Data/",x_name,"_",y_name,"/X_",coef,"_",(iter-1),".csv")
      ## file_nameY = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Data/",x_name,"_",y_name,"/Y_",coef,"_",(iter-1),".csv")
        ## file_nameG = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Data/",x_name,"_",y_name,"/G_",coef,"_",(iter-1),".csv")
      file_nameX = paste0(idir,"/Data/",x_name,"_",y_name,"/X_",coef_paste0,midfix,(iter-1),".csv")  # X of new data
      file_nameY = paste0(idir,"/Data/",x_name,"_",y_name,"/Y_",coef_paste0,midfix,(iter-1),".csv")  # Y of new data
      file_nameG = paste0(idir,"/Data/",x_name,"_",y_name,"/G_",coef_paste0,midfix,(iter-1),".csv")  # this does not exist for new data
    }
    else{
      ## file_nameX = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Data/",x_name,"_",y_name,"/X_",coef,"_",(iter-1),".csv")
      ## file_nameY = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Data/",x_name,"_",y_name,"/Y_",coef,"_",(iter-1),".csv")
      ## file_nameG = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Data/",x_name,"_",y_name,"/G_",coef,"_",(iter-1),".csv")
      file_nameX = paste0(idir,"/Data/",x_name,"_",y_name,"/X_",coef_paste0,midfix,(iter-1),".csv")
      file_nameY = paste0(idir,"/Data/",x_name,"_",y_name,"/Y_",coef_paste0,midfix,(iter-1),".csv")
      file_nameG = paste0(idir,"/Data/",x_name,"_",y_name,"/G_",coef_paste0,midfix,(iter-1),".csv")
    }
    
    print(file_nameX)
    # Read X, Y, G
    X = read.table(file_nameX)
    y = read.csv(file_nameY, header = F)
    g = read.csv(file_nameG, header = F)
    
    # Convert to the matrix
    X = as.matrix(X)
    y = as.matrix(y)
    
    # For each model
    for (k in 1:Model_No){
      
      # Select Model
      model = model_sets[k]
      
      # Print the coefficient, iteration, and model information
      diag = paste0("Coef: ",coef,", Iteration: ",iter, ", Model: ", model)
      print(diag)
      
      # KnockoffGAN
      if (model == 'KnockoffGAN'){
        
        # Read generated knockoff variables
        
        file_feat_selected = paste0(odir,"/Result/",x_name,"_",y_name,"_",k,"_coef_",i,"_it_", iter, "_feat_selected.csv")
        if (file.exists(file_feat_selected)) {
                print(sprintf("%s exist", file_feat_selected))
                s_out = read.csv(file_feat_selected)
                s_out = as.matrix(s_out)

        } else {
        
            if (round(coef) == coef){
              # file_nameX_k = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Knockoff_Data/",x_name,"_",y_name,"/X_",coef,"_",(iter-1),".csv")
              file_nameX_k = paste0(idir,"/Knockoff_Data/",x_name,"_",y_name,"/X_",coef_paste0,midfix,(iter-1),".csv")
            }
            else{
              # file_nameX_k = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Knockoff_Data/",x_name,"_",y_name,"/X_",coef,"_",(iter-1),".csv")
              file_nameX_k = paste0(idir,"/Knockoff_Data/",x_name,"_",y_name,"/X_",coef_paste0,midfix,(iter-1),".csv")
            }
            
            X_k = read.table(file_nameX_k)  # X_k is generated by knockoff gan aka X~
            X_k = as.matrix(X_k)
            
            # Generate the statistics
            if (y_name == 'Logit'){
              W = stat.glmnet_coefdiff(X, X_k, y, nfolds=10, family="binomial")
            } else if (y_name == 'Gauss'){
              W = stat.glmnet_coefdiff(X, X_k, y, nfolds=10, family="gaussian")
            }
            
            # Find the knockoff threshold
            thres = knockoff.threshold(W, fdr=fdr_thresh/2, offset=1)
            
            # Select the features with thres condition
            select_out = as.numeric(which(abs(W) >= thres))
            
            s_out = matrix(0,ncol(X),1)   
            s_out[select_out] = 1
            print(sprintf("# selected features:%d", sum(s_out)))
            ### *********************************************** the most important output (selected features)
            # Output the selected features as the matrix -> s_out is the final output of features selected: 0 mean not selected, 1 means selected, per all samples
    
            # file_feat_selected = paste0(odir,"/Result/",x_name,"_",y_name,"_",k,"_feat_selected.csv")
            write.csv(s_out, file_feat_selected, row.names = F)
          
          # Knockoff
          } 
      }      
      ### Performance metrics when you have a groundtruth
      # 1. TPR
      Output_TPR[iter, i + Coef_No * (k-1)] = sum(s_out * g) / sum(g)
      
      # 2. FDR
      Output_FDR[iter, i + Coef_No * (k-1)] = sum(s_out * (1-g)) / (sum(s_out) + 1e-8)
      
      # Print the current TPR and FDR
      diag = paste0("TPR: ", Output_TPR[iter, i + Coef_No * (k-1)], "FDR: ", Output_FDR[iter, i + Coef_No * (k-1)])
      print(diag)
    }
  }
}

### Write the results
# TPR
# file_tpr = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Result/",x_name,"_",y_name,"_TPR.csv")
file_tpr = paste0(odir,"/Result/",x_name,"_",y_name,"_TPR.csv")
write.csv(Output_TPR, file_tpr, row.names = F)

# FDR
# file_fdr = paste0("/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Result/",x_name,"_",y_name,"_FDR.csv")
file_fdr = paste0(odir,"/Result/",x_name,"_",y_name,"_FDR.csv")
write.csv(Output_FDR, file_fdr, row.names = F)
print(idir)
print(file_fdr)
print(file_tpr)
