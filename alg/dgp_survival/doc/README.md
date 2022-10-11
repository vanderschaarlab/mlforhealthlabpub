## Deep Gaussian Processes for Survival Analysis

To run the "test_model.py" script, use the following commands:

python test_model.py -z "Prediction horizon" -n "number of epochs" -t "number of iterations" -b "batch size" -lr "learning rate" -c "number of causes" -d "number of hidden dimensions" -i "number of inducing points"

It is recommended to use the default settings. The expected results for the "Breast cancer cause" after running the scripts are C-index: 0.758 and AUC-ROC: 0.820.
