## Deep Gaussian Processes for Survival Analysis

Deep Multi-task Gaussian Processes for Survival Analysis with Competing Risks [paper](https://papers.nips.cc/paper/6827-deep-multi-task-gaussian-processes-for-survival-analysis-with-competing-risks.pdf)

To run the "dpg.py" script, use the following commands:

```
python3 dpg.py -i <csv>  --target <event> --time <event time> [ -z "Prediction horizon" -n "number of epochs" -t "number of iterations" -b "batch size" -lr "learning rate" -c "number of causes" -d "number of hidden dimensions" --n-inducing "number of inducing points" ]
```

It is recommended to use the default settings.

This sofware uses scikit-survival.

1. Pölsterl, S., Navab, N., and Katouzian, A., Fast Training of Support Vector Machines for Survival Analysis. Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2015, Porto, Portugal, Lecture Notes in Computer Science, vol. 9285, pp. 243-259 (2015)

2. Pölsterl, S., Navab, N., and Katouzian, A., An Efficient Training Algorithm for Kernel Survival Support Vector Machines. 4th Workshop on Machine Learning in Life Sciences, 23 September 2016, Riva del Garda, Italy

3. Pölsterl, S., Gupta, P., Wang, L., Conjeti, S., Katouzian, A., and Navab, N., Heterogeneous ensembles for predicting survival of metastatic, castrate-resistant prostate cancer patients. F1000Research, vol. 5, no. 2676 (2016).
