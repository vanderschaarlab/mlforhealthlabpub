# DeepHit
Title: "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks"

Authors: Changhee Lee, William R. Zame, Jinsung Yoon, Mihaela van der Schaar

- Reference: C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, "Yoon, J. Jordon, M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks," AAAI Conference on Artificial Intelligence (AAAI), 2018
- Paper: http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
- Supplementary: http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit_Appendix

### Description of the code
This code shows the modified implementation of DeepHit on Metabric (single risk) and Synthetic (competing risks) datasets.

The detailed modifications are as follows:
- Hyper-parameter opimization using random search is implemented
- Residual connections are removed
- The definition of the time-dependent C-index is changed; please refer to T.A. Gerds et al, "Estimating a Time-Dependent Concordance Index for Survival Prediction Models with Covariate Dependent Censoring," Stat Med., 2013
- Set "EVAL_TIMES" to a list of evaluation times of interest for optimizating the network with respect these evaluation times.


### Note
This implementation reports the time-dependent concordance index (C-index) that
is defined in (T. Gerds, 2013) as a measure of discriminative
performance instead of that in (L. Antolini, 2005) which was
originally reported in our paper. In particular, the time-dependent
C-index in (L. Antollini, 2005) reports a single value by averaging
the discriminative performance of a survival model over time
horizons. However, the time-dependent C-index in (T. Gerds, 2013)
provides different values based on the evaluation times and, thus,
provides additive value if one is interested in specific (or
prescribed) time horizons.

References:
T. Gerds et al, "Estimating a Time-Dependent Concordance Index for Survival Prediction Models with Covariate Dependent Censoring," Stat Med., 2013.
L. Antolini et al, "A Time-Dependent Discrimination Index for Survival Data," Stat Med., 2005.
