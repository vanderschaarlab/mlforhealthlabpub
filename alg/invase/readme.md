# Invase

Written by Jinsung Yoon
Date: Jan 1th 2019
INVASE: Instance-wise Variable Selection using Neural Networks Implementation on Synthetic Datasets
Reference: J. Yoon, J. Jordon, M. van der Schaar, "INVASE: Instance-wise Variable Selection using Neural Networks," International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@g.ucla.edu


## Usage

see also tutorial_invase.ipynb

```
python3 invase.py -i <csv> --targe <response var> -o <feature_score.csv.gz>  # calculate feature weighting
python3 invase_plot.py  -i <csv> -oglobal <globalweighting.png> -osample <sampleweighting.png> -isstd 1  # generate plots
```

