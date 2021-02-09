# When and How to Lift the Lockdown? Global COVID-19 Scenario Analysis and Policy Assessment using Compartmental Gaussian Processes (NeurIPS 2020)

Compartmental Gaussian Processes (CGP) is a machine learning tool developed to guide government decision-making around measures to prevent the spread of COVID-19.

In addition to accurately modeling COVID-19 mortality trends for countries under their current policy sets, CGP can adaptively tailor forecasts to show the potential impact of specific policy changes, such as reopening schools or workplaces, allowing international travel, or relaxing stay-at-home requirements.

Unlike other spread models, CGP is able to tackle “What if?” policy questions looking into the future and the past. For example, CGP can estimate what would have happened if Italy’s government had waited a week before imposing lockdown measures, or predict what would happen if India were to loosen all existing spread prevention policies.

![Image of CGP](https://i.imgur.com/K0cKhTo.png)

## Usage

To run the code locally, make sure to first install the required python packages specified in `requirements.txt`.

The `run.sh` file contains commands to reproduce the tables and figures in the paper (Note it may take 2-3 days to run the entire study on a standard desktop). 
The results will be written in the `tables` folder.

The implementation of CGP is provided in folder `pyro_model`. Pre-trained models are available in the folder `trained_models`.

## Citation

If you find the software useful, please consider citing the following paper:

```
@inproceedings{cgp2020,
  title={When and How to Lift the Lockdown? Global COVID-19 Scenario Analysis and Policy Assessment using Compartmental Gaussian Processes},
  author={Qian, Zhaozhi and Alaa, Ahmed M and van der Schaar, Mihaela},
  booktitle={Advances in neural information processing systems},
  year={2020}
}
```

## License
Copyright 2020, Zhaozhi Qian.

This software is released under the MIT license unless mentioned otherwise.

The repository also contains the public data used for training and the public benchmarks used for evaluation. The licenses for these external files are also included.   

 