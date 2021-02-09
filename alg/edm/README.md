# [Strictly Batch Imitation Learning by Energy-based Distribution Matching](https://arxiv.org/abs/2006.14154)

### Daniel Jarrett, Ioana Bica,  Mihaela van der Schaar

#### Neural Information Processing Systems (NeurIPS) 2020


## Dependencies

The code was implemented in Python 3.6 and the following packages are needed for running it:

- gym==0.17.2

- numpy==1.18.2

- pandas==1.0.4

- tensorflow==1.15.0

- torch==1.6.0

- tqdm==4.32.1

- scipy==1.1.0

- scikit-learn==0.22.2

- stable-baselines==2.10.1



## Running and evaluating the model:

The environments used for experiments are from OpenAI gym [1]. Each environment is associated with a true reward 
function (unknown to the imitation algorithm). In each case, the “expert” demonstrator can be obtained by using a 
pre-trained and hyperparameter-optimized agent from the RL Baselines Zoo [2] in Stable OpenAI Baselines [3]. 

In this implementation we provide the demonstrations datasets for the CartPole-v1 in 'volume/CartPole-v1'. Note that the 
code in 'contrib/baselines_zoo' was taken from [2]. 
  
To train and evaluate EDM on CartPole-v1, run the following command with the chosen command line arguments. 

```bash
python testing/il.py
```
```
Options :
   --env                  # Environment name. 
   --num_trajectories	  # Number of expert trajectories used for training the imitation learning algorithm. 
   --trial                # Trial number.
```

Outputs:
   - Average reward for 10 repetitions of running EDM.  

#### Example usage

```
python testing/il.py  --env='CartPole-v1' --num_trajectories=10 --trial=0 
```
 

#### References

[1] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang,
and Wojciech Zaremba. Openai gym. OpenAI, 2016

[2] Antonin Raffin. Rl baselines zoo. https://github.com/araffin/rl-baselines-zoo, 2018

[3] Ashley Hill, Antonin Raffin, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, Rene Traore, Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert,
Alec Radford, John Schulman, Szymon Sidor, and Yuhuai Wu. Stable baselines. https://github.com/hill-a/stable-baselines, 2018.

 
### Citation

If you use this code, please cite:

```
@article{jarrett2020strictly,
  title={Strictly Batch Imitation Learning by Energy-based Distribution Matching},
  author={Jarrett, Daniel and Bica, Ioana and van der Schaar, Mihaela},
  journal={Advances in neural information processing systems (NeurIPS)},
  year={2020}
}
```