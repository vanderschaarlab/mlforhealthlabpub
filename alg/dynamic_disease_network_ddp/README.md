# Learning Dynamic and Personalized Comorbidity Networks

Implementation of *Learning Dynamic and Personalized Comorbidity Networks from Event Data using Deep Diffusion Processes* (AISTATS2020) in Python runnable both on CPU and GPU.

## Reference

If you use this code as part of any published research, please acknowledge the following paper:
```
@article{qian2020learning,
  title={Learning Dynamic and Personalized Comorbidity Networks from Event Data using Deep Diffusion Processes},
  author={Qian, Zhaozhi and Alaa, Ahmed M and Bellot, Alexis and Rashbass, Jem and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2001.02585},
  year={2020}
}
```
## Running the Code

### Dependency

The minimum dependency requirements are specified in `requirements.txt`. 

You will need an installation of PyTorch (>=1.3) in addition to the standard data science python packages.
We highly recommend using the conda distributions.

### Project Structure

The model itself is implemented in `models.py`. 
The utility functions related to data ingestion and manipulation are implemented in `data_loader.py`. 
`simulation.py` is the entry point to the simulation. To run the simulation, run
```
python simulation.py
```

The simulated data set is generated using [neurawkes](https://github.com/HMEIatJHU/neurawkes) and is located in folder `data`.
The simulation output is a figure `sim_result.png`. It reproduces the simulation study in the appendix (figure 9). 

## License

This project is licensed under the MIT License.
