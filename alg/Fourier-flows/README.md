# Fourier-flows

Code for [Generative Time-series Modeling with Fourier Flows](https://openreview.net/forum?id=PpshD0AXfA).

### Installation
* Install with `pip`:
  ```bash
  pip install -r requirements.txt
  ```
* Install with `conda`:
  ```bash
  conda env create --file environment.yml
  ```

For requirements for the GPU-compatible installation, see comments in [`environment.yml`](./environment.yml).

### Data
The public datasets used in the paper are available at:
* https://drive.google.com/drive/folders/1UILaMFnZpRUf_IhOIkxK2wzBjWBTB86G

### Experiments
* For Experiment 1 (Section 5.1), run [`ICLR 2021 - Experiment 1.ipynb`](./ICLR%202021%20-%20Experiment%201.ipynb).
* For Experiment 2 (Section 5.2), run [`run_experiment_2.py`](./run_experiment_2.py).

### Citing
Please cite:
~~~bibtex
@inproceedings{alaa2020generative,
  title={Generative Time-series Modeling with Fourier Flows},
  author={Alaa, Ahmed and Chan, Alex James and van der Schaar, Mihaela},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
~~~
