# Causal Confusion in Imitation Learning
This is the code accompanying the paper:
"[Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979)"
by Pim de Haan, Dinesh Jayaraman and Sergey Levine, published at NeurIPS 2019.
See the [website](https://sites.google.com/view/causal-confusion) for a video presentation of the work.

This simplified code implements the graph conditioned policy learning and intervention by policy execution for the MountainCar environment.
Code for the other environments and intervention modes may be published at a later stage.

For questions or comments, feel free to submit an issue.

## Dependencies
Assumes machines with CUDA 10. For machine without GPU or different CUDA versions, you may need to tweak the pytorch and tensorflow dependency.

Full dependency setup:
```
conda env create
```
Or by hand:
```
conda env create -n causal-confusion python=3.6
conda activate causal-confusion
conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 ignite -c pytorch
conda install tensorflow-gpu==1.14 mpi4py scikit-learn
pip install git+https://github.com/pimdh/baselines@no-mujoco
```
Note I reference to a modified version of OpenAI baselines, as the provide pickle of the MountainCar expert does not work with the upstream version.
Also, I modified Baselines' `setup.py` to remove the Mujoco dependency, to allow for easier setup.

## Usage
First generate demonstrations:
```
python -m ccil.gen_data
```

To show causal confusion with simple behaviour cloning agent on original and confounded state:
```
python -m ccil.imitate original simple
python -m ccil.imitate confounded simple
```

To train graph-parametrized policy on confounded state with uniform graph sampling:
```
python -m ccil.imitate confounded uniform --save
```
To train graph-parametrized policy on confounded state with variational causal discovery (2^N categorical distribution):
```
python -m ccil.imitate confounded combinatorial --save
```

To perform intervention by policy execution:
```
python -m ccil.intervention_policy_execution --num_its 10
```
Optionally, setting the `DATA_PATH` environment variable allows one to change the location of data files from the default `./data`.
