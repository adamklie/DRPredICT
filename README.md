# DRPredICT (<u>D</u>rug <u>R</u>esponse <u>Pred</u>ictor of <u>I</u>mmune <u>C</u>heckpoint <u>T</u>herapy)

## Requirements

### Set up a conda environment with required packages

 - pytorch
 - pytorch-lightning
 - pytorch-geometric
 - optuna
 - torchvision
 - ipython
 
#### Option 1: From requirements file
```bash
conda create --name <ENV> --file environment.txt
```
#### Option 2: Clone someone's environment who has it working
As an example `/cellar/users/aklie/opt/miniconda3/envs/pytorch_dev2/`
```bash
conda create --name <ENV> --clone /cellar/users/aklie/opt/miniconda3/envs/pytorch_dev2/
```
And activate it using:
```bash
source activate <ENV>
```

## Directory organization

- bin: 
- config: config files
- data:
- drpredict:
- fit:
- notebooks:
- test: