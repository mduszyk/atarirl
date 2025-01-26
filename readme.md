
## Create conda environment
```shell
mamba env create -f environment.yaml
```

## Run local mlflow
```shell
mlflow ui
```

## Train using params profile
```adam
DQN_PARAMS_PROFILE=adam python dqn_train.py
```

## Evaluate model stored in mlflow
```shell
DQN_MODEL_URI='runs:/f176f5800e1243b2bfd5cc2a7fd2470e/q0_episode_8000' python dqn_eval.py
```
