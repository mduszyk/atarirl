
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
# for pytorch deterministic algorithms
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DQN_PARAMS_PROFILE=dqn-adam python dqn_train.py
```

## Evaluate model stored in mlflow
```shell
DQN_MODEL_URI='runs:/f176f5800e1243b2bfd5cc2a7fd2470e/q0_episode_8000' python dqn_eval.py
```

## Alternatives that train faster
- Rainbow DQN (2017): Uses prioritized replay, multi-step returns,
  and distributional RL to speed up learning.
- Ape-X DQN: Uses multiple learners for faster wall-clock time.
