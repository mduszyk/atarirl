# Atari RL
This project implements reinforcement learning agents based on Deep Q-Networks (DQN)
and Double DQN for playing Atari 2600 games using the OpenAI Gym environment.
It includes key features such as experience replay, target networks,
ε-greedy exploration, and support for multiple Atari titles.

## Create conda environment
```shell
mamba env create -f environment.yaml
```

## Run local mlflow
```shell
mlflow ui
```

## Train using params profile
```shell
# for pytorch deterministic algorithms
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python dqn_train.py --profile double-dqn-tuned-adam
```

## Evaluate model stored in mlflow
```shell
python dqn_eval.py --profile double-dqn-tuned-adam --model_uri 'runs:/8d3c31e65a3240eda1e0e92890057f46/q0_episode_22000'
```

## Alternatives that train faster
- Rainbow DQN (2017): Uses prioritized replay, multi-step returns,
  and distributional RL to speed up learning.
- Ape-X DQN: Uses multiple learners for faster wall-clock time.
