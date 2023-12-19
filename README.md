# Procgen

This is a repo for my submissions to the NeurIPS 2020 Procgen competition. My solution incorporated a lot of experimentation. Effective strategies included sparse and noisy pseudo-ensembling, reward shaping, action masking, entropy based exploration strategies, curiosity bonuses, framestacking, continuous life accumulation, episode checkpointing, and network hyperparameter tuning.

![](https://raw.githubusercontent.com/openai/procgen/master/screenshots/procgen.gif)

# 🔧 Installation

```
pip install ray[rllib]==0.8.4
pip install tensorflow==2.1.0 # or tensorflow-gpu
pip install procgen==0.10.2
```

# Train

```
./run.sh --train
```

# Rollout

```
./run.sh --rollout
```
