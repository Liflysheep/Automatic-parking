#!/bin/bash


# Define hyperparameters
gamma=0.99
alpha=0.1
batch_size=128
update_after=500
#lr_decay_period=None
lr_critic=1e-3
lr_actor=1e-3
tau=0.005
q_loss_cls="MSELoss"
#grad_clip=None
adaptive_alpha=True
#target_entropy=None
lr_alpha=1e-3
alpha_optim_cls="Adam"
device="cuda"  # or "cpu" based on your setup
max_episode=20000
learn_freq=200
output_freq=200

# Run the Python script with the parameters
python demo_train_mixed_obs.py \
  --gamma $gamma \
  --alpha $alpha \
  --batch_size $batch_size \
  --update_after $update_after \
  --lr_critic $lr_critic \
  --lr_actor $lr_actor \
  --tau $tau \
  --q_loss_cls $q_loss_cls \
  --adaptive_alpha $adaptive_alpha \
  --lr_alpha $lr_alpha \
  --alpha_optim_cls $alpha_optim_cls \
  --device $device \
  --max_episode $max_episode \
  --learn_freq $learn_freq \
  --output_freq $output_freq