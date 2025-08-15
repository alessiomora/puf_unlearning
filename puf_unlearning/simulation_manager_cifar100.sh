#!/usr/bin/bash
## CIFAR-100
rounds_per_run=10
total_rounds=200
iterations=$total_rounds/$rounds_per_run
recovery_rounds_max=30
unl_clients=9
#
## training original model
## to train IID, set alpha=-1
## to train MiT-B0 model, set model="MitB0" and learning_rate=3e-4
for (( i=1; i <= iterations; ++i ))
do
    echo "$i"
    python -m puf_unlearning.main_fl_2 dataset="cifar100" alpha=0.1 retraining=False model="ResNet18" learning_rate=0.1
done

## training the retrained models
## to train IID, set alpha=-1
## to train MiT-B0 model, set model="MitB0" and learning_rate=3e-4
for (( i=1; i <= iterations; ++i ))
do
  for (( i=0; i <= unl_clients; ++i ))
  do
    echo "$i"
    python -m puf_unlearning.main_fl_2 seed=2 dataset="cifar100" alpha=0.1 retraining=True unlearned_cid=[$i] model="ResNet18" learning_rate=0.1
  done
done

# unlearning and recovery phase
## to train IID, set alpha=-1
## to train MiT-B0 model, set model="MitB0" and learning_rate=3e-4
for (( j=1; j <= recovery_rounds_max; ++j ))
do
  for (( i=0; i <= unl_clients; ++i ))
  do
    echo "$i"
    python -m puf_unlearning.main_fl_2 --multirun seed=2  dataset="cifar100" alpha=0.1 learning_rate=0.1 local_epochs=1 total_rounds=1 resume_training=True unlearned_cid=[$i] resuming_after_unlearning.algorithm="puf_special" resuming_after_unlearning.unlearning_lr=2.0 resuming_after_unlearning.unlearning_epochs=1 model="ResNet18"
    python -m puf_unlearning.main_fl_2 --multirun seed=2  dataset="cifar100" alpha=0.1 learning_rate=0.1 local_epochs=1 total_rounds=1 resume_training=True unlearned_cid=[$i] resuming_after_unlearning.algorithm="puf_regular" resuming_after_unlearning.unlearning_lr=20.0 resuming_after_unlearning.unlearning_epochs=1 model="ResNet18"
  done
done

