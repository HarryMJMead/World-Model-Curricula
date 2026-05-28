# World Model Curricula

Dynamically-controlled Model Generation (DyMGen) is a controllable world model architecture that allows adversary-controlled generation of the environment. 

## Installation

A docker environment can be built using the command

```bash
make build
```

and the container can be run using

```bash
make run
```

## Experiments
The training data folder contains the dataset used to train the world model.

In the *world_model_key.yaml* config file, the *dataset_path* needs to be set to the path ot the *training_data* folder.

The encoder/decoder can be trained with the command

```bash
python train_encoder_decoder.py lr=2e-4 vary_z_noise=True noise_z=0.05 batch_size=64 use_frozen_encoder=False
```

Once the encoder/decoder is trained, in the same *world_model_key.yaml* config file, the *encoder_path* needs to be set to the encoder/decoder checkpoint. Then the world model can be trained using the command

```bash
python hybrid_key_world_model_training.py noise_z=0.01
```

Finally, in the *world_model/key_wm.yaml* config file, the *checkpoint_dir* needs to be set to the trained world model checkpoint. Then, the student can be trained using the command

```bash
python maze_world_model_ued.py
```

To train using a randomly controlled world model, the *random_gen_actions* flag can be set to True.

```bash
python maze_world_model_ued.py random_gen_actions=True
```