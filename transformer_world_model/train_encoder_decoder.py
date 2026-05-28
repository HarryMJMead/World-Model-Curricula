import json
import os
import time
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jaxued.environments.underspecified_env import EnvParams, EnvState, UnderspecifiedEnv
from jaxued.utils import compute_max_mean_returns_epcount, compute_max_mean_returns_epcount_w_idxs
from jax.tree_util import Partial
import optax
from flax import struct, traverse_util
from flax.training.train_state import TrainState
import flax.linen as nn
import distrax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.maze.env import Observation
from jaxued.wrappers import AutoReplayWrapper
import chex

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from networks import WorldModel

logger = logging.getLogger(__name__)

@struct.dataclass
class Dataset:
    obs: Observation
    actions: chex.Array
    rewards: chex.Array
    dones: chex.Array

example_arr = jnp.zeros((0,))
example_dataset = Dataset(
    Observation(image=example_arr, agent_dir=example_arr, has_key=example_arr),
    example_arr,
    example_arr,
    example_arr
)

# region checkpointing
def setup_checkpointing(config: dict) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.
        It also saves the config in `checkpoints/group/run_name/seed/config.json`

    Args:
        config (dict): 
        train_state (TrainState): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 

    Returns:
        ocp.CheckpointManager: 
    """
    overall_save_dir = os.path.join(os.getcwd(), f"checkpoints/{'EncoderDecoder'}", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)
    
    # save the config
    config_dict = OmegaConf.to_container(config, resolve=True)
    with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config_dict, indent=True))
    
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
            enable_async_checkpointing=False,
        )
    )
    
    return checkpoint_manager
#endregion

def main(config=None, project="JAXUED_TEST"):
    wandb_config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=False
        )
    tags = []
    run = wandb.init(config=wandb_config, project=project, tags=tags, group='EncoderDecoder')

    # Match Config Run name and WandB run name
    config['run_name'] = run.name

    wandb.define_metric("num_updates")
    wandb.define_metric("losses/*", step_metric="num_updates")
    wandb.define_metric("training/*", step_metric="num_updates")

    data_recorder = ocp.PyTreeCheckpointer()
    dataset_path = os.path.join(os.getcwd(), config['dataset_path'], config['dataset_name'])
    dataset = data_recorder.restore(
        dataset_path,
        item=example_dataset
    )
    dataset_steps_per_traj, dataset_size = dataset.actions.shape[0:2]

    def linear_schedule(count):
        anneal_percentage = 1 - config['warm_up_percentage']
        anneal_frac = 1/anneal_percentage - count/(config["num_updates"]*anneal_percentage)
        if config['warm_up_percentage'] == 0:
            return config[f"lr"] * anneal_frac
        warm_up_frac = count/(config['warm_up_percentage']*config['num_updates'])
        frac = jnp.minimum(anneal_frac, warm_up_frac)
        return config[f"lr"] * frac

    def log_eval(stats):
        logger.info(f"Logging update: {stats['update_count']}")
        
        num_steps = config['batch_size'] * stats['update_count'] * dataset_steps_per_traj

        # generic stats
        log_dict = {
            "losses/world_model_loss": stats["losses"][0].mean(),
            "losses/reconstruction_loss": stats["losses"][1][0].mean(),
            "losses/latent_entropy": stats["losses"][1][1].mean(),
            "num_updates": stats["update_count"],
            "sps": num_steps / stats['time_delta'],
            "training/grad_norm": stats['grad_norm'].mean(),
            "training/lr": linear_schedule(stats['update_count'])
        }
        
        wandb.log(log_dict)

    def create_train_state(rng):
        rng_batch, rng_init, rng_app = jax.random.split(rng, 3)
        batch_indices = jax.random.randint(
            rng_batch, (config["batch_size"],), 0, dataset_size
        )
        batch = jax.tree_util.tree_map(lambda x: x[:, batch_indices], dataset)
        actions = jnp.pad(batch.actions, ((0, 1), (0, 0)))
        obs = batch.obs.replace(image=batch.obs.image/10)
        
        world_model = WorldModel((5, 5, 3), config, no_dynamics=True)
        wm_params = world_model.init(rng_init, rng_app, obs, actions, None, None, deterministic=True)

        learning_rate = linear_schedule if config[f"anneal_lr"] else config[f"lr"]
        tx = optax.chain(
            optax.clip_by_global_norm(config[f"max_grad_norm"]),
            optax.adamw(learning_rate=learning_rate, eps=config["eps"], weight_decay=config["weight_decay"])
        )

        return TrainState.create(
            apply_fn=world_model.apply,
            params=wm_params,
            tx=tx
        )

    def train_step(runner_state, _):
        rng, train_state, update_count = runner_state

        rng, rng_batch, rng_wm = jax.random.split(rng, 3)
        batch_indices = jax.random.randint(
            rng_batch, (config["batch_size"],), 0, dataset_size
        )
        batch = jax.tree_util.tree_map(lambda x: x[:, batch_indices], dataset)
        
        # Pad the actions, as the obs array contains an extra timestep
        actions = jnp.pad(batch.actions, ((0, 1), (0, 0)))

        # obs are stored as uint8 for space, so need to be normalised
        obs = batch.obs.replace(image=batch.obs.image/10)

        def loss_fn(params):
            (z_dist, recon_obs, l_recon) = train_state.apply_fn(
                params, 
                rng_wm, 
                obs, 
                actions, 
                None, 
                None,
                deterministic=False
            )

            # Reconstruction loss only
            #l_recon = jnp.mean(((recon_image - obs) ** 2).sum(axis=(2,3,4)))
            entropy = z_dist.entropy().mean()

            loss = l_recon - config['entropy_coeff'] * entropy
            return loss, (l_recon, entropy)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        losses, grads = grad_fn(train_state.params)
        grad_norm = optax.global_norm(grads)
        train_state = train_state.apply_gradients(grads=grads)

        metrics = {
            "losses": losses,
            "grad_norm": grad_norm,
        }

        return (rng, train_state, update_count+1), metrics

    #@jax.jit
    def train_scan(runner_state, _):
        (rng, train_state, update_count), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        metrics["update_count"] = update_count

        return (rng, train_state, update_count), metrics

    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state, 0)

    logger.info('Training')
    start_time = time.time()
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        runner_state, metrics = train_scan(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics)
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(int(metrics['update_count']), args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()
    return runner_state[1]

@hydra.main(config_path="config", config_name="world_model_key", version_base=None)
def config_main(config: DictConfig):
    logger.info('Initialising')

    wandb.login()
    main(config, project=config["project"])
    wandb.finish()


if __name__=="__main__":
    config_main()
