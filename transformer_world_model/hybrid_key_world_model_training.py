import json
import os
import time
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jaxued.environments.underspecified_env import EnvParams, EnvState, UnderspecifiedEnv
from jax.tree_util import Partial
import optax
from flax import struct, traverse_util
from flax.training.train_state import TrainState
import flax.linen as nn
import distrax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.maze.env import Observation
from jaxued.environments import Maze
from jaxued.wrappers import AutoReplayWrapper
import chex

import logging
import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from networks import WorldModelwLatent, ActorCritic
from training import sample_trajectories_rnn

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
    overall_save_dir = os.path.join(os.getcwd(), f"checkpoints/{config['group']}", f"{config['run_name']}", str(config['seed']))
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
    tags = ['key_minigrid']
    run = wandb.init(config=wandb_config, project=project, tags=tags, group=config["group"])

    # Match Config Run name and WandB run name
    config['run_name'] = run.name

    wandb.define_metric("num_updates")
    wandb.define_metric("losses/*", step_metric="num_updates")
    wandb.define_metric("training/*", step_metric="num_updates")
    wandb.define_metric("eval/*", step_metric="num_updates")
    wandb.define_metric("latent_eval/*", step_metric="num_updates")

    data_recorder = ocp.PyTreeCheckpointer()
    dataset_path = os.path.join(os.getcwd(), config['dataset_path'], config['dataset_name'])
    dataset = data_recorder.restore(
        dataset_path,
        item=example_dataset
    )
    dataset_steps_per_traj, dataset_size = dataset.actions.shape[0:2]

    # Setup Environment for Eval
    env = Maze(max_height=config['max_height'], max_width=config['max_width'], agent_view_size=5, normalize_obs=True)
    level_generator = get_method(config.generator._target_)
    sample_random_level = level_generator(env.max_height, env.max_width, config["n_walls"])
    env = AutoReplayWrapper(env)
    env_params = env.default_params.replace(max_steps_in_episode=config['max_steps_in_episode'])

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
            "losses/world_model_loss": stats["loss"].mean(),
            "losses/reconstruction_loss": stats["losses"][0].mean(),
            "losses/prediction_loss": stats["losses"][1].mean(),
            "losses/dynamic_loss": stats["losses"][2].mean(),
            "losses/latent_entropy": stats["losses"][3].mean(),
            "losses/accuracy": stats["accuracies"][0].mean(),
            "losses/conditioned_accuracy": stats["accuracies"][1].mean(),

            "losses/prediction_loss_alt": stats["losses"][4].mean(),
            "losses/dynamic_loss_alt": stats["losses"][5].mean(),
            "losses/selection_loss": stats["losses"][6].mean(),
            "losses/selection_override_loss": stats["losses"][7].mean(),
            "losses/use_alt": stats["use_alts"][0].mean(),
            "losses/use_alt_true": stats["use_alts"][1].mean(),
            "losses/use_alt_override": stats["use_alts"][2].mean(),
            "losses/use_alt_accuracy": stats["accuracies"][2].mean(),
            "losses/use_alt_override_accuracy": stats["accuracies"][3].mean(),

            "num_updates": stats["update_count"],
            "sps": num_steps / stats['time_delta'],

            "training/grad_norm": stats['grad_norm'].mean(),
            "training/lr": linear_schedule(stats['update_count']),

            "eval/world_model_loss": stats["eval_loss"].mean(),
            "eval/reconstruction_loss": stats["eval_losses"][0].mean(),
            "eval/prediction_loss": stats["eval_losses"][1].mean(),
            "eval/dynamic_loss": stats["eval_losses"][2].mean(),
            "eval/latent_entropy": stats["eval_losses"][3].mean(),
            "eval/accuracy": stats["eval_accuracies"][0].mean(),
            "eval/conditioned_accuracy": stats["eval_accuracies"][1].mean(),

            "eval/prediction_loss_alt": stats["eval_losses"][4].mean(),
            "eval/dynamic_loss_alt": stats["eval_losses"][5].mean(),
            "eval/selection_loss": stats["eval_losses"][6].mean(),
            "eval/selection_override_loss": stats["eval_losses"][7].mean(),
            "eval/use_alt": stats["eval_use_alts"][0].mean(),
            "eval/use_alt_true": stats["eval_use_alts"][1].mean(),
            "eval/use_alt_override": stats["eval_use_alts"][2].mean(),
            "eval/use_alt_accuracy": stats["eval_accuracies"][2].mean(),
            "eval/use_alt_override_accuracy": stats["eval_accuracies"][3].mean(),

            "latent_eval/world_model_loss": stats["latent_eval_loss"].mean(),
            "latent_eval/reconstruction_loss": stats["latent_eval_losses"][0].mean(),
            "latent_eval/prediction_loss": stats["latent_eval_losses"][1].mean(),
            "latent_eval/dynamic_loss": stats["latent_eval_losses"][2].mean(),
            "latent_eval/latent_entropy": stats["latent_eval_losses"][3].mean(),
            "latent_eval/accuracy": stats["latent_eval_accuracies"][0].mean(),
            "latent_eval/conditioned_accuracy": stats["latent_eval_accuracies"][1].mean(),

            "latent_eval/prediction_loss_alt": stats["latent_eval_losses"][4].mean(),
            "latent_eval/dynamic_loss_alt": stats["latent_eval_losses"][5].mean(),
            "latent_eval/selection_loss": stats["latent_eval_losses"][6].mean(),
            "latent_eval/selection_override_loss": stats["latent_eval_losses"][7].mean(),
            "latent_eval/use_alt": stats["latent_eval_use_alts"][0].mean(),
            "latent_eval/use_alt_true": stats["latent_eval_use_alts"][1].mean(),
            "latent_eval/use_alt_override": stats["latent_eval_use_alts"][2].mean(),
            "latent_eval/use_alt_accuracy": stats["latent_eval_accuracies"][2].mean(),
            "latent_eval/use_alt_override_accuracy": stats["latent_eval_accuracies"][3].mean(),
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
        
        world_model = WorldModelwLatent((5, 5, 3), config, dtype=jnp.bfloat16)
        wm_params = world_model.init(rng_init, rng_app, obs, actions, batch.rewards, batch.dones, deterministic=True)

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

    def get_frozen_train_state(train_state, checkpoint_dir, checkpoint_to_eval):
        checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_dir, 'models'), item_handlers=ocp.StandardCheckpointHandler())
        step = checkpoint_manager.latest_step() if checkpoint_to_eval == -1 else checkpoint_to_eval
        loaded_checkpoint = checkpoint_manager.restore(step)

        if 'wm_train_state' in loaded_checkpoint:
            frozen_params = traverse_util.flatten_dict(loaded_checkpoint['wm_train_state']['params'])
        else:
            frozen_params = traverse_util.flatten_dict(loaded_checkpoint['params'])
        frozen_param_paths = set(['encoder', 'decoder'])

        params = train_state.params
        apply_fn = train_state.apply_fn
        tx = train_state.tx

        params = traverse_util.path_aware_map(
            lambda path, v: frozen_params[path] if frozen_param_paths & set(path) else v, params
        )

        partition_optimizers = {'trainable': tx, 'frozen': optax.set_to_zero()}
        param_partitions = traverse_util.path_aware_map(
            lambda path, _: 'frozen' if frozen_param_paths & set(path) else 'trainable', params
        )
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        return TrainState.create(
            apply_fn=apply_fn,
            params=params,
            tx=tx)
    
    def load_full_train_state(train_state, checkpoint_dir, checkpoint_to_eval):
        checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_dir, 'models'), item_handlers=ocp.StandardCheckpointHandler())
        step = checkpoint_manager.latest_step() if checkpoint_to_eval == -1 else checkpoint_to_eval
        restored_state = checkpoint_manager.restore(
                step, 
                args=ocp.args.StandardRestore(train_state),
            )
        
        # Reset global step and optimizer state.
        new_opt_state = train_state.tx.init(restored_state.params)

        return restored_state.replace(
            step=0,
            opt_state=new_opt_state,
        )

    def get_agent_train_state(rng, checkpoint_dir, checkpoint_to_eval):
        checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_dir, 'models'), item_handlers=ocp.StandardCheckpointHandler())
        step = checkpoint_manager.latest_step() if checkpoint_to_eval == -1 else checkpoint_to_eval
        loaded_checkpoint = checkpoint_manager.restore(step)

        agent_params = loaded_checkpoint['pro_train_state']['params']

        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_eval_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_eval_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        _ = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_eval_envs"],)))

        return TrainState.create(
            apply_fn=network.apply,
            params=agent_params,
            tx=optax.set_to_zero())


    def train_step(runner_state, _):
        rng, train_state, update_count = runner_state

        rng, rng_batch, rng_wm, rng_dropout = jax.random.split(rng, 4)
        batch_indices = jax.random.randint(
            rng_batch, (config["batch_size"],), 0, dataset_size
        )
        batch = jax.tree_util.tree_map(lambda x: x[:, batch_indices], dataset)
        
        # Pad the actions, as the obs array contains an extra timestep
        actions = jnp.pad(batch.actions, ((0, 1), (0, 0)))

        # obs are stored as uint8 for space, so need to be normalised
        obs = batch.obs.replace(image=batch.obs.image/10)

        def loss_fn(params):
            (z_dist, pred_next_z_dist), (recon_obs, pred_reward, pred_done), (use_alt, use_alt_true, use_alt_override), (l_recon, l_pred, l_dyn, l_pred_alt, l_dyn_alt, l_select, l_select_override) = train_state.apply_fn(
                params, 
                rng_wm, 
                obs, 
                actions,
                batch.rewards,
                batch.dones,
                deterministic=False,
                gen_first_obs=config['gen_first_obs'],
                rngs={'dropout': rng_dropout}
            )

            mode_accuracy = (pred_next_z_dist.mode() == z_dist.mode()[1:]).mean()
            conditioned_mode_accuracy = (pred_next_z_dist.mode() == z_dist.mode()[1:])[100:].mean()

            use_alt_accuracy = (use_alt[:-1] == use_alt_true).mean()

            if config['use_hybrid_override']:
                use_alt_override_accuracy = (use_alt_override[:use_alt_true.shape[0]] == use_alt_true).mean()
                use_alt_override = use_alt_override.mean()
            else:
                use_alt_override_accuracy = 0

            entropy = z_dist.entropy().mean()
            loss = l_recon + l_pred + l_dyn - config['entropy_coeff'] * entropy + l_pred_alt + l_dyn_alt + l_select + l_select_override
            return loss, ((l_recon, l_pred, l_dyn, entropy, l_pred_alt, l_dyn_alt, l_select, l_select_override), (mode_accuracy, conditioned_mode_accuracy, use_alt_accuracy, use_alt_override_accuracy), (use_alt.mean(), use_alt_true.mean(), use_alt_override))
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (losses, accuracies, use_alts)), grads = grad_fn(train_state.params)
        grad_norm = optax.global_norm(grads)
        train_state = train_state.apply_gradients(grads=grads)

        metrics = {
            "loss": loss,
            "losses": losses,
            "accuracies": accuracies,
            "use_alts": use_alts,
            "grad_norm": grad_norm,
        }

        return (rng, train_state, update_count+1), metrics


    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train, rng_agent = jax.random.split(rng, 3)
    agent_train_state = get_agent_train_state(rng_agent, config['agent_path'], config['agent_checkpoint_num'])


    @jax.jit
    def eval(rng, train_state):
        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_eval_envs"]))
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_eval_envs"]), new_levels, env_params)
        # Rollout
        (
            (rng, _, _, last_obs, _, _),
            (obs, actions, rewards, dones, _, _, _),
        ) = sample_trajectories_rnn(
            rng,
            env,
            env_params,
            agent_train_state,
            ActorCritic.initialize_carry((config["num_eval_envs"],)),
            init_obs,
            init_env_state,
            config["num_eval_envs"],
            config["num_eval_steps"],
        )

        obs = jax.tree_map(lambda x, y: jnp.append(x, y[None, ...], axis=0), obs, last_obs)
        actions = jnp.pad(actions, ((0, 1), (0, 0)))

        rng, rng_gen, rng_wm = jax.random.split(rng, 3)
        random_latent_gen_actions = jax.random.randint(rng_gen, (*actions.shape, config['num_gen_latents']), minval=0, maxval=config['num_gen_cats'])
        random_latent_gen_actions = jax.nn.one_hot(random_latent_gen_actions, config['num_gen_cats']).reshape(*random_latent_gen_actions.shape[:2], -1)

        def loss_fn(params, latent_gen_actions=None):
            (z_dist, pred_next_z_dist), (recon_obs, pred_reward, pred_done), (use_alt, use_alt_true, use_alt_override), (l_recon, l_pred, l_dyn, l_pred_alt, l_dyn_alt, l_select, l_select_override) = train_state.apply_fn(
                params, 
                rng_wm, # rng not used for deterministic world model, so does not need to change
                obs, 
                actions,
                rewards,
                dones,
                latent_gen_actions=latent_gen_actions,
                deterministic=True
            )

            if latent_gen_actions != None:
                pred_next_z_dist = pred_next_z_dist[:-1]

            mode_accuracy = (pred_next_z_dist.mode() == z_dist.mode()[1:]).mean()
            conditioned_mode_accuracy = (pred_next_z_dist.mode() == z_dist.mode()[1:])[100:].mean()

            use_alt_accuracy = (use_alt[:-1] == use_alt_true).mean()

            if config['use_hybrid_override']:
                use_alt_override_accuracy = (use_alt_override[:use_alt_true.shape[0]] == use_alt_true).mean()
                use_alt_override = use_alt_override.mean()
            else:
                use_alt_override_accuracy = 0

            entropy = z_dist.entropy().mean()
            loss = l_recon + l_pred + l_dyn - config['entropy_coeff'] * entropy + l_pred_alt + l_dyn_alt + l_select + l_select_override
            return loss, ((l_recon, l_pred, l_dyn, entropy, l_pred_alt, l_dyn_alt, l_select, l_select_override), (mode_accuracy, conditioned_mode_accuracy, use_alt_accuracy, use_alt_override_accuracy), (use_alt.mean(), use_alt_true.mean(), use_alt_override))
        
        return loss_fn(train_state.params, random_latent_gen_actions), loss_fn(train_state.params)
    

    def train_and_eval(runner_state, _):
        (rng, train_state, update_count), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])
        
        rng, _rng = jax.random.split(rng)
        (eval_loss, (eval_losses, eval_accuracies, eval_use_alts)), (latent_eval_loss, (latent_eval_losses, latent_eval_accuracies, latent_eval_use_alts)) = eval(_rng, train_state)

        metrics['eval_loss'] = eval_loss
        metrics['eval_losses'] = eval_losses
        metrics['eval_accuracies'] = eval_accuracies
        metrics['eval_use_alts'] = eval_use_alts

        metrics['latent_eval_loss'] = latent_eval_loss
        metrics['latent_eval_losses'] = latent_eval_losses
        metrics['latent_eval_accuracies'] = latent_eval_accuracies
        metrics['latent_eval_use_alts'] = latent_eval_use_alts

        metrics["update_count"] = update_count
        logger.debug(f"Grad Norm: {metrics['grad_norm'].mean()}")

        return (rng, train_state, update_count), metrics

    
    train_state = create_train_state(rng_init)
    if config['use_frozen_encoder']:
        train_state = get_frozen_train_state(train_state, config['encoder_path'], config['encoder_checkpoint_num'])
    if config['use_fine_tuning']:
        train_state = load_full_train_state(train_state, config['fine_tune_path'], config['fine_tune_checkpoint_num'])
    runner_state = (rng_train, train_state, 0)

    logger.info('Training')
    start_time = time.time()
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        runner_state, metrics = train_and_eval(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics)
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(int(metrics['update_count']), args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()
    return runner_state[1]

@hydra.main(config_path="config", config_name="world_model", version_base=None)
def config_main(config: DictConfig):
    logger.info('Initialising')

    config['group'] = "RoPE Transformer Hybrid World Model"
    if config['use_fine_tuning']:
        config['group'] += " - Fine Tuning"
    config['use_hybrid_gen'] = True

    wandb.login()
    main(config, project=config["project"])
    wandb.finish()


if __name__=="__main__":
    config_main()
