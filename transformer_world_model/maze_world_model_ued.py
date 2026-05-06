import json
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jaxued.environments.underspecified_env import EnvParams, UnderspecifiedEnv
from jaxued.utils import compute_max_mean_returns_epcount
import optax
from flax import struct
from flax.training.train_state import TrainState as BaseTrainState
import distrax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_w_key_generator
from jaxued.wrappers import AutoReplayWrapper

import logging
import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from training import compute_gae, compute_clipped_gae, sample_trajectories_world_model, evaluate_rnn, update_actor_critic_rnn, update_actor_critic
from networks import ActorCritic, AdversaryActorCritic
from networks.world_model import Encoder, Decoder, LatentDecoder, WorldModelDynamicsDecode, WorldModelLatentDecode
from env_utils import AdvObservation
from env_utils.sample_trajectories import sample_first_obs

logger = logging.getLogger(__name__)
 
@struct.dataclass
class TrainState:
    update_count: int
    pro_train_state: BaseTrainState
    adv_train_state: BaseTrainState

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
    overall_save_dir = os.path.join(os.getcwd(), f"checkpoints/UED/{config['group']}", f"{config['run_name']}")
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

def restore_checkpoint(config: dict, run_name: str) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.

    Args:
        config (dict): 
        train_state (TrainState): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 

    Returns:
        ocp.CheckpointManager: 
    """
    overall_save_dir = os.path.join(os.getcwd(), f"checkpoints/UED/{config['group']}", f"{run_name}")
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

def main(config: DictConfig = None, project="JAXUED_TEST"):
    wandb_config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=False
        )
    
    #run = wandb.init(config=config, project=project, group=config["group_name"], tags=["PAIRED",])
    if config["random_gen_actions"]:
        config["group"] = "WM - Random"

    if config["resume_run"]:
        api = wandb.Api()
        prev_run = api.run(f"{project}/{config['resume_run_id']}")
        checkpoint_manager = restore_checkpoint(config, prev_run.name)
        restore_step = checkpoint_manager.latest_step()
        run = wandb.init(config=wandb_config, project=project, group=config["group"], id=prev_run.id, resume="must")
        num_updates = config["num_updates"] - restore_step
    else:
        run = wandb.init(config=wandb_config, project=project, group=config["group"])
        num_updates = config["num_updates"]

    # Match Config Run name and WandB run name
    config['run_name'] = run.name
    
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("misc/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_length/*", step_metric="num_updates")

    def log_eval(stats):
        logger.info(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = 2 * stats["update_count"] * config["num_train_envs"] * config["student_num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
            "misc/prot_perf_mean": stats['pro_mean_returns'].mean(),
            "misc/prot_perf_max":  stats['pro_max_returns'].mean(),
            "misc/pro_num_episodes": stats['pro_eps'].mean(),
            "misc/adv_perf_mean": stats['adv_returns'].mean(),
            "misc/completed": stats['completed'].mean(),
        }
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config.eval.levels, solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config.eval.levels, returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})

        # log training losses
        def create_loss_dict(prefix, losses):
            return {
                f"{prefix}_losses/total_loss": losses[0].mean(),
                f"{prefix}_losses/value_loss": losses[1][0].mean(),
                f"{prefix}_losses/policy_loss": losses[1][1].mean(),
                f"{prefix}_losses/entropy": losses[1][2].mean(),
                f"{prefix}_losses/approx_kl": losses[1][3].mean(),
                f"{prefix}_losses/clipfrac": losses[1][4].mean(),
                f"{prefix}_losses/ratio": losses[1][5].mean(),
                f"{prefix}_losses/init_state_entropy": losses[1][6].mean(),
            }
        log_dict.update(create_loss_dict("pro", stats["pro_losses"]))
        if not config["random_gen_actions"]:
            log_dict.update(create_loss_dict("adv", stats["adv_losses"]))

        # animations
        if config["log_animations"]:
            for i, level_name in enumerate(config.eval.levels):
                frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
                frames = np.array(frames[:episode_length])
                log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)
    
    env = Maze(max_height=config['max_height'], max_width=config['max_width'], agent_view_size=5, normalize_obs=True)
    eval_env = env
    env_renderer = MazeRenderer(env, tile_size=8)
    level_generator = get_method(config.generator._target_)
    sample_random_level = level_generator(env.max_height, env.max_width, config["n_walls"])
    env = AutoReplayWrapper(env)
    env_params = env.default_params

    def init_actor_critic(rng, network_kws={}):
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], config["student_num_steps"], axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((config["student_num_steps"], config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n, **network_kws) # Could remove the use action with -1 here
        network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
        return network, network_params

    def init_adversary_actor_critic(rng, network_kws={}):
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], config["student_num_steps"], axis=0),
            obs,
        )
        adv_obs = AdvObservation(
            image=obs.image,
            agent_dir=obs.agent_dir,
            has_key=obs.has_key,
            agent_action=jnp.zeros((config["student_num_steps"], config["num_train_envs"]), dtype=jnp.int32),
            agent_value=jnp.zeros((config["student_num_steps"], config["num_train_envs"]), dtype=jnp.float32),
            step_num=jnp.zeros((config["student_num_steps"], config["num_train_envs"]), dtype=jnp.int32),
            hidden=jnp.zeros(((config["student_num_steps"], config["num_train_envs"], config["embed_dim"]))),
        )
        init_x = (adv_obs, jnp.zeros((config["student_num_steps"], config["num_train_envs"])))
        network = AdversaryActorCritic(config["num_gen_latents"], config["num_gen_cats"], config["student_num_steps"], **network_kws) # Could remove the use action with -1 here
        network_params = network.init(rng, init_x, AdversaryActorCritic.initialize_carry((config["num_train_envs"],)))
        return network, network_params

    @jax.jit
    def create_train_state(rng):
        def create_inner_train_state(rng, init_func, prefix, network_kws={}):
            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config[f"{prefix}num_minibatches"] * config[f"{prefix}epoch_ppo"]))
                    / config["num_updates"]
                )
                return config[f"{prefix}lr"] * frac
            network, network_params = init_func(rng, network_kws)
            learning_rate = linear_schedule if config[f"{prefix}anneal_lr"] else config[f"{prefix}lr"]
            tx = optax.chain(
                optax.clip_by_global_norm(config[f"{prefix}max_grad_norm"]),
                #optax.adam(learning_rate=linear_schedule, eps=1e-5),
                #optax.adam(learning_rate=config[f"{prefix}lr"], eps=1e-5),
                optax.adam(learning_rate=learning_rate, eps=1e-5),
            )
            return BaseTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        rng_pro, rng_adv = jax.random.split(rng)
        return TrainState(
            update_count = 0,
            pro_train_state = create_inner_train_state(rng_pro, init_actor_critic, "student_"),
            adv_train_state = create_inner_train_state(rng_adv, init_adversary_actor_critic, "adv_", network_kws={"use_pro_obs": config['use_pro_obs_for_adv']}),
        )
    
    def init_world_model(rng, checkpoint_dir, checkpoints_to_eval):
        # Load World Model Parameters
        checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_dir, 'models'), item_handlers=ocp.StandardCheckpointHandler())
        step = checkpoint_manager.latest_step() if checkpoints_to_eval == -1 else checkpoints_to_eval
        loaded_checkpoint = checkpoint_manager.restore(step)
        wm_params = loaded_checkpoint['params']

        # Get Dummy Obs and Actions
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], config["student_num_steps"], axis=0),
            obs,
        )
        actions = jnp.zeros((config["student_num_steps"], config["num_train_envs"]), dtype=jnp.int32)
        
        rng, rng_encoder, rng_decoder, rng_dynamics, rng_latent = jax.random.split(rng, 5)

        # Init Encoder
        z_dim = config['num_latents'] * config['num_cats']
        encoder = Encoder(
            hidden=config['encoder_hidden'],
            layers=config['encoder_layers'],
            activation=config['encoder_activation'],
            use_layer_norm=config["encoder_normalize"],
            conv_layers=config["encoder_conv_layers"],
            conv_channels=config["encoder_conv_channels"],
            z_dim=z_dim, 
            include_key=config['include_key']
        )
        _ = encoder.init(rng_encoder, obs)
        encoder_params = {'params': wm_params['params']['encoder']}

        # Get z Sample
        z_logits = encoder.apply(encoder_params, obs)
        z_logits = z_logits.reshape(*z_logits.shape[:-1], config['num_latents'], config['num_cats'])
        z_dist = distrax.Categorical(logits=z_logits)
        z_sample = jax.nn.one_hot(z_dist.mode(), config['num_cats'])
        z = z_sample.reshape(*z_sample.shape[:-2], -1)

        # Init Decoder
        decoder = Decoder(
            hidden=config['decoder_hidden'],
            layers=config['decoder_layers'],
            activation=config["decoder_activation"],
            use_layer_norm=config["decoder_normalize"],
            conv_layers=config["decoder_conv_layers"],
            conv_channels=config["decoder_conv_channels"],
            out_shape=(5,5,3), 
            include_key=config['include_key']
        )
        _ = decoder.init(rng_decoder, z)
        decoder_params = {'params': wm_params['params']['decoder']}

        # Init Dynamics Model
        context_size = config["context_size"]
        dynamics_model = WorldModelDynamicsDecode(
            (5,5,3),
            config, 
            decode=True,
            dtype=jnp.bfloat16,
        )
        dynamics_model_vars = dynamics_model.init(rng_dynamics, z[:context_size], actions[:context_size])
        dynamics_params = wm_params['params']
        init_cache = dynamics_model_vars["cache"]
        dynamics_outputs, _ = dynamics_model.apply(
            {"params": dynamics_params, "cache": init_cache}, 
            z[:context_size], 
            actions[:context_size],
            mutable=["cache"]
        )
        hiddens = dynamics_outputs['hiddens']

        # Init World Model Latent Decoder
        latent_gen_actions = jax.nn.one_hot(jnp.zeros((*hiddens.shape[:-1], config['num_gen_latents'])), config['num_gen_cats'])
        latent_gen_actions = latent_gen_actions.reshape(*latent_gen_actions.shape[:-2], -1)
        wm_latent_decoder = WorldModelLatentDecode((5,5,3), config)
        _ = wm_latent_decoder.init(rng_latent, dynamics_outputs, latent_gen_actions)

        # Init Latent Decoder for Generating First Obs
        latent_decoder = LatentDecoder(z_dim, config)
        _ = latent_decoder.init(rng_latent, latent_gen_actions, hiddens)
        latent_params = {'params': wm_params['params']['latent_decoder']}

        world_model = {
            "encoder": (encoder, encoder_params),
            "decoder": (decoder, decoder_params),
            "dynamics_model": (dynamics_model, dynamics_params),
            "latent_decoder": (wm_latent_decoder, wm_params),
            "first_obs_decoder": (latent_decoder, latent_params),
        }

        return world_model, init_cache
    
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_wm, rng_train = jax.random.split(rng, 3)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    world_model, init_cache = init_world_model(rng_wm, config.world_model.checkpoint_dir, config.world_model.checkpoint_to_eval)

    @jax.jit
    def train_step(runner_state, _):  
        def update(rng, train_state, init_hstate, rollout, prefix, multi_dim_action=False, action_mask=None, init_state_entropy_coeff=0.0):
            # Returns: (rng, train_state), losses
            return update_actor_critic_rnn(
                rng,
                train_state,
                init_hstate,
                rollout,
                config["num_train_envs"],
                config[f"{prefix}num_minibatches"],
                config[f"{prefix}epoch_ppo"],
                config[f"{prefix}clip_eps"],
                config[f"{prefix}entropy_coeff"],
                config[f"{prefix}critic_coeff"],
                init_state_entropy_coeff=init_state_entropy_coeff,
                update_grad=True,
                multi_dim_actions=multi_dim_action,
                action_mask=action_mask,
            )
        
        rng, train_state = runner_state
        pro_train_state = train_state.pro_train_state

        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
        new_levels = new_levels.replace(goal_placed=jnp.zeros_like(new_levels.goal_placed))
        init_obs, _ = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)

        if config["adv_gen_first_state"]:
            (adv_hstate, init_obs), adv_first_step = sample_first_obs(
                rng,
                config,
                train_state,
                world_model,
                init_obs,
                AdversaryActorCritic.initialize_carry((config["num_train_envs"],)),
                config["num_train_envs"],
                random_gen_actions=False,
                use_pro_obs_for_adv=config['use_pro_obs_for_adv'],
            )
        else:
            adv_hstate = AdversaryActorCritic.initialize_carry((config["num_train_envs"],))

        (
            (rng, last_value, adv_last_value), 
            (
                (obs, actions, rewards, dones, log_probs, values, info), 
                adv_traj,
                use_alt, # Boolean for whether the alt world model was used (and therefore whether the adv action had an effect)
            )
        ) = sample_trajectories_world_model(
            rng,
            config,
            train_state,
            world_model,
            ActorCritic.initialize_carry((config["num_train_envs"],)),
            adv_hstate,
            init_obs,
            init_cache,
            config["num_train_envs"],
            config["student_num_steps"],
            random_gen_actions=False,
        )

        # Add first state if it was generated
        if config["adv_gen_first_state"]:
            adv_obs, adv_actions, adv_log_probs, adv_values, adv_info = jax.tree_map(lambda x, y: jnp.concatenate((x, y), axis=0), adv_first_step, adv_traj)
            use_alt = jnp.pad(use_alt, ((1, 0), (0, 0)))
        else:
            adv_obs, adv_actions, adv_log_probs, adv_values, adv_info = adv_traj

        # Get Protagonist Rollout
        advantages, targets = compute_gae(config[f"student_gamma"], config[f"student_gae_lambda"], last_value, values, rewards, dones)
        pro_rollout = (obs, actions, dones, log_probs, values, targets, advantages)
        pro_mean_returns, pro_max_returns, pro_eps = compute_max_mean_returns_epcount(dones, rewards)

        # Get Adversary rollout
        completed_level = (rewards >= 0.1).any(axis=0)
        clipped_advantages, _ = compute_clipped_gae(config[f"student_gamma"], config[f"student_gae_lambda"], last_value, values, rewards, dones, use_max_value=True)
        maximised_negative_advantage = -clipped_advantages * completed_level

        # If the first state wasn't generated by the adversary, roll back rewards
        if config["adv_gen_first_state"]:
            adv_rewards = jnp.pad(maximised_negative_advantage, ((0, 1), (0, 0)))
        else:
            adv_rewards = jnp.roll(maximised_negative_advantage, -1, axis=0).at[-1].set(0)

        adv_dones = jnp.zeros(adv_actions.shape[:-1], dtype=bool).at[-1].set(True)
        adv_advantages, adv_targets = compute_gae(config[f"adv_gamma"], config[f"adv_gae_lambda"], adv_last_value, adv_values, adv_rewards, adv_dones)
        adv_rollout = (adv_obs, adv_actions, adv_dones, adv_log_probs, adv_values, adv_targets, adv_advantages)

        (rng, pro_train_state), pro_losses = update(rng, train_state.pro_train_state, ActorCritic.initialize_carry((config["num_train_envs"],)), pro_rollout, "student_")
        if config['random_gen_actions']:
            adv_train_state = train_state.adv_train_state
        else:
            (rng, adv_train_state), adv_losses = update(
                rng, train_state.adv_train_state, AdversaryActorCritic.initialize_carry((config["num_train_envs"],)), adv_rollout, "adv_", 
                multi_dim_action=True, action_mask=~use_alt, init_state_entropy_coeff=config["adv_init_state_entropy_coeff"]
            )

        metrics = {
            "pro_losses": jax.tree_map(lambda x: x.mean(), pro_losses),
            "pro_mean_returns": pro_mean_returns,
            "pro_max_returns": pro_max_returns,
            "pro_eps": pro_eps,
            "adv_returns": maximised_negative_advantage.sum(axis=0),
            "completed": completed_level,
        }

        if not config['random_gen_actions']:
            metrics["adv_losses"] = jax.tree_map(lambda x: x.mean(), adv_losses)

        train_state = train_state.replace(
            update_count=train_state.update_count + 1,
            pro_train_state=pro_train_state,
            adv_train_state=adv_train_state,
        )

        return (rng, train_state), metrics
    
    def eval(rng, train_state):
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config.eval.levels)
        num_levels = len(config.eval.levels)
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])
        
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state.pro_train_state)
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params) # (num_steps, num_eval_levels, ...)
        frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        metrics["update_count"] = train_state.update_count
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        
        return (rng, train_state), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['pro_train_state']['params']
            train_state = train_state_og.replace(pro_train_state=train_state_og.pro_train_state.replace(params=params))
            return train_state.pro_train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config.eval.levels)
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval': return eval_checkpoint(config) # evaluate and exit early

    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)

    if config["resume_run"]:
        train_state = checkpoint_manager.restore(
            restore_step, 
            args=ocp.args.StandardRestore(train_state),
        )
    elif config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config)
    runner_state = (rng_train, train_state)

    logger.info('Training')
    start_time = time.time()
    for eval_step in range(num_updates // config["eval_freq"]):
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics)
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(int(metrics['update_count']), args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()
    return runner_state[1]

@hydra.main(config_path="config", config_name="maze_degen", version_base=None)
def config_main(config: DictConfig):
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])

    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    logger.info('Initialising')

    wandb.login()
    main(config, project=config["project"])
    wandb.finish()


if __name__=="__main__":
    config_main()
