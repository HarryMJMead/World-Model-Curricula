from typing import Sequence, Tuple, Dict
import jax
import jax.numpy as jnp
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv, Level
import chex
import distrax

from flax.training.train_state import TrainState
from .utils import update_env_state_from_obs, update_level_from_state, AdvObservation

def get_mode_logits(logits, num_latents, num_cats):
    logits = logits.reshape(*logits.shape[:-1], num_latents, num_cats)
    dist = distrax.Categorical(logits=logits)
    sample = jax.nn.one_hot(dist.mode(), num_cats)
    return sample.reshape(*sample.shape[:-2], -1)

def sample_first_obs(
    rng: chex.PRNGKey,
    config: Dict,
    train_state: TrainState,
    world_model: Dict,
    init_obs: Observation,
    init_adv_hstate: chex.ArrayTree,
    num_envs: int,
    random_gen_actions: bool = False,
    use_pro_obs_for_adv: bool = False,
):
    adv_train_state = train_state.adv_train_state

    decoder, decoder_params = world_model['decoder']
    latent_decoder, latent_params = world_model['first_obs_decoder']

    obs_unsqueezed = jax.tree_map(lambda x: x[None, ...], init_obs)

    adv_obs = AdvObservation(
        image=obs_unsqueezed.image,
        agent_dir=obs_unsqueezed.agent_dir,
        has_key=obs_unsqueezed.has_key,
        agent_action=jnp.zeros((1, num_envs), dtype=jnp.int32),
        agent_value=jnp.zeros((1, num_envs)),
        step_num=jnp.zeros((1, num_envs), dtype=jnp.int32),
        hidden=jnp.zeros((1, num_envs, config["embed_dim"])),
    )
    
    rng, rng_latents = jax.random.split(rng)
    x = (adv_obs, jnp.zeros((1, num_envs), dtype=bool))
    adv_hstate, adv_pi, adv_value = adv_train_state.apply_fn(adv_train_state.params, x, init_adv_hstate)
    adv_action = adv_pi.sample(seed=rng_latents)
    adv_log_prob = adv_pi.log_prob(adv_action)

    # Get Latent Gen Action
    if random_gen_actions:
        latent_gen_actions = jax.random.randint(rng_latents, (1, num_envs, config['num_gen_latents']), 0, config['num_gen_cats'])
        latent_gen_actions = jax.nn.one_hot(latent_gen_actions, config['num_gen_cats'])
        latent_gen_actions = latent_gen_actions.reshape(*latent_gen_actions.shape[:-2], -1)
    else:
        latent_gen_actions = jax.nn.one_hot(adv_action, config['num_gen_cats'])
        latent_gen_actions = latent_gen_actions.reshape(*latent_gen_actions.shape[:-2], -1)

    # Get Predicted Latent, Reward, Done
    latent_decoder_outputs = latent_decoder.apply(latent_params, latent_gen_actions, jnp.zeros((1, num_envs, config["embed_dim"])))
    pred_z_logits, _, _ = latent_decoder_outputs['outputs']
    pred_z = get_mode_logits(pred_z_logits, config['num_latents'], config['num_cats'])

    # Get World Model Predicted Obs
    pred_next_obs = decoder.apply(decoder_params, pred_z)
    pred_next_obs = jax.tree_map(lambda x: x.squeeze(0), pred_next_obs)

    # We only want to actually modify the init_obs image
    pred_next_obs = init_obs.replace(image=pred_next_obs.image)

    return (adv_hstate, pred_next_obs), (adv_obs, adv_action, adv_log_prob, adv_value, {})
    

def sample_trajectories_world_model_forced(
    rng: chex.PRNGKey,
    config: Dict,
    train_state: TrainState,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    world_model: Dict,
    init_obs: Observation,
    init_env_state: EnvState,
    init_pro_hstate: chex.ArrayTree,
    init_adv_hstate: chex.ArrayTree,
    init_cache: Dict,
    num_envs: int,
    max_episode_length: int,
    return_use_alt: bool = False,
    random_gen_actions: bool = False,
):
    pro_train_state = train_state.pro_train_state
    adv_train_state = train_state.adv_train_state

    decoder, decoder_params = world_model['decoder']
    encoder, encoder_params = world_model['encoder']
    dynamics_model, dynamics_params = world_model['dynamics_model']
    latent_decoder, latent_params = world_model['latent_decoder']

    init_env_state = init_env_state.replace(
        env_state=init_env_state.env_state.replace(
            observation_map=jnp.zeros_like(init_env_state.env_state.wall_map),
            goal_placed=jnp.zeros_like(init_env_state.env_state.goal_placed),
            door_placed=jnp.zeros_like(init_env_state.env_state.door_placed),
            key_placed=jnp.zeros_like(init_env_state.env_state.key_placed),
        )
    )

    new_env_state, _ = update_env_state_from_obs(init_env_state.env_state, init_obs)
    init_env_state = init_env_state.replace(env_state=new_env_state)

    init_obs = jax.vmap(env._env.get_obs)(init_env_state.env_state)

    def sample_step(carry, _):
        rng, obs, prev_env_state, pro_hstate, adv_hstate, cache, last_done, step_num = carry

        # Split RNG for different uses
        rng, rng_action, rng_latents, rng_step = jax.random.split(rng, 4)

        obs_unsqueezed = jax.tree_map(lambda x: x[None, ...], obs)

        # Get Protagonist Action
        x = (obs_unsqueezed, last_done[None, ...])
        pro_hstate, pi, value = pro_train_state.apply_fn(pro_train_state.params, x, pro_hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)

        # Get Latent Representation
        z_logits = encoder.apply(encoder_params, obs_unsqueezed)
        latent_z = get_mode_logits(z_logits, config['num_latents'], config['num_cats'])
        dynamics_outputs, updated_vars = dynamics_model.apply(
            {"params": dynamics_params, "cache": cache}, 
            latent_z, 
            action,
            mutable=["cache"]
        )
        hiddens = dynamics_outputs['hiddens']

        adv_obs = AdvObservation(
            image=obs_unsqueezed.image,
            agent_dir=obs_unsqueezed.agent_dir,
            has_key=obs_unsqueezed.has_key,
            agent_action=action,
            agent_value=value,
            step_num=step_num[None, ...],
            hidden=hiddens,
        )
        
        # Get Adversary Action
        x = (adv_obs, jnp.zeros_like(last_done)[None, ...])
        adv_hstate, adv_pi, adv_value = adv_train_state.apply_fn(adv_train_state.params, x, adv_hstate)
        adv_action = adv_pi.sample(seed=rng_latents)
        adv_log_prob = adv_pi.log_prob(adv_action)

        # Get Latent Gen Action
        if random_gen_actions:
            latent_gen_actions = jax.random.randint(rng_latents, (*hiddens.shape[:-1], config['num_gen_latents']), 0, config['num_gen_cats'])
            latent_gen_actions = jax.nn.one_hot(latent_gen_actions, config['num_gen_cats'])
            latent_gen_actions = latent_gen_actions.reshape(*latent_gen_actions.shape[:-2], -1)
        else:
            latent_gen_actions = jax.nn.one_hot(adv_action, config['num_gen_cats'])
            latent_gen_actions = latent_gen_actions.reshape(*latent_gen_actions.shape[:-2], -1)

        # Get Predicted Latent, Reward, Done
        latent_decoder_outputs = latent_decoder.apply(latent_params, dynamics_outputs, latent_gen_actions, return_use_alt=return_use_alt)
        pred_z = latent_decoder_outputs[0]

        # Get World Model Predicted Obs
        pred_next_obs = decoder.apply(decoder_params, pred_z)
        pred_next_obs = jax.tree_map(lambda x: x.squeeze(0), pred_next_obs)

        # Squeeze Protagonist and Antagonist Outputs
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )
        adv_value, adv_action, adv_log_prob = (
            adv_value.squeeze(0),
            adv_action.squeeze(0),
            adv_log_prob.squeeze(0),
        )
        adv_obs = jax.tree_map(lambda x: x.squeeze(0), adv_obs)

        # Step Env State
        _, new_env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), prev_env_state, action, env_params)
        env_state = new_env_state.replace(
            env_state=new_env_state.env_state.replace(
                observation_map=prev_env_state.env_state.observation_map,
            )
        )

        # Update Env State with World Model Observation
        new_env_state, no_updates = update_env_state_from_obs(env_state.env_state, pred_next_obs)
        new_level = update_level_from_state(new_env_state, env_state.level)
        env_state = env_state.replace(env_state=new_env_state, level=new_level)
        
        next_obs = jax.vmap(env._env.get_obs)(env_state.env_state)

        if return_use_alt:
            # If return_use_alt is True, use the world model prediction to determine
            # whether the adversary action had any effect
            no_updates = latent_decoder_outputs[3].squeeze(0)

        carry = (rng, next_obs, env_state, pro_hstate, adv_hstate, updated_vars['cache'], done, step_num + 1)
        return carry, ((obs, action, reward, done, log_prob, value, {}), (adv_obs, adv_action, adv_log_prob, adv_value, {}), no_updates, prev_env_state)
    
    (rng, last_obs, last_env_state, pro_hstate, adv_hstate, last_cache, last_done, step_num), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            init_obs,
            init_env_state,
            init_pro_hstate,
            init_adv_hstate,
            init_cache,
            jnp.zeros(num_envs, dtype=bool),
            jnp.zeros(num_envs, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = pro_train_state.apply_fn(pro_train_state.params, x, pro_hstate)
    adv_last_value = jnp.zeros_like(last_value)

    return (rng, last_value.squeeze(0), adv_last_value.squeeze(0), last_env_state, (last_obs, pro_hstate, last_done)), traj