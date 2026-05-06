from typing import Sequence, Tuple, Dict, Optional
import jax
import jax.numpy as jnp
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
import chex
import distrax

from flax.training.train_state import TrainState
from flax import struct

from env_utils import AdvObservation

def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
    init_done: Optional[chex.Array] = None,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): Singleton
        init_hstate (chex.ArrayTree): This is the init RNN hidden state, has to have shape (NUM_ENVS, ...)
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)
    
    if init_done is None:
        init_done = jnp.zeros(num_envs, dtype=bool)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            init_done,
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): Shape (num_levels, )
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int): 

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]
    
    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        
        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)
    
    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths


def sample_trajectories_world_model(
    rng: chex.PRNGKey,
    config: Dict,
    train_state: TrainState,
    world_model: Dict,
    init_pro_hstate: chex.ArrayTree,
    init_adv_hstate: chex.ArrayTree,
    init_obs: chex.Array,
    init_cache: Dict,
    num_envs: int,
    max_episode_length: int,
    random_gen_actions: bool = False,
):
    pro_train_state = train_state.pro_train_state
    adv_train_state = train_state.adv_train_state

    decoder, decoder_params = world_model['decoder']
    encoder, encoder_params = world_model['encoder']
    dynamics_model, dynamics_params = world_model['dynamics_model']
    latent_decoder, latent_params = world_model['latent_decoder']

    # Encode initial observation to get initial latent state
    init_obs = jax.tree_map(lambda x: x[None, ...], init_obs)
    init_z_logits = encoder.apply(encoder_params, init_obs)
    init_z_logits = init_z_logits.reshape(*init_z_logits.shape[:-1], config["num_latents"], config["num_cats"])
    init_z_dist = distrax.Categorical(logits=init_z_logits)
    init_latent_z = jax.nn.one_hot(init_z_dist.mode(), config["num_cats"])
    init_latent_z = init_latent_z.reshape(*init_latent_z.shape[:-2], -1)

    def sample_step(carry, _):
        rng, pro_hstate, adv_hstate, latent_z, cache, last_done, step_num = carry

        obs = decoder.apply(decoder_params, latent_z)
        if not config["include_key"]:
            obs = obs.replace(has_key=jnp.zeros_like(obs.agent_dir))

        rng, rng_action, rng_latents = jax.random.split(rng, 3)

        x = (obs, last_done[None, ...])
        pro_hstate, pi, value = pro_train_state.apply_fn(pro_train_state.params, x, pro_hstate)

        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)

        dynamics_outputs, updated_vars = dynamics_model.apply(
            {"params": dynamics_params, "cache": cache}, 
            latent_z, 
            action,
            mutable=["cache"]
        )
        hiddens = dynamics_outputs['hiddens']

        adv_obs = AdvObservation(
            image=obs.image,
            agent_dir=obs.agent_dir,
            has_key=obs.has_key,
            agent_action=action,
            agent_value=value,
            step_num=step_num[None, ...],
            hidden=hiddens,
        )

        x = (adv_obs, jnp.zeros_like(last_done)[None, ...])
        adv_hstate, adv_pi, adv_value = adv_train_state.apply_fn(adv_train_state.params, x, adv_hstate)
        adv_action = adv_pi.sample(seed=rng_latents)
        adv_log_prob = adv_pi.log_prob(adv_action)

        if random_gen_actions:
            latent_gen_actions = jax.random.randint(rng_latents, (*hiddens.shape[:-1], config['num_gen_latents']), 0, config['num_gen_cats'])
            latent_gen_actions = jax.nn.one_hot(latent_gen_actions, config['num_gen_cats'])
            latent_gen_actions = latent_gen_actions.reshape(*latent_gen_actions.shape[:-2], -1)
        else:
            latent_gen_actions = jax.nn.one_hot(adv_action, config['num_gen_cats'])
            latent_gen_actions = latent_gen_actions.reshape(*latent_gen_actions.shape[:-2], -1)

        pred_z, pred_reward, pred_done, use_alt = latent_decoder.apply(latent_params, dynamics_outputs, latent_gen_actions, return_use_alt=True)

        obs = jax.tree_map(lambda x: x.squeeze(0), obs)

        value, action, log_prob, pred_reward, pred_done = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
            pred_reward.squeeze(0),
            pred_done.squeeze(0),
        )

        adv_value, adv_action, adv_log_prob = (
            adv_value.squeeze(0),
            adv_action.squeeze(0),
            adv_log_prob.squeeze(0),
        )
        adv_obs = jax.tree_map(lambda x: x.squeeze(0), adv_obs)

        pred_done = jnp.round(pred_done).astype(bool)

        carry = (rng, pro_hstate, adv_hstate, pred_z, updated_vars['cache'], pred_done, step_num + 1)
        return carry, ((obs, action, pred_reward, pred_done, log_prob, value, {}), (adv_obs, adv_action, adv_log_prob, adv_value, {}), use_alt.squeeze(0))
    
    (rng, pro_hstate, adv_hstate, last_latent_z, last_cache, last_done, step_num), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            init_pro_hstate,
            init_adv_hstate,
            init_latent_z,
            init_cache,
            jnp.zeros(num_envs, dtype=bool),
            jnp.zeros(num_envs, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    last_obs = decoder.apply(decoder_params, last_latent_z)
    last_obs = last_obs.replace(has_key=jnp.zeros_like(last_obs.agent_dir))

    x = (last_obs, last_done[None, ...])
    _, _, last_value = pro_train_state.apply_fn(pro_train_state.params, x, pro_hstate)
    adv_last_value = jnp.zeros_like(last_value)

    return (rng, last_value.squeeze(0), adv_last_value.squeeze(0)), traj
