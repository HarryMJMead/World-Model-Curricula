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
from flax import struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.maze.env_editor import MazeEditor, Observation, LocalKeyMazeEditorRotateSplitAct
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer, ObservedMazeRenderer, LocalObservedMazeRenderer
from jaxued.environments.maze import Level, ObservedLevel
from jaxued.wrappers import AutoReplayWrapper
import chex

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# region PPO helper functions    
@struct.dataclass
class TrainState:
    update_count: int
    pro_train_state: BaseTrainState
    adv_train_state: BaseTrainState

def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
    traj_idxs: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value, cur_idx = carry
        traj_started = cur_idx <= traj_idxs
    
        value, reward, done = x
        delta = (reward + gamma * next_value * (1 - done) - value) * traj_started
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, jnp.where(traj_started, value, next_value), cur_idx - 1), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value, values.shape[0]),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values

def compute_clipped_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
    traj_idxs: chex.Array, # Currently not used
    use_max_value: bool = False,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    extended_values = jnp.append(values, last_value[None, ...], axis=0)
    deltas = rewards + gamma * extended_values[1:] * (1 - dones) - extended_values[:-1]

    start_index = jnp.arange(values.shape[0])
    def compute_gae_at_timestep(carry, x):
        gae_array, td_total_array, is_done_array, cur_idx = carry
        
        delta, done = x
        delta = delta[None, ...] * (cur_idx >= start_index[..., None])
        done = done[None, ...] * (cur_idx >= start_index[..., None])

        i = (cur_idx - start_index)[..., None]

        td_total_array = td_total_array + gamma**i * delta
    
        clipped_td_total_array = jnp.minimum(td_total_array, 0)
        if use_max_value:
            td_total_array = clipped_td_total_array

        gae_array = gae_array + (lambd**(i)) * clipped_td_total_array * (1 - is_done_array) + ((lambd**(i+1))/(1 - lambd)) * clipped_td_total_array * done * (1 - is_done_array)
        is_done_array = jnp.logical_or(is_done_array, done)

        return (gae_array, td_total_array, is_done_array, cur_idx + 1), None

    carry, _ = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(values), jnp.zeros_like(values), jnp.zeros_like(dones), 0),
        (deltas, dones),
        unroll=16,
    )
    gae_array, td_total_array, is_done_array, cur_idx = carry
    i = (cur_idx - start_index)[..., None]
    clipped_td_total_array = jnp.minimum(td_total_array, 0)
    advantages = (1 - lambd)*(gae_array + ((lambd**(i))/(1 - lambd)) * clipped_td_total_array * (1 - is_done_array))

    return advantages, advantages + values

def sample_trajectories(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    adv_env: UnderspecifiedEnv,
    adv_env_params: EnvParams,
    train_state: TrainState,
    init_levels: ObservedLevel,
    num_envs: int,
    num_pro_traj: int,
    max_student_episode_length: int,
    max_adv_episode_length: int,
):
    pro_train_state = train_state.pro_train_state
    adv_train_state = train_state.adv_train_state

    num_pro_envs = num_envs * num_pro_traj

    rng, rng_adv, rng_pro = jax.random.split(rng, 3)
    adv_init_obs, adv_init_env_state = jax.vmap(adv_env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_adv, num_envs), init_levels, adv_env_params)
    pro_init_obs, pro_init_env_state =  jax.tree_map(lambda x: x.repeat(num_pro_traj, axis=0), jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_pro, num_envs), adv_init_env_state.level, env_params))

    # Replace Carry
    def replace_carry(new, old, new_carry):
        expanded_new_carry = jnp.reshape(new_carry, (new_carry.shape[0],) + (1,) * (new.ndim - 1))
        return jnp.where(expanded_new_carry, new, old)

    # Agent Step in Env
    def sample_agent_step(rng, train_state, env, num_envs, carry):
        hstate, obs, env_state, last_done, total_steps, _ = carry
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

        carry = (hstate, next_obs, env_state, done, total_steps + 1, value)
        return rng, carry, (obs, action, reward, done, log_prob, value, info)

    def sample_step(carry, _):
        rng, pro_carry, adv_carry, step_count, (pro_traj_idxs, adv_traj_idxs), pro_not_done = carry

        # Take Adv Step
        rng, next_adv_carry, adv_traj = sample_agent_step(rng, adv_train_state, adv_env, num_envs, adv_carry)
        adv_action = adv_traj[1]
        TOTAL_OBJECT_TYPES = 5
        edit_type_idx = adv_action[..., 1]

        # Determine which steps to take
        adv_obs = adv_carry[1]
        take_adv_step = ~jnp.all(~adv_obs.action_mask[0], axis=1)

        take_pro_step = jnp.logical_and(jnp.logical_and(~take_adv_step.repeat(num_pro_traj, axis=0), pro_carry[4] < max_student_episode_length), pro_not_done)
        
        # Replace Adv Carry where necessary
        take_either_step = jnp.logical_or(take_adv_step, take_pro_step.reshape(num_envs, num_pro_traj).any(axis=1))
        next_adv_hstate, _, next_adv_env_state, next_adv_last_done, next_adv_steps, next_adv_value = jax.tree_map(Partial(replace_carry, new_carry=take_either_step), next_adv_carry, adv_carry)
        next_adv_env_state = jax.tree_map(Partial(replace_carry, new_carry=take_adv_step), next_adv_env_state, adv_carry[2])

        def student_step(rng, train_state, carry, num_traj, take_step, not_done):
            # Update student envs
            hstate, obs, env_state, last_done, steps, last_value = carry
            repeated_levels = jax.tree_map(lambda x: x.repeat(num_traj, axis=0), next_adv_env_state.level)
            obs, env_state = jax.vmap(env.update_state_from_level, in_axes=(0, 0))(repeated_levels, env_state)
            carry = hstate, obs, env_state, last_done, steps, last_value
            location = jnp.concatenate([env_state.env_state.has_key[:, None], env_state.env_state.agent_dir[:, None], env_state.env_state.agent_pos], axis=-1)

            # Take Pro Step
            rng, next_carry, traj = sample_agent_step(rng, train_state, env, num_envs*num_traj, carry)
            next_carry = jax.tree_map(Partial(replace_carry, new_carry=take_step), next_carry, carry)
            #not_done = jnp.logical_and(not_done, ~jnp.logical_and(traj[3], take_step)) #add this back for single traj

            # Update adv obs
            next_value = next_carry[5].reshape(num_envs, num_traj)
            next_env_state = next_carry[2]
            next_locs = next_env_state.env_state.agent_pos.reshape(num_envs, num_traj, 2)
            next_dirs = next_env_state.env_state.agent_dir.reshape(num_envs, num_traj)
            next_goal_placed = next_env_state.env_state.goal_placed.reshape(num_envs, num_traj)
            next_key_placed = next_env_state.env_state.key_placed.reshape(num_envs, num_traj)
            next_door_placed = next_env_state.env_state.door_placed.reshape(num_envs, num_traj)

            return rng, next_carry, location, not_done, (next_locs, next_dirs, next_goal_placed, next_key_placed, next_door_placed, next_value), traj
        rng, next_pro_carry, pro_location, pro_not_done, (next_locs, next_dirs, next_goal_placed, next_key_placed, next_door_placed, next_pro_value), pro_traj = student_step(rng, pro_train_state, pro_carry, num_pro_traj, take_pro_step, pro_not_done)
        
        next_adv_env_state = next_adv_env_state.replace(agent_locs=next_locs, agent_dirs=next_dirs)
        not_done = pro_not_done.reshape(num_envs, num_pro_traj)
        rng, _rng = jax.random.split(rng)
        next_adv_obs = jax.vmap(adv_env.get_finished_obs, in_axes=(0, 0, 0))(jax.random.split(_rng, num_envs), next_adv_env_state, not_done)
        place_goal = ~jnp.logical_and(take_adv_step, edit_type_idx <= 1)
        # action_mask_1, action_mask_2 = next_adv_obs.action_mask
        # action_mask_2 = jnp.where(
        #     ~place_goal[:, None],
        #     jnp.logical_and(action_mask_2, jnp.array([True, True, False, False, False])),
        #     action_mask_2
        # )
        # action_mask = (action_mask_1, action_mask_2)
        next_adv_carry = next_adv_hstate, next_adv_obs.replace(agent_values=next_pro_value, place_goal=jnp.logical_and(place_goal, ~next_adv_env_state.level.goal_placed), goal_placed=next_goal_placed, key_placed=next_key_placed, door_placed=next_door_placed), next_adv_env_state, next_adv_last_done, next_adv_steps, next_adv_value
        
        # Set traj idx array
        pro_traj_idxs = pro_traj_idxs.at[pro_carry[4], jnp.arange(num_pro_envs)].set(step_count)
        adv_traj_idxs = adv_traj_idxs.at[adv_carry[4], jnp.arange(num_envs)].set(step_count)

        return (rng, next_pro_carry, next_adv_carry, step_count+1, (pro_traj_idxs, adv_traj_idxs), pro_not_done), ((pro_traj, adv_traj), pro_location, adv_carry[4], adv_carry[2])

    pro_carry = (
        ActorCritic.initialize_carry((num_pro_envs,)), 
        pro_init_obs, 
        pro_init_env_state, 
        jnp.zeros(num_pro_envs, dtype=bool),
        jnp.zeros(num_pro_envs, dtype=jnp.int32),
        jnp.zeros(num_pro_envs, dtype=jnp.float32)
    )

    adv_carry = (
        AdversaryActorCritic.initialize_carry((num_envs,)),
        adv_init_obs,
        adv_init_env_state,
        jnp.zeros(num_envs, dtype=bool),
        jnp.zeros(num_envs, dtype=jnp.int32),
        jnp.zeros(num_pro_envs, dtype=jnp.float32)
    )

    step_count = 0

    pro_traj_idxs = (max_student_episode_length+max_adv_episode_length-1)*jnp.ones((max_student_episode_length, num_pro_envs), dtype=jnp.int32)
    adv_traj_idxs = (max_student_episode_length+max_adv_episode_length-1)*jnp.ones((max_adv_episode_length, num_envs), dtype=jnp.int32)

    pro_not_done = jnp.ones(num_pro_envs, dtype=jnp.bool_)

    (rng, pro_carry, adv_carry, step_count, (pro_traj_idxs, adv_traj_idxs), pro_not_done), ((full_pro_traj, full_adv_traj), full_pro_locations, full_student_adv_idxs, full_student_adv_env_states) = jax.lax.scan(
        sample_step,
        (rng, pro_carry, adv_carry, step_count, (pro_traj_idxs, adv_traj_idxs), pro_not_done),
        None,
        length=max_student_episode_length+max_adv_episode_length,
    )

    def get_last_value(train_state, carry):
        hstate, last_obs, _, last_done, _, _ = carry
        x = jax.tree_map(lambda x: x[None, ...], (last_obs, last_done))
        _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)
        return last_value

    last_values = get_last_value(pro_train_state, pro_carry).squeeze(0), get_last_value(adv_train_state, adv_carry).squeeze(0)
    last_pro_env_state = pro_carry[2]
    last_pro_location = jnp.concatenate([last_pro_env_state.env_state.has_key[:, None], last_pro_env_state.env_state.agent_dir[:, None], last_pro_env_state.env_state.agent_pos], axis=-1)

    (pro_traj, pro_locations) = jax.tree_map(lambda x: x[pro_traj_idxs, jnp.arange(num_pro_envs)], (full_pro_traj, full_pro_locations))

    student_traj_idxs = pro_traj_idxs.reshape(-1, num_envs, num_pro_traj)
    student_adv_idxs = full_student_adv_idxs[student_traj_idxs.min(axis=2), jnp.arange(num_envs)]
    student_adv_env_states = jax.tree_map(lambda x : x[student_traj_idxs.min(axis=2), jnp.arange(num_envs)], full_student_adv_env_states)

    return rng, (pro_carry, adv_carry), (pro_traj, full_adv_traj), last_values, jnp.concatenate([pro_locations, last_pro_location[None, ...]], axis=0), student_adv_idxs, student_adv_env_states

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

        location = jnp.concatenate([env_state.env_state.has_key[:, None], env_state.env_state.agent_dir[:, None], env_state.env_state.agent_pos], axis=-1)
        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, ((obs, action, reward, done, log_prob, value, info), location)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), (traj, locations) = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)
    last_location = jnp.concatenate([last_env_state.env_state.has_key[:, None], last_env_state.env_state.agent_dir[:, None], last_env_state.env_state.agent_pos], axis=-1)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj, jnp.concatenate([locations, last_location[None, ...]], axis=0)

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

def masked_mean(arr, mask):
    return jnp.sum(arr * mask) / jnp.sum(mask)

def masked_std(arr, mask):
    mean = masked_mean(arr, mask)
    var = jnp.sum(mask*(arr - mean)**2)/jnp.sum(mask)
    return jnp.sqrt(var)

def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages, update_mask = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages, update_mask
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages, update_mask = minibatch
            action_mask = jnp.logical_and(update_mask, actions.reshape(actions.shape[0], actions.shape[1], -1)[..., 0] != -1)
            
            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = masked_mean(pi.entropy(), action_mask)

                ratio = jnp.exp(log_probs_pred - log_probs * action_mask)
                A = (advantages - masked_mean(advantages, action_mask)) / (masked_std(advantages, action_mask) + 1e-5)
                l_clip = masked_mean((-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)), action_mask)

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = masked_mean(0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2), update_mask)

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)

def compute_min_steps_to_goal(level, has_key, to_key=False, key_values=0):
    #wall_values = jnp.repeat(jnp.where(level.wall_map, jnp.inf, -jnp.inf)[None, ...], 4, axis=0)
    door_map = jnp.zeros_like(level.wall_map)
    door_map = jax.lax.select(
        jnp.logical_and(~has_key, level.door_placed == 1),
        door_map.at[level.door_pos[1], level.door_pos[0]].set(True),
        door_map
    )
    wall_values = jnp.repeat(jnp.where(jnp.logical_or(door_map, jnp.logical_or(level.wall_map, ~level.observation_map)), jnp.inf, -jnp.inf)[None, ...], 4, axis=0)
    max_height, max_width = level.wall_map.shape
    
    def compute_next(values):
        fwd_values = jnp.array([
            jnp.roll(values[0], -1, axis=1).astype(float).at[:,-1].set(jnp.inf),
            jnp.roll(values[1], -1, axis=0).astype(float).at[-1,:].set(jnp.inf),
            jnp.roll(values[2], 1, axis=1).astype(float).at[:,0].set(jnp.inf),
            jnp.roll(values[3], 1, axis=0).astype(float).at[0,:].set(jnp.inf),
        ])
        new_values = jnp.empty_like(values)
        for i in range(4):
            new_values = new_values.at[i].set(jnp.min(
                jnp.array([values[i], values[i-1] + 1, values[(i+1)%4] + 1, fwd_values[i] + 1]), axis=0
            ))
        return jnp.maximum(new_values, wall_values)
    
    def cond_fn(carry):
        values, next_values = carry
        return jnp.any(values != next_values)
    
    def body_fn(carry):
        _, values = carry
        return values, compute_next(values)
    
    values = jnp.full((4, max_height, max_width), jnp.inf)
    values = jax.lax.select(
        to_key,
        jax.lax.select(level.key_placed == 1, values.at[:, level.key_pos[1], level.key_pos[0]].set(key_values), values),
        jax.lax.select(level.goal_placed, values.at[:, level.goal_pos[1], level.goal_pos[0]].set(key_values), values)
    )
    
    return jax.lax.while_loop(cond_fn, body_fn, (values, compute_next(values)))[0]

NO_KL = False
GOAL_PROB = 0.01
WALL_PROB = (1 - 3*GOAL_PROB)/2

EMPTY_PROB_SEP = False
EMPTY_PROB = 0.7 - 3*GOAL_PROB
class DoubleCategorical(distrax.Distribution):
    def __init__(self, logits_1, logits_2):
        self.pi_1 = distrax.Categorical(logits=logits_1)
        self.pi_2 = distrax.Categorical(logits=logits_2)

        target_probs = jnp.array([WALL_PROB, WALL_PROB, GOAL_PROB, GOAL_PROB, GOAL_PROB])
        if EMPTY_PROB_SEP:
            target_probs = jnp.array([EMPTY_PROB, 0.3, GOAL_PROB, GOAL_PROB, GOAL_PROB])
        self.target_pi = distrax.Categorical(probs=target_probs)

    def _sample_n(self, key, n):
        key_1, key_2 = jax.random.split(key)
        samples_1 = self.pi_1._sample_n(key_1, n)
        samples_2 = self.pi_2._sample_n(key_2, n)
        return jnp.stack((samples_1, samples_2), axis=-1)

    def log_prob(self, value):
        log_prob_1 = self.pi_1.log_prob(value[..., 0])
        log_prob_2 = self.pi_2.log_prob(value[..., 1])
        return log_prob_1 + log_prob_2

    def entropy(self):
        #return self.pi_1.entropy() + self.pi_2.entropy()
        if NO_KL:
            return self.pi_1.entropy()
        return self.pi_1.entropy() - self.pi_2.kl_divergence(self.target_pi)

    def event_shape(self):
        return (2,)

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        
        embedding = jnp.concatenate((img_embed, dir_embed, obs.has_key[..., None]), axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)
        embedding = nn.LayerNorm()(embedding)

        actor_mean = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.LayerNorm()(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.LayerNorm()(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))


class AdversaryActorCritic(nn.Module):
    # The adversary's network architecture
    action_dim: Sequence[int]
    max_timesteps: int = 50
    student_max_timesteps: int = 250
    
    @nn.compact
    def __call__(self, inputs: Tuple[Observation, chex.Array], hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(jnp.concatenate((obs.image, obs.observation_map), axis=-1))
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        time_value = nn.Embed(self.max_timesteps + 1, 10, name="time_embed", embedding_init=orthogonal(1.0))(jnp.clip(obs.time, None, self.max_timesteps))
        student_time_value = nn.Embed(self.student_max_timesteps + 1, 10, name="student_time_embed", embedding_init=orthogonal(1.0))(jnp.clip(obs.agent_steps, None, self.student_max_timesteps))
        dirs_embedding = jax.nn.one_hot(obs.agent_dirs, 4).reshape(*obs.agent_dirs.shape[:2], -1)
        embedding = jnp.concatenate((img_embed, time_value, student_time_value, obs.agent_values, obs.place_goal[..., None], obs.goal_placed, obs.key_placed, obs.door_placed, dirs_embedding), axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)
        embedding = nn.LayerNorm()(embedding)

        actor_mean = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.LayerNorm()(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        actor_mean_0 = nn.Dense(25, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor10")(actor_mean)
        actor_mean_1 = nn.Dense(5, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor11")(actor_mean)

        # Mask out this
        actor_mean_0 = jnp.where(obs.action_mask[0], actor_mean_0, -jnp.inf)
        actor_mean_1 = jnp.where(obs.action_mask[1], actor_mean_1, -jnp.inf)
        pi = DoubleCategorical(logits_1=actor_mean_0, logits_2=actor_mean_1)

        critic = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.LayerNorm()(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
# endregion

# region checkpointing
def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
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
    overall_save_dir = os.path.join(os.getcwd(), f"checkpoints/{config['group']}", f"{config['run_name']}")
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
    wandb_config["goal_prob"] = GOAL_PROB
    wandb_config["no_kl"] = NO_KL
    wandb_config["empty_prob_separate"] = EMPTY_PROB_SEP
    tags = ["obs_gen", "local", "NVL", "key"]
    run = wandb.init(config=wandb_config, project=project, tags=tags, group=config["group"])
    
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
            "misc/mean_num_blocks": stats["mean_num_blocks"].mean(),
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
            "misc/prot_perf_mean": stats['pro_returns'].mean(),
            "misc/pro_num_episodes": stats['pro_eps'].mean(),
            "misc/pro_regret":   stats['pro_regret'].mean(),
            "misc/adv_perf_mean": stats['adv_returns'].mean(),
            "misc/pro_extra_perf_mean": stats['pro_extra_mean_returns'].mean(),
            "misc/pro_extra_perf_max": stats['pro_extra_max_returns'].mean(),
            "misc/pro_extra_num_episodes": stats['pro_extra_eps'].mean(),
            "misc/pro_extra_regret":   stats['pro_extra_regret'].mean(),
            "misc/key_optimal":   stats['key_optimal'].mean(),
            "misc/unsolvable":   stats['unsolvable'].mean(),
            "misc/unsolvable_approx":   stats['unsolvable_approx'].mean(),
        }
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        def make_caption(i):
            pro_mean_returns = jnp.round(stats['pro_returns'][-1][i], 2) # .flatten()
            pro_extra_mean_returns = jnp.round(stats['pro_extra_mean_returns'][-1][i], 2) # .flatten()
            pro_regret       = jnp.round(stats['pro_regret'][-1][i], 2) # .flatten()
            pro_extra_regret       = jnp.round(stats['pro_extra_regret'][-1][i], 2) # .flatten()
            key_optimal = stats['key_optimal'][-1][i]
            unsolvable_approx = stats['unsolvable_approx'][-1][i]
            adv_return = jnp.round(stats['adv_returns'][-1][i], 2) # .flatten()
            return f"P({pro_mean_returns:.2f}, {pro_extra_mean_returns:.2f})|R({pro_regret:.2f}, {pro_extra_regret:.2f})|Adv({adv_return:.2f})|K({key_optimal})|U({unsolvable_approx})"
    
        log_dict.update({f"images/levels": [wandb.Image(np.array(image), caption=make_caption(i)) for i, image in enumerate(stats["levels"][:32])]})

        if config["log_animations"]:
            # generation animations
            animations = []
            for i in range(8):
                frames, episode_length = stats["animated_levels"][0][:, i], stats["animated_levels"][1][i]
                frames = np.array(frames[:episode_length])
                animations.append(wandb.Video(frames, fps=4))
            log_dict.update({f"images/level_animations": animations})

            # animations
            for i, level_name in enumerate(config["eval_levels"]):
                frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
                frames = np.array(frames[:episode_length])
                log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)
    
    env = Maze(max_height=config['max_height'], max_width=config['max_width'], agent_view_size=5, normalize_obs=True)
    adv_env = LocalKeyMazeEditorRotateSplitAct(env, random_z_dimensions=config['adv_random_z_dimension'], zero_out_random_z=config['adv_zero_out_random_z'], num_agents=config["num_pro_traj"], agent_view_size=5)
    eval_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    adv_env_renderer = ObservedMazeRenderer(env, tile_size=8)
    ani_adv_env_renderer = LocalObservedMazeRenderer(env, tile_size=8)
    env_renderer = MazeRenderer(env, tile_size=8)
    eval_env_renderer = MazeRenderer(eval_env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params.replace(max_steps_in_episode=config['max_steps_in_episode'])
    eval_env_params = env.default_params
    adv_env_params = adv_env.default_params

    def sample_empty_level():
        w, h = env._env.max_width, env._env.max_height
        return ObservedLevel(
            wall_map=jnp.zeros((h, w), dtype=jnp.bool_),
            observation_map=jnp.zeros((h, w), dtype=jnp.bool_),
            width=w,
            height=h,
            
            # These values don't matter, as the adversary overwrites them.
            goal_pos=jnp.array([0, 0], dtype=jnp.uint32),
            agent_pos=jnp.array([1, 1], dtype=jnp.uint32),
            agent_dir=jnp.array(0, dtype=jnp.uint8),
            goal_placed=jnp.array(False, dtype=jnp.bool_),
        )
    
    def sample_random_init_level(rng):
        w, h = env._env.max_width, env._env.max_height
        observation_map = jnp.zeros((h, w), dtype=jnp.bool_)
        rng_pos, rng_dir = jax.random.split(rng)
        agent_pos = jax.random.randint(rng_pos, (2,), 0, jnp.array([h, w]), dtype=jnp.uint32)
        agent_dir = jax.random.randint(rng_dir, (), 0, 4, dtype=jnp.uint8)
        return ObservedLevel(
            wall_map=jnp.zeros((h, w), dtype=jnp.bool_),
            observation_map=observation_map.at[agent_pos[1], agent_pos[0]].set(True),
            width=w,
            height=h,
            
            # These values don't matter, as the adversary overwrites them.
            goal_pos=jnp.array([0, 0], dtype=jnp.uint32),
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            goal_placed=jnp.array(False, dtype=jnp.bool_),
        )

    def create_train_state(rng):
        def create_inner_train_state(rng, env, env_params, network_cls, prefix, network_kws={}):
            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config[f"{prefix}num_minibatches"] * config[f"{prefix}epoch_ppo"]))
                    / config["num_updates"]
                )
                return config[f"{prefix}lr"] * frac
            obs, _ = env.reset_to_level(rng, sample_empty_level(), env_params)
            obs = jax.tree_map(
                lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
                obs,
            )
            init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
            network = network_cls(env.action_space(env_params).n, **network_kws)
            network_params = network.init(rng, init_x, network_cls.initialize_carry((config["num_train_envs"],)))
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
            pro_train_state = create_inner_train_state(rng_pro, env, env_params, ActorCritic, "student_"),
            adv_train_state = create_inner_train_state(rng_adv, adv_env, adv_env_params, AdversaryActorCritic, "adv_", network_kws={"max_timesteps": config["adv_num_steps"], "student_max_timesteps": config["max_steps_in_episode"]}),
        )

    def train_step(carry, _):
        def get_rollout(traj, traj_idxs, last_value, prefix, reward_dones=None):
            obs, actions, rewards, dones, log_probs, values, info = traj
            if reward_dones == None:
                reward_dones = dones

            advantages, targets = compute_gae(config[f"{prefix}gamma"], config[f"{prefix}gae_lambda"], last_value, values, rewards, reward_dones, traj_idxs)
            update_mask = jnp.tile(jnp.arange(actions.shape[0])[:, None], (1, actions.shape[1])) < traj_idxs
            return (obs, actions, dones, log_probs, values, targets, advantages, update_mask), (dones, rewards, update_mask)
        
        def update(rng, train_state, init_hstate, rollout, prefix, num_envs):
            # Returns: (rng, train_state), losses
            return update_actor_critic_rnn(
                rng,
                train_state,
                init_hstate,
                rollout,
                num_envs,
                config[f"{prefix}num_steps"],
                config[f"{prefix}num_minibatches"],
                config[f"{prefix}epoch_ppo"],
                config[f"{prefix}clip_eps"],
                config[f"{prefix}entropy_coeff"],
                config[f"{prefix}critic_coeff"],
                update_grad=True,
            )
        
        def get_agent_min_steps_to_goal(env_state, agent_locations):
            level = env_state.level
            distances_to_goal = compute_min_steps_to_goal(level, jnp.array(True))
            distances_to_goal_no_key = compute_min_steps_to_goal(level, jnp.array(False))

            key_values = distances_to_goal[:, level.key_pos[1], level.key_pos[0]]
            distances_via_key = compute_min_steps_to_goal(level, jnp.array(False), jnp.array(True), key_values)

            all_optimal_distances = jnp.stack((
                jnp.minimum(distances_via_key, distances_to_goal_no_key),
                distances_to_goal
            )).swapaxes(2, 3)

            key_optimal = (distances_via_key < distances_to_goal_no_key)[level.agent_dir, level.agent_pos[1], level.agent_pos[0]]

            agent_optimal_distances = jax.vmap(
                jax.vmap(lambda x : all_optimal_distances[tuple(x)],
                        in_axes=(0)),
                in_axes=(0)
            )(agent_locations)

            return agent_optimal_distances, key_optimal
        
        rng, train_state = carry
        
        # Initialise Levels
        rng, _rng = jax.random.split(rng)
        empty_levels = jax.vmap(sample_random_init_level)(jax.random.split(_rng, config["num_train_envs"]))

        # Gather Trajectories
        (rng, (pro_carry, adv_carry), (pro_traj, adv_traj), (pro_last_value, adv_last_value), pro_locations, student_adv_idxs, student_adv_env_states) = sample_trajectories(
            rng,
            env,
            env_params,
            adv_env,
            adv_env_params,
            train_state,
            empty_levels,
            config["num_train_envs"],
            config["num_pro_traj"],
            config["student_num_steps"],
            config["adv_num_steps"]
        )

        def rollout(rng, env, env_params, train_state, init_hstate, levels, num_steps, prefix, num_envs=config["num_train_envs"]):
            # Single rollout
            rng, _rng = jax.random.split(rng)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, num_envs), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info), locations
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                num_envs,
                num_steps,
            )
            traj_idxs = jnp.ones(num_envs) * config['student_num_steps']
            advantages, targets = compute_gae(config[f"{prefix}gamma"], config[f"{prefix}gae_lambda"], last_value, values, rewards, dones, traj_idxs)
            update_mask = jnp.tile(jnp.arange(actions.shape[0])[:, None], (1, actions.shape[1])) < traj_idxs
            return (obs, actions, dones, log_probs, values, targets, advantages, update_mask), (dones, rewards, locations)

        # Do Addtional Rollouts on Fixed Level
        rng, _rng = jax.random.split(rng)
        adv_env_state_modified = adv_carry[2].replace(level=adv_carry[2].level.replace(observation_map=jnp.ones_like(adv_carry[2].level.observation_map)))
        pro_extra_rollout, (dones, rewards, locations) = rollout(_rng, env, env_params, train_state.pro_train_state, ActorCritic.initialize_carry((config["num_train_envs"],)), adv_env_state_modified.level, config['student_num_steps'], 'student_')
        pro_extra_mean_returns, pro_extra_max_returns, pro_extra_eps = compute_max_mean_returns_epcount(dones, rewards)

        agent_extra_optimal_distances, key_optimal = jax.vmap(
            get_agent_min_steps_to_goal,
            in_axes=(0, 1)
        )(adv_env_state_modified, locations.reshape(-1, config['num_train_envs'], 1, 4))
        agent_extra_optimal_distances = agent_extra_optimal_distances.swapaxes(0, 1).squeeze(-1)

        agent_extra_regret = jnp.nan_to_num((1 + agent_extra_optimal_distances[1:]) - (agent_extra_optimal_distances[:-1]), 0) * (1 - dones)
        mean_extra_ep_regret = agent_extra_regret.sum(axis=0)/pro_extra_eps

        # Get Rollouts for Protagonist and Antagonist
        def get_trajectory_metrics(traj, carry, last_value, locations, num_traj):
            rollout, (dones, rewards, update_mask) = get_rollout(traj, carry[4], last_value, 'student_')
            returns, max_returns, eps = compute_max_mean_returns_epcount(dones * update_mask, rewards * update_mask)
            mean_returns = returns.reshape((config['num_train_envs'], num_traj)).mean(axis=1)
            max_returns = max_returns.reshape((config['num_train_envs'], num_traj)).max(axis=1)

            agent_optimal_distances, _ = jax.vmap(
                get_agent_min_steps_to_goal,
                in_axes=(0, 1)
            )(adv_carry[2], locations.reshape(-1, config['num_train_envs'], num_traj, 4))
            agent_optimal_distances = agent_optimal_distances.swapaxes(0, 1)

            target_shape = (config['student_num_steps'], config['num_train_envs'], num_traj)
            agent_regret = jnp.nan_to_num((1 + agent_optimal_distances[1:]) - (agent_optimal_distances[:-1]), 0) * (1 - traj[3].reshape(target_shape)) * update_mask.reshape(target_shape)
            regret = agent_regret.mean(axis=2).sum(axis=0) / eps
            mean_rewards = (rewards * update_mask).reshape(target_shape).mean(axis=2)

            return rollout, mean_rewards, mean_returns, max_returns, eps, regret, agent_regret, update_mask.reshape(target_shape)

        pro_rollout, pro_rewards, pro_mean_returns, pro_max_returns, pro_eps, pro_regret, step_regret, step_count = get_trajectory_metrics(pro_traj, pro_carry, pro_last_value, pro_locations, config['num_pro_traj'])

        obs, actions, rewards, dones, log_probs, values, info = adv_traj
        dones = jnp.zeros_like(dones).at[adv_carry[4]-1, jnp.arange(dones.shape[1])].set(True)
        adv_pro_dones = jnp.logical_or(jnp.zeros_like(dones).at[student_adv_idxs-1, jnp.arange(rewards.shape[1])].set(pro_traj[3]), dones)

        # Compute Actual Level Solvability
        shortest_path = agent_extra_optimal_distances[0]
        unsolvable = shortest_path == jnp.inf

        # Compute Approximate Solvability
        agent_failed = pro_mean_returns == 0

        # Calculate Metrics
        _, _, pro_dones, _, pro_values, targets, advantages, update_mask = pro_rollout
        PVL = jnp.maximum((advantages * update_mask), 0)
        MaxMC = pro_max_returns - pro_values

        def get_clipped_advantages(traj, last_value, prefix, num_envs=config["num_train_envs"]):
            obs, actions, rewards, dones, log_probs, values, info = traj
            clipped_advantages, _ = compute_clipped_gae(config[f"{prefix}gamma"], config[f"{prefix}gae_lambda"], last_value, values, rewards, dones, None, use_max_value=True)
            return clipped_advantages
        clipped_advantages = get_clipped_advantages(pro_traj, pro_last_value, 'student_')
        clipped_advantages_reward = clipped_advantages * update_mask * (~agent_failed)

        if config['score_function'] == "MaxMC":
            optimisation_metric = MaxMC
        elif config['score_function'] == "pvl":
            optimisation_metric = PVL
        elif config['score_function'] == "mna":
            optimisation_metric = -clipped_advantages_reward
        else:
            raise ValueError(f"Unknown score function: {config['score_function']}")
        
        sparse_rewards = jnp.zeros(rewards.shape).at[adv_carry[4]-1, jnp.arange(rewards.shape[1])].set(optimisation_metric.sum(axis=0))
        dense_rewards = jnp.zeros(rewards.shape).at[student_adv_idxs-1, jnp.arange(rewards.shape[1])].add(optimisation_metric)
        rewards = config["reward_mix"] * dense_rewards + (1 - config["reward_mix"]) * sparse_rewards

        adv_traj = obs, actions, rewards, dones, log_probs, values, info
        adv_rollout, _ = get_rollout(adv_traj, adv_carry[4], adv_last_value, 'adv_')

        (rng, pro_train_state), pro_losses = update(rng, train_state.pro_train_state, ActorCritic.initialize_carry((config["num_train_envs"]*config["num_pro_traj"],)), pro_rollout, "student_", num_envs=config["num_train_envs"]*config["num_pro_traj"])
        (rng, adv_train_state), adv_losses = update(rng, train_state.adv_train_state, AdversaryActorCritic.initialize_carry((config["num_train_envs"],)), adv_rollout, "adv_", num_envs=config["num_train_envs"])

        adv_last_env_state = adv_carry[2]
        levels = adv_last_env_state.level

        episode_lengths = pro_carry[4].reshape(config['num_train_envs'], config['num_pro_traj']).max(axis=1)

        metrics = {
            "pro_losses": jax.tree_map(lambda x: x.mean(), pro_losses),
            "adv_losses": jax.tree_map(lambda x: x.mean(), adv_losses),
            "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
            "pro_returns": pro_mean_returns,
            "pro_regret":       pro_regret,
            "adv_returns":   rewards.sum(axis=0),
            "pro_eps": pro_eps,
            "levels": levels,
            "animated_levels": (student_adv_env_states, episode_lengths),
            "pro_extra_mean_returns": pro_extra_mean_returns,
            "pro_extra_max_returns": pro_extra_max_returns,
            "pro_extra_eps": pro_extra_eps,
            "pro_extra_regret": mean_extra_ep_regret,
            "key_optimal": key_optimal,
            "unsolvable": unsolvable,
            "unsolvable_approx": agent_failed,
        }

        train_state = train_state.replace(
            update_count=train_state.update_count + 1,
            pro_train_state=pro_train_state,
            adv_train_state=adv_train_state,
        )
        return (rng, train_state), metrics
    
    def eval(rng, train_state):
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, eval_env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            eval_env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            eval_env_params.max_steps_in_episode,
        )
        mask = jnp.arange(eval_env_params.max_steps_in_episode)[..., None] < episode_lengths
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
        images = jax.vmap(jax.vmap(eval_env_renderer.render_state, (0, None)), (0, None))(states, eval_env_params) # (num_steps, num_eval_levels, ...)
        frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        level_states, level_ep_lengths = jax.tree_map(lambda x: x[-1], metrics["animated_levels"])
        level_images = jax.vmap(jax.vmap(ani_adv_env_renderer.render_state, (0, None)), (0, None))(level_states, adv_env_params)
        level_frames = level_images.transpose(0, 1, 4, 2, 3)

        metrics["update_count"] = train_state.update_count
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["levels"] = jax.vmap(adv_env_renderer.render_level, (0, None))(jax.tree_map(lambda x: x[-1], metrics["levels"]), env_params)
        metrics["animated_levels"] = (level_frames, level_ep_lengths)

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
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config['eval_levels'])
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval': return eval_checkpoint(config) # evaluate and exit early

    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    logger.info('Training')
    
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    start_time = time.time()
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics)
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(int(metrics['update_count']), args=ocp.args.StandardSave(runner_state[1]))
            #checkpoint_manager.wait_until_finished()
    return runner_state[1]

@hydra.main(config_path="config", config_name="main_obs_gen", version_base=None)
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
