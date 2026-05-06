import jax
import jax.numpy as jnp
import chex
from flax import struct

from jaxued.environments.maze.env import DIR_TO_VEC, make_maze_map
from jaxued.environments.underspecified_env import EnvState, Level

@struct.dataclass
class AdvObservation:
    image: chex.Array
    agent_dir: int
    has_key: chex.Array
    agent_action: int
    agent_value: float
    step_num: int
    hidden: chex.Array

EMPTY = jnp.array([0.1, 0, 0])
WALL = jnp.array([0.2, 0.5, 0])
GOAL = jnp.array([0.8, 0.1, 0])
KEY = jnp.array([0.5, 0.1, 0])
KEY_PICKED_UP = jnp.array([0.5, 0.2, 0])
DOOR = jnp.array([0.4, 0.1, 0])
DOOR_UNLOCKED = jnp.array([0.4, 0.2, 0])

def get_tile_dist(obs_image):
    tile_array = jnp.stack([EMPTY, WALL, GOAL, KEY, KEY_PICKED_UP, DOOR, DOOR_UNLOCKED])
    img = obs_image[:, :, None, :]
    t_a = tile_array[None, None, ...]
    tile_dist = jnp.mean((img - t_a) ** 2, axis=-1)
    return tile_dist

def get_rot_tile_dist(obs):
    tile_dist = get_tile_dist(obs.image)
    tile_dist = jnp.flip(tile_dist, axis=0)
    tile_dist = (obs.agent_dir == 0)*jnp.rot90(tile_dist, 2) + \
                   (obs.agent_dir == 1)*jnp.rot90(tile_dist, 1) + \
                   (obs.agent_dir == 2)*jnp.rot90(tile_dist, 0) + \
                   (obs.agent_dir == 3)*jnp.rot90(tile_dist, -1)
                   
    return tile_dist

def get_target(tile_dist, closest_tiles, target):
    is_target = closest_tiles == target
    has_tile = jnp.any(is_target, axis=(1,2))
    masked_dist = jnp.where(is_target[...], tile_dist[..., target], jnp.inf)

    num_batches, height, _ = masked_dist.shape
    masked_dist = masked_dist.reshape(num_batches, -1)
    max_idx = jnp.argmin(masked_dist, axis=-1)
    x_idx = max_idx % height
    y_idx = max_idx // height
    return has_tile, x_idx, y_idx

def update_env_state(env_state, closest_tile, tile_updates):
    goal_update, key_update, door_update = tile_updates

    dir_vec = DIR_TO_VEC[env_state.agent_dir]
    agent_view_size = closest_tile.shape[0]
            
    obs_fwd_bound1 = env_state.agent_pos
    obs_fwd_bound2 = env_state.agent_pos + dir_vec*(agent_view_size-1)

    side_offset = agent_view_size//2
    obs_side_bound1 = env_state.agent_pos + (dir_vec == 0)*side_offset
    obs_side_bound2 = env_state.agent_pos - (dir_vec == 0)*side_offset

    all_bounds = jnp.stack([obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2])
    height, width = env_state.wall_map.shape

    # Clip obs to grid bounds appropriately
    padding = agent_view_size-1
    xmin, ymin = jnp.min(all_bounds, 0) + padding

    padded_wall_map = jnp.pad(env_state.wall_map, padding, mode='constant', constant_values=True)
    padded_obs_map = jnp.pad(env_state.observation_map, padding, mode='constant', constant_values=True)

    def update_tile_pos(new_tile_update, old_tile_placed, old_tile_pos):
        has_tile, x_idx, y_idx = new_tile_update
        new_tile_x = xmin + x_idx - padding
        new_tile_y = ymin + y_idx - padding
        valid = has_tile * \
                (old_tile_placed == 0) * \
                (new_tile_x >= 0) * \
                (new_tile_x < width) * \
                (new_tile_y >= 0) * \
                (new_tile_y < height) * \
                (jax.lax.dynamic_slice(padded_obs_map, (new_tile_y + padding, new_tile_x + padding), (1, 1)).squeeze((0, 1)) == False)
        tile_pos = jnp.where(valid, jnp.array([new_tile_x, new_tile_y], dtype=jnp.uint32), old_tile_pos)
        tile_placed = jnp.where(valid, valid.astype(old_tile_placed.dtype), old_tile_placed)
        return tile_placed, tile_pos

    goal_placed, goal_pos = update_tile_pos(goal_update, env_state.goal_placed, env_state.goal_pos)
    key_placed, key_pos = update_tile_pos(key_update, env_state.key_placed, env_state.key_pos)
    door_placed, door_pos = update_tile_pos(door_update, env_state.door_placed, env_state.door_pos)

    old_wall_map = jax.lax.dynamic_slice(padded_wall_map, (ymin, xmin), (agent_view_size, agent_view_size))
    old_obs_map = jax.lax.dynamic_slice(padded_obs_map, (ymin, xmin), (agent_view_size, agent_view_size))

    new_wall_map = jnp.where(old_obs_map, old_wall_map, closest_tile == 1)

    padded_wall_map = jax.lax.dynamic_update_slice(padded_wall_map, new_wall_map, (ymin, xmin))
    wall_map = padded_wall_map[padding:-padding, padding:-padding]

    padded_obs_map = jax.lax.dynamic_update_slice(padded_obs_map, jnp.ones((agent_view_size, agent_view_size), dtype=bool), (ymin, xmin))
    obs_map = padded_obs_map[padding:-padding, padding:-padding]

    new_env_state = env_state.replace(
        wall_map=wall_map, 
        observation_map=obs_map,
        goal_pos=goal_pos,
        goal_placed=goal_placed,
        key_pos=key_pos,
        key_placed=key_placed,
        door_pos=door_pos,
        door_placed=door_placed,
    )

    maze_map = make_maze_map(new_env_state, agent_view_size-1)
    new_env_state = new_env_state.replace(maze_map=maze_map)

    return new_env_state, old_obs_map.all()

def update_env_state_from_obs(env_state, obs):
    tile_dist = jax.vmap(get_rot_tile_dist, in_axes=(0,))(obs)
    closest_tiles = jnp.argmin(tile_dist, axis=-1)

    goal_updates = get_target(tile_dist, closest_tiles, target=2)
    key_updates = get_target(tile_dist, closest_tiles, target=3)
    door_updates = get_target(tile_dist, closest_tiles, target=5)

    new_env_state, no_updates = jax.vmap(
        update_env_state, 
        in_axes=(0, 0, 0)
    )(env_state, closest_tiles, (goal_updates, key_updates, door_updates))

    return new_env_state, no_updates


def update_level_from_state(env_state: EnvState, level: Level):
    new_level = level.replace(
        wall_map=env_state.wall_map,
        goal_placed=env_state.goal_placed,
        goal_pos=env_state.goal_pos,
        door_placed=jnp.minimum(env_state.door_placed, 1),
        door_pos=env_state.door_pos,
        key_placed=jnp.minimum(env_state.key_placed, 1),
        key_pos=env_state.key_pos,
    )
    return new_level

