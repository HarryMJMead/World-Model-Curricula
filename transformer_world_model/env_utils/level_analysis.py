import jax
import jax.numpy as jnp

def compute_min_steps_to_goal(level, has_key, to_key=False, key_values=0):
    #wall_values = jnp.repeat(jnp.where(level.wall_map, jnp.inf, -jnp.inf)[None, ...], 4, axis=0)
    door_map = jnp.zeros_like(level.wall_map)
    door_map = jax.lax.select(
        jnp.logical_and(~has_key, level.door_placed == 1),
        door_map.at[level.door_pos[1], level.door_pos[0]].set(True),
        door_map
    )
    wall_values = jnp.repeat(jnp.where(jnp.logical_or(door_map, level.wall_map), jnp.inf, -jnp.inf)[None, ...], 4, axis=0)
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


def get_shortest_path(level):
    distances_to_goal = compute_min_steps_to_goal(level, jnp.array(True))
    distances_to_goal_no_key = compute_min_steps_to_goal(level, jnp.array(False))

    key_values = distances_to_goal[:, level.key_pos[1], level.key_pos[0]]
    distances_via_key = compute_min_steps_to_goal(level, jnp.array(False), jnp.array(True), key_values)

    all_optimal_distances = jnp.stack((
        jnp.minimum(distances_via_key, distances_to_goal_no_key),
        distances_to_goal
    ))

    key_optimal = (distances_via_key < distances_to_goal_no_key)[level.agent_dir, level.agent_pos[1], level.agent_pos[0]]
    agent_pos = level.agent_pos
    agent_dir = level.agent_dir

    return all_optimal_distances[0, agent_dir, agent_pos[1], agent_pos[0]], key_optimal