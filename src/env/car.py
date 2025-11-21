import jax
from jax import numpy as jnp
import equinox
import chex

from functools import partial
from typing import Tuple, Dict, Any

import numpy as np

from env.tasks import BaseTaskSampler
from utils import utils

from env.base import BaseEnv, BaseEnvObservation, BaseEnvParams, BaseEnvState

# NOTE:
# All colliders are modeled as cicles of radius 1!

np.set_printoptions(linewidth=300, threshold=np.inf)

RADIUS = 1.0  # radius of each circle
NUM_RAY_SENSORS = 8
FOV = jnp.pi  # used to get 360 degree vision
OBSTACLE_COST = 1000  # we are using int16, so choose this carefully
FPS = 60

MAX_STEER = np.radians(30)  # 30 degrees
ACCELERATION = 0.2
FRICTION = 0.03
USELITEL_RULA = np.radians(5)
WHEELBASE = 10

# TODO: collision
# TODO: observation. why ray is flat
# TODO: termination
# TODO: reset, autoreset wrapper


# env params
class RoboCarEnvParams(BaseEnvParams):
    pass

# env state
class RoboCarEnvState(BaseEnvState):
    # Direction of a car
    theta: chex.Scalar
    # Rotation of front axis
    phi: chex.Scalar
    # Car speed
    agent_speed: chex.Scalar


# env observation
class RoboCarEnvObservation(BaseEnvObservation):
    pass


class RoboCarEnv(BaseEnv):

    # abstract
    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5,
        num_ray_sensors: float = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: float = FPS,
    ) -> Tuple[RoboCarEnvParams, RoboCarEnvState]:

        if discretization_scale != 1:
            raise Exception("discretization_scale != 1 is not implemented!")

        task = BaseTaskSampler(map_id)
        # TODO: get map height and width
        env_params = RoboCarEnvParams(
            max_steps_in_episode=max_steps,
            fps=fps,
            step_size=1 / fps,
            map_height_width=(task.height, task.width),
            nav_grid_shape_dtype=(
                (task.height * discretization_scale, task.width * discretization_scale),
                jnp.float32,
            ),
            path_array_shape_dtype=((path_length, 2), jnp.float32),
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            agent_FOV=fov,
            num_ray_sensors=num_ray_sensors,
        )

        shape, dtype = env_params.path_array_shape_dtype

        convert_to_world_f = partial(
            utils.convert_to_world_view, map_shape=env_params.map_height_width
        )
        convert_to_world_vmap = jax.vmap(convert_to_world_f, in_axes=(0))

        init_state = RoboCarEnvState(
            time=jnp.asarray(0),
            agent_forward_dir=jnp.array([-1, 0], dtype=jnp.float32),
            goal_pos=convert_to_world_f(task.goal_pos).astype(jnp.float32),
            agent_pos=convert_to_world_f(task.agent_pos).astype(jnp.float32),
            theta=jnp.pi/2,
            phi=0.0,
            agent_speed=0.0,
            # agent_goal_dir=task.agent_goal_dir,
            static_obstacles=convert_to_world_vmap(task.static_obstacles).astype(
                jnp.float32
            ),
            kinematic_obstacles=convert_to_world_vmap(task.kinematic_obstacles).astype(
                jnp.float32
            ),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(
                jnp.float32
            ),
            path_array=jnp.zeros(shape=shape, dtype=dtype),
        )
        obstacles = jnp.concatenate(
            [init_state.static_obstacles, init_state.kinematic_obstacles], axis=0
        )
        path_array = RoboCarEnv._find_path(
            init_state.agent_pos, init_state.goal_pos, obstacles, env_params
        )
        # replace path_array
        init_state = equinox.tree_at(lambda t: t.path_array, init_state, path_array)

        return env_params, init_state

    @partial(jax.jit, static_argnames=("env_params",))
    def reward_function(
        env_state: RoboCarEnvState,
        env_params: RoboCarEnvParams, 
    ):
        till_goal = 0
        prev_point = env_state.agent_pos
        for point in env_state.path_array:
            till_goal += jnp.linalg.norm(point - prev_point)
            prev_point = point
        reward = 1 / till_goal - env_state.time
        return reward

    # abstract
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey, env_params: RoboCarEnvParams, init_state: RoboCarEnvState | None
    ) -> Tuple[chex.Array, RoboCarEnvState]:

        if init_state is not None:
            state = init_state
        else:
            raise Exception("init_state is None is Not implemented!")
        obs = RoboCarEnv.get_observation(state, env_params)
        return obs, state

    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: RoboCarEnvState,
        action: chex.Scalar | chex.Array,
        env_params: RoboCarEnvParams,
    ) -> Tuple[
        chex.Array, RoboCarEnvState, chex.Scalar | chex.Array, chex.Array, Dict[Any, Any]
    ]:
        # concatenate static and dynamic obstacles
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )

        # Check Done.
        goal_done = RoboCarEnv._check_goal(env_state.agent_pos, env_state.goal_pos)
        collision_done = RoboCarEnv._check_collisions(env_state.agent_pos, obstacles)
        time_done = env_state.time >= env_params.max_steps_in_episode

        done = jnp.logical_or(goal_done, jnp.logical_or(collision_done, time_done))

        # Calculate reward
        reward = RoboCarEnv.reward_function(env_state, env_params)

        # --- CORRECTED KINEMATIC UPDATE ---
        dt = 1.0 / env_params.fps  # Proper time step
        TIME_SCALE = 60.0  # Match your Pygame dt * 60
        ACCELERATION_SCALE = 90.0
        THETA_SCALE = 10.0
        
        # Process action (acceleration and steering)
        dv, dphi = action
        
        # Update agent speed
        v = env_state.agent_speed + dv * ACCELERATION * dt * ACCELERATION_SCALE
        
        # Update steering angle with limits
        phi = env_state.phi + dphi * dt * TIME_SCALE - jnp.sign(env_state.phi) * USELITEL_RULA
        phi = jnp.clip(phi, -MAX_STEER, MAX_STEER)
        phi = jnp.where(jnp.abs(phi) < 0.001, 0.0, phi)

        # Apply friction
        v_sign = jnp.sign(v)
        v_magnitude = jnp.abs(v)
        v_magnitude = jnp.maximum(0.0, v_magnitude - FRICTION * dt * TIME_SCALE)
        v = v_sign * v_magnitude
        
        # Clamp near-zero velocities to 0
        v = jnp.where(jnp.abs(v) < 0.01, 0.0, v)
        
        # Kinematic equations (with proper time step)
        x = env_state.agent_pos[0] + v * jnp.cos(env_state.theta) * dt * TIME_SCALE
        y = env_state.agent_pos[1] + v * jnp.sin(env_state.theta) * dt * TIME_SCALE
        theta = env_state.theta - (v / WHEELBASE) * jnp.tan(phi) * dt * TIME_SCALE * THETA_SCALE
        new_agent_pos = jnp.array([x, y], dtype=jnp.float32)
        
        # Move obstacles
        kinematic_obstacles = RoboCarEnv._move_kinematic_obstacles(env_state, env_params)
        
        # Increment time
        new_time = env_state.time + 1

        # path array. update every second.
        pred = jnp.mod(new_time, env_params.fps)
        path_array = jax.lax.cond(
            pred,
            lambda _: RoboCarEnv._find_path(
                new_agent_pos, env_state.goal_pos, obstacles, env_params
            ),
            lambda _: env_state.path_array,
            None,
        )

        new_state = RoboCarEnvState(
            time=new_time,
            agent_forward_dir=jnp.array([jnp.sin(theta), jnp.cos(theta)], dtype=jnp.float32),
            goal_pos=env_state.goal_pos,
            agent_pos=new_agent_pos,
            theta=theta,
            phi=phi,
            agent_speed=v,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path_array,
        )
        obs = RoboCarEnv.get_observation(new_state, env_params)
        info = {
            "time": new_time, 
            "reward": reward,
            "distance_to_path": obs.distance_to_path, 
            "direction_of_path": obs.direction_of_path, 
            "theta": theta,
            "phi": phi,
            "speed": v
        }
        # NOTE: info is used for evaluation

        return obs, new_state, reward, done, info

    def get_observation(
        env_state: RoboCarEnvState, env_params: RoboCarEnvParams
    ) -> RoboCarEnvObservation:
        # find the closest waypoints
        dists = jnp.linalg.norm(env_state.path_array - env_state.agent_pos, axis=1)
        closest_idx = jnp.argmin(dists)
        # construct a vector with the next after closest. it is safe since we always have excess number of waypoints
        path_vec = (
            env_state.path_array.at[closest_idx + 1].get()
            - env_state.path_array.at[closest_idx].get()
        )
        # normalize it
        path_dir = path_vec / (jnp.linalg.norm(path_vec) + 1e-8)

        # project agent's position to that vector to find distance
        anchor = env_state.path_array.at[closest_idx].get()
        rel_vec = env_state.agent_pos - anchor
        proj_len = jnp.dot(rel_vec, path_dir)
        proj_point = anchor + proj_len * path_dir
        distance_to_path = jnp.linalg.norm(env_state.agent_pos - proj_point)

        # perception:
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )
        rays, ray_perceptions = RoboCarEnv._collision_ray_intersections(
            env_state.agent_pos, env_state.agent_forward_dir, obstacles, env_params
        )
        # rays start at the center of the agent. we need to compensate for that
        ray_perceptions = ray_perceptions
        # rays were normalized. now we need to give them magnitude
        # jax.debug.print("rays before scale {x}", x=rays)
        rays = rays * ray_perceptions[:, None]

        return RoboCarEnvObservation(
            distance_to_path=distance_to_path,
            direction_of_path=path_dir,
            collision_rays=rays,
        )
