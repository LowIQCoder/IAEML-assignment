import jax

from env.car import RoboCarEnv
from utils.renderer import PygameFrontend
from utils.autoreset import AutoResetWrapper


key = jax.random.PRNGKey(0)
env_params, init_state = RoboCarEnv.init_params(
    key=key,
    map_id=1,
    max_steps=1000,
    discretization_scale=1,
    path_length=100,
    fps=60,
    perception_radius=5.0,
    num_ray_sensors=32,
)

env = AutoResetWrapper(RoboCarEnv, env_params, init_state)

# test parallel envs
# TBD

# test renderer
frontend = PygameFrontend(env, env_params, init_state, eval_mode=False)
frontend.run()
