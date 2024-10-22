# from pathlib import Path
# from typing import Any, Dict, Optional, Union

# import jax
# import jax.numpy as jnp
# import mujoco
# from brax.envs.base import PipelineEnv, State
# from brax.io import mjcf
# from flax import struct
# from mujoco import mjx

# from playground import ROOT

# @struct.dataclass
# class SlipConfig:
#     """Config dataclass for Spring Loaded Inverted Pendulum."""

#     # model path: scene.xml contains ground + other niceties in addition to the pendulum
#     model_path: Union[Path, str] = ROOT + "/envs/slip/slip.xml"

#     # reset state params, Uniform(-p, p)
#     pos_reset_noise_scale: float = 0.3
#     vel_reset_noise_scale: float = 0.2

#     # reward coefficients
#     height_reward_weight: float = 1.0
#     forward_reward_weight: float = 1.0
#     ctrl_cost_weight: float = 0.05

# class SlipEnv(PipelineEnv):
#     """Environment for training a Spring Loaded Inverted Pendulum to hop fwd.

#     States: x = (qpos, qvel), shape=(12,)
#     Observations: All states except for horiz pos, shape=(11,)
#     Actions: motor torques of roll, pitch, and prismatic. tau, shape=(3,)
#     """

#     def __init__(self, config: Optional[SlipConfig] = None) -> None:
#         """Initialize the SLIP environment."""
#         if config is None:
#             config = SlipConfig()
#         self.config = config
#         mj_model = mujoco.MjModel.from_xml_path(config.model_path)
#         sys = mjcf.load_model(mj_model)

#         super().__init__(sys, n_frames=5, backend="mjx")

#     # define a reset function
#     def reset(self, rng: jax.random.PRNGKey) -> State:
#         """Resets the environment to an initial state."""

#         # PRNG Key Split, 3 keys (Beware of using the same key)
#         rng, rng1, rng2 = jax.random.split(rng, 3)

#         # randomize the intial state
#         qpos = self.sys.init_q + jax.random.uniform(  # TODO: Check obs size
#             rng1,
#             (self.sys.q_size(),),
#             minval=-self.config.pos_reset_noise_scale,
#             maxval=self.config.pos_reset_noise_scale,
#         )
#         qvel = jax.random.uniform(
#             rng2,
#             (self.sys.q_size(),),
#             minval=-self.config.vel_reset_noise_scale,
#             maxval=self.config.vel_reset_noise_scale,
#         )
#         data = self.pipeline_init(qpos, qvel)

#         # get the observation
#         obs = self._compute_obs(data, {})
#         reward, done, zero = jnp.zeros(3)  # TODO: check the metrics
#         metrics = {
#             "x_position": zero,
#             "x_velocity": zero,
#             "reward_ctrl": zero,
#             "reward_run": zero,
#         }

#         state_info = {"rng": rng, "step": 0}
#         return State(data, obs, reward, done, metrics, state_info)

#     # define a step function
#     def step(self, state: State, action: jax.Array) -> State:
#         """Take a step in the environment."""

#         # Simulate physics
#         data0 = state.pipeline_state
#         data = self.pipeline_step(data0, action)

#         # Compute fwd velocity rewards
#         x_vel = (data.x.pos[0] - data0.x.pos[0]) / self.dt
#         forward_reward = self.config.forward_reward_weight * x_vel

#         # Compute the height reward
#         height_reward = self.config.height_reward_weight * data.x.pos[1]

#         # Compute control cost
#         ctrl_cost = self.config.ctrl_cost_weight * jnp.sum(jnp.square(action))
#         reward = forward_reward - ctrl_cost

#     # compute the observation from the mj data
#     def _compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
#         """Compute the observation from the current state."""

#         # skip the horizontal element    # TODO: Check obs size
#         position = data.qpos[1:]
#         velocity = data.qvel

#         return jnp.concatenate([position, velocity])
