from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from flax import struct
from mujoco import mjx

from playground import ROOT


@struct.dataclass
class SlipConfig:
    """Config dataclass for Spring Loaded Inverted Pendulum."""

    # model path
    model_path: Union[Path, str] = ROOT + "/envs/slip/slip.xml"

    # reset state params, Uniform(-p, p)
    pos_reset_noise_scale: float = 0.3
    vel_reset_noise_scale: float = 0.2

    # desired hopping apex height
    desired_height: float = 0.8  # in the z-direction

    # reward coefficients
    height_cost_weight: float = 1.0  # hopping at certain height reward
    angled_leg_cost_weight: float = 0.1  # straight leg reward
    vel_cost_weight: float = 0.1  # x-y velocity cost
    ctrl_cost_weight: float = 0.05  # effort applied cost


class SlipEnv(PipelineEnv):
    """Environment for training a Spring Loaded Inverted Pendulum to hop.

    States: x = (qpos, qvel), shape=(12,) [px,py,pz,leg_roll,leg_pitch,leg_pos]
    Observations: All states, shape=(12,)
    Actions: motor torques of roll, pitch, and prismatic. tau, shape=(3,)
    """

    # init function
    def __init__(self, config: Optional[SlipConfig] = None) -> None:
        """Initialize the SLIP environment."""
        if config is None:
            config = SlipConfig()
        self.config = config
        mj_model = mujoco.MjModel.from_xml_path(config.model_path)
        sys = mjcf.load_model(mj_model)

        super().__init__(
            sys, n_frames=5, backend="mjx"
        )  # n_frames: number of sim steps per control step, dt = n_frames * xml_dt

    # define a reset function
    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Resets the environment to an initial state."""
        # PRNG Key Split, 3 keys (Beware of using the same key)
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # randomize the intial state
        qpos = self.sys.init_q + jax.random.uniform(  # TODO: Check obs size
            rng1,
            (self.sys.q_size(),),
            minval=-self.config.pos_reset_noise_scale,
            maxval=self.config.pos_reset_noise_scale,
        )
        qvel = jax.random.uniform(
            rng2,
            (self.sys.qd_size(),),
            minval=-self.config.vel_reset_noise_scale,
            maxval=self.config.vel_reset_noise_scale,
        )

        # initialize the pipeline
        data = self.pipeline_init(qpos, qvel)

        # get the new observation
        obs = self._compute_obs(data, {})

        # fill in other state info
        reward, done, zero = jnp.zeros(3)  # for Tensorboard
        metrics = {
            "reward_height": zero,
            "reward_angled_leg": zero,
            "reward_vel": zero,
            "reward_ctrl": zero,
        }
        state_info = {"rng": rng, "step": 0}

        return State(data, obs, reward, done, metrics, state_info)

    # define a step function
    def step(self, state: State, action: jax.Array) -> State:
        """Take a step in the environment."""
        # Simulate physics
        data0 = state.pipeline_state
        # TODO: scale the action, by default it is in (-1, 1). either divide or mulitply by something, idk.
        # print(type(data0))
        data = self.pipeline_step(data0, action)
        # print(type(data))

        # TODO: play with the reward function
        # Compute fwd velocity cost
        vx_vel = (
            data.qpos[0] - data0.qpos[0]
        ) / self.dt  # TODO: check state access correctly. The 'x' might be some JAX thing
        vy_vel = (data.qpos[1] - data0.qpos[1]) / self.dt
        vel_cost = self.config.vel_cost_weight * (
            jnp.square(vx_vel) + jnp.square(vy_vel)
        )

        # Compute the height reward
        height_cost = self.config.height_cost_weight * jnp.square(
            data.qpos[2]
            - self.config.desired_height  # TODO: check the state access
        )

        # Compute the straight leg reward
        angled_leg_cost = self.config.angled_leg_cost_weight * (
            jnp.square(data.qpos[3])
            + jnp.square(data.qpos[4])  # TODO: check the state access
        )

        # Compute the control cost
        ctrl_cost = self.config.ctrl_cost_weight * jnp.sum(
            jnp.square(action)
        )  # TODO: is redundant since we do pos control

        # compute the total reward
        reward = -(height_cost + angled_leg_cost + vel_cost + ctrl_cost)
        # reward = -ctrl_cost
        print(reward)

        # update the state
        obs = self._compute_obs(data, state.info)
        state.metrics.update(  # for Tensorboard
            reward_height=-height_cost,
            reward_vel=-vel_cost,
            reward_angled_leg=-angled_leg_cost,
            reward_ctrl=-ctrl_cost,  # TODO: first pass just see if the pipeline works with just the control cost
        )
        state.info["step"] += 1

        return state.replace(pipeline_state=data, obs=obs, reward=reward)

    # compute the observation from the mj data
    def _compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Compute the observation from the current state."""
        # skip the x-y position elements
        position = data.qpos
        velocity = data.qvel
        return jnp.concatenate([position, velocity])

    @property
    def observation_size(self) -> int:
        """Size of the observation space."""
        return 12

    @property
    def action_size(self) -> int:
        """Size of the action space."""
        return 3
