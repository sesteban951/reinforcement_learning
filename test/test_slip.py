import jax
import mujoco

from playground import ROOT
from playground.envs.slip.slip_env import SlipConfig, SlipEnv


def test_slip_mujoco_model():
    """Test that the slip mujoco model."""
    print("Testing SLIP Mujoco Model")
    model_file = ROOT + "/envs/slip/slip.xml"

    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)

    assert isinstance(model, mujoco.MjModel)
    assert isinstance(data, mujoco.MjData)
    assert model.nq == 6
    assert model.nv == 6
    assert model.nu == 3
    assert data.qpos.shape == (6,)
    assert data.qvel.shape == (6,)
    assert data.ctrl.shape == (3,)


def test_slip_config():
    """Test that the slip config can be created."""
    print("Creating SlipConfig")
    config = SlipConfig()

    for key in config.__annotations__:
        assert hasattr(config, key)


def test_slip_env():
    """Test that the slip environment can be created."""
    print("Creating SlipEnv")
    env = SlipEnv()

    print("Resetting SlipEnv")
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    assert state.obs.shape == (12,)
    assert state.reward == 0.0
    assert state.done == 0.0
    assert state.info["step"] == 0

    print("Taking a step")
    ctrl = jax.numpy.zeros(3)
    state = env.step(state, ctrl)

    assert state.reward != 0.0
    assert state.done == 0.0
    assert state.info["step"] == 1


if __name__ == "__main__":
    print("Running SLIP tests")
    # test_slip_mujoco_model()
    # test_slip_config()
    test_slip_env()
