from rogi_rl import RogiSimEnv
from gym import spaces  # , wrappers
import numpy as np

from rogi_rl.agent_state import AgentState
import gym
from gym.spaces import Box, MultiDiscrete, Dict  # Discrete

from rogi_rl.env import ActionType


class RogiRlWrapper(RogiSimEnv):

    def __init__(self, env_config):
        super().__init__(env_config)
        self.observation_space = spaces.Box(
            low=np.int8(0),
            high=np.int8(1),
            shape=(
                self.width *
                self.height *
                len(AgentState),))

    def reset(self):
        obs = super().reset()
        return obs.flatten()

    def step(self, action):
        _observation, _step_reward, _done, _info = super().step(action)
        return _observation.flatten(), _step_reward, _done, _info

        # TODO : Add Observation Preprocessor in wrapper or Reward Hacking


class RogiRlParamAction(gym.Env):
    """Parametric action version.

    In this env there is a multi-discrete actions
                with [len(ActionType, width, height]
    Total_Size = len(ActionType) + width + height

    We have 2 action types STEP and VACCINATE supported currently

    logits[:len(ActionType)] corresponds to the logits for step, vaccinate
    logits[len(ActionType):width + len(ActionType)] are logits for X Coordinate
    logits[width + len(ActionType):] are logits for Y Coordinate

    Currently we support masking based on the vaccinate cell if non-susceptible

    At each step, we emit a dict of:
        - the actual observation
        - a mask of valid actions (e.g., [0, 0, 1, ...n]
                                   for `n` max avail actions)
    """

    def __init__(self, env_config={}):

        self.wrapped = RogiRlWrapper(env_config)
        self.observation_space = Dict({
            "action_mask": Box(np.int8(0), np.int8(1),
                               shape=(1, self.wrapped.width,
                                      self.wrapped.height,)),
            "rogi": Box(low=np.float32(0),
                        high=np.float32(1),
                        shape=(
                            self.wrapped.width *
                            self.wrapped.height *
                            len(AgentState),))
        })
        self.action_space = MultiDiscrete(
            [
                len(ActionType), self.wrapped.width, self.wrapped.height
            ])

    def update_avail_actions(self, orig_obs, flattened=True):
        width, height = self.wrapped.width, self.wrapped.height
        self.action_mask = np.zeros((1, width, height))
        if flattened:
            orig_obs = orig_obs.reshape(width, height, len(AgentState))
        for i in range(self.action_mask.shape[0]):
            # TODO: Support for batch - vectorize with Observation
            self.action_mask[i] = orig_obs[:, :, AgentState.SUSCEPTIBLE.value]

    def reset(self):
        orig_obs = self.wrapped.reset()
        self.update_avail_actions(orig_obs, True)
        return {
            "action_mask": self.action_mask,
            "rogi": orig_obs,
        }

    def step(self, action):
        orig_obs, rew, done, info = self.wrapped.step(action)
        self.update_avail_actions(orig_obs)
        obs = {
            "action_mask": self.action_mask,
            "rogi": orig_obs,
        }
        return obs, rew, done, info
