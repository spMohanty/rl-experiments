from rogi_rl import RogiSimEnv


class RogiRlSmall(RogiSimEnv):

    def __init__(self, env_config):
        super().__init__(env_config)

        # TODO : Add Observation Preprocessor in wrapper or Reward Hacking
