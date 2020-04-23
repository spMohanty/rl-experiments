import numpy as np
from ray.rllib.models.preprocessors import Preprocessor


class ObsFlattenPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        return (np.prod(obs_space.shape),)

    def transform(self, observation):
        flattened_obs = observation.flatten()
        return flattened_obs
