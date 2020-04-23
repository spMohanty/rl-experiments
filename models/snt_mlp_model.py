import sonnet as snt
import gym
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


class SntMlpNet(snt.Module):

    def __init__(self, obs_space, action_space, model_config, name=None):
        super().__init__(name)
        assert isinstance(action_space, gym.spaces.Discrete)
        self.obs_space = obs_space
        self.action_space = action_space
        self.model_config = model_config

    def _build(self, obs, prev_actions, *unused_args, **unused_kwargs):
        mlp = snt.nets.MLP(output_sizes=self.model_config['fcnet_hiddens'],
                           activation=self.model_config['fcnet_activation'],
                           activate_final=True)
        latent_vec = mlp(obs)

        logits = snt.Linear(output_size=self.action_space.n)(latent_vec)
        baseline = snt.Linear(output_size=1)(latent_vec)
        return logits, baseline


class SntMlpModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name):
        super().__init__(obs_space, action_space, num_outputs,
                         model_config, name)
        self.model = SntMlpNet(obs_space, action_space, model_config)

    def forward(self, input_dict, state, seq_lens):
        obs, prev_actions = input_dict['obs'], input_dict["prev_actions"]

        logits, baseline = self.model(obs, prev_actions)

        self.baseline = tf.reshape(baseline, [-1])

        return logits, state

    def variables(self):
        if not self.model.is_connected:
            return []
        return self.model.variables

    def value_function(self):
        return self.baseline
