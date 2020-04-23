import numpy as np
from rogi_rl.agent_state import AgentState
from rogi_rl.env import ActionType
from gym.spaces import Box

from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class ParamActionModel(TFModelV2):
    """Parametric action model that handles the dot product and masking.
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        super(ParamActionModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        action_dims = self.action_space.nvec
        self.action_embed_model = FullyConnectedNetwork(
            Box(np.float32(0), np.float32(1), shape=(action_dims[1] *
                                                     action_dims[2] *
                                                     len(AgentState),)),
            action_space, num_outputs, model_config, name + "_action_mask")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_embed_orig, _ = self.action_embed_model({
            "obs": input_dict["obs"]["rogi"]
        })
        action_dims = self.action_space.nvec
        num_outputs = sum(action_dims)
        width = action_dims[1]

        action_embed = tf.squeeze(action_embed_orig)
        action_mask_actual = np.array([1.] * num_outputs)
        action_mask = tf.squeeze(action_mask)
        n_action_types = len(ActionType)
        end_x = width + n_action_types
        action_type = tf.argmax(action_embed[:n_action_types], axis=-1)
        cell_x = tf.argmax(action_embed[n_action_types:end_x], axis=-1)
        cell_y = tf.argmax(action_embed[end_x:], axis=-1)

        action_mask_value = action_mask[cell_x, cell_y]

        if(int(action_type) == 1):
            # Change vaccination to Step
            cell_x_idx = n_action_types + cell_x
            cell_y_idx = n_action_types + width + cell_y
            if int(action_mask_value) == 0:
                action_mask_actual[1] = 0  # Disable Vaccinate
                action_mask_actual[cell_x_idx] = 0  # Mask Cell X
                action_mask_actual[cell_y_idx] = 0  # Mask Cell Y

        action_logits = action_embed_orig

        # Mask out invalid actions (use tf.float32.min for stability)
        action_mask_actual = tf.constant(action_mask_actual, dtype=tf.float32)
        inf_mask = tf.maximum(tf.log(action_mask_actual),
                              tf.float32.min)
        inf_mask = tf.expand_dims(inf_mask, 0)

        action_logits_updated = action_logits + inf_mask
        return action_logits_updated, state

    def value_function(self):
        return self.action_embed_model.value_function()
