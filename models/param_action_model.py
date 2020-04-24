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
        float_tf_type = tf.float32
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_embed_orig, _ = self.action_embed_model({
            "obs": input_dict["obs"]["rogi"]
        })
        action_embed_orig = tf.cast(action_embed_orig, float_tf_type)
        action_mask = tf.cast(action_mask, float_tf_type)

        action_dims = self.action_space.nvec
        num_outputs = sum(action_dims)
        width = action_dims[1]

        action_embed = tf.squeeze(action_embed_orig)
        action_mask_actual = tf.constant([1.]*num_outputs,
                                         dtype=float_tf_type)

        action_mask = tf.squeeze(action_mask)
        n_action_types = len(ActionType)
        end_x = n_action_types + width
        action_type = tf.argmax(action_embed[:n_action_types], axis=-1)
        cell_x = tf.argmax(action_embed[n_action_types:end_x], axis=-1)
        cell_y = tf.argmax(action_embed[end_x:], axis=-1)

        action_mask_value = action_mask[cell_x, cell_y]

        cell_x_idx = tf.add(tf.constant([n_action_types],
                            dtype=cell_x.dtype), cell_x)
        cell_y_idx = tf.add(tf.constant([end_x], dtype=cell_y.dtype), cell_y)
        vaccinate_idx = tf.constant([ActionType.VACCINATE.value],
                                    dtype=cell_x.dtype)

        vaccinate_idx = tf.cast(vaccinate_idx, tf.int32)
        cell_x_idx = tf.cast(cell_x_idx, tf.int32)
        cell_y_idx = tf.cast(cell_y_idx, tf.int32)
        indices = tf.stack(values=[vaccinate_idx,
                                   cell_x_idx, cell_y_idx], axis=0)
        updates = tf.constant([1, 1, 1])
        shape = tf.constant([num_outputs])
        masker = tf.scatter_nd(indices, updates, shape)

        masker = tf.cast(masker, float_tf_type)

        action_mask_vaccinate = tf.cond(tf.equal(action_mask_value, 1),
                                        lambda: action_mask_actual,
                                        lambda: action_mask_actual - masker)

        action_mask_actual = tf.cond(tf.equal(action_type,
                                     ActionType.VACCINATE.value),
                                     lambda: action_mask_vaccinate,
                                     lambda: action_mask_actual)

        action_logits = action_embed_orig

        # Mask out invalid actions (use tf.float32.min for stability
        inf_mask = tf.maximum(tf.log(action_mask_actual),
                              float_tf_type.min)

        inf_mask = tf.expand_dims(inf_mask, 0)

        action_logits_updated = action_logits + inf_mask
        return action_logits_updated, state

    def value_function(self):
        return self.action_embed_model.value_function()
