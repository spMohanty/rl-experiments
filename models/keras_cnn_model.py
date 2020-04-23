from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import get_activation_fn
from ray.rllib.utils.framework import try_import_tf

tf = try_import_tf()


class KerasCnnModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name):
        super(KerasCnnModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=3, strides=2,
                                       activation=get_activation_fn(
                                           model_config.get("conv_activation")
                                       ))(self.inputs)
        conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2,
                                       activation=get_activation_fn(
                                           model_config.get("conv_activation")
                                       ))(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                       activation=get_activation_fn(
                                           model_config.get("conv_activation")
                                       ))(conv2)
        conv_flatten = tf.keras.layers.Flatten()(conv3)
        state = tf.keras.layers.Dense(model_config['custom_options']
                                      ['hidden_units'],
                                      activation=get_activation_fn(
            model_config.get("fcnet_activation")))(conv_flatten)
        layer_out = tf.keras.layers.Dense(
            num_outputs, name="act_output")(state)
        value_out = tf.keras.layers.Dense(1, name="value_output")(state)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
