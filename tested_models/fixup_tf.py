from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf = try_import_tf()


class FixupTF(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")

        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0
        x = scaled_inputs

        for depth in [32, 64, 64]:
            x = tf.keras.layers.Conv2D(filters=depth, kernel_size=3)(x)
            x = tf.keras.layers.MaxPool2D(
                pool_size=3, strides=2, padding="same")(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            x = ResidualBlock(depth)(x)
            x = ResidualBlock(depth)(x)

        x = ResidualBlock(64)(x)
        x = ResidualBlock(64)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units=256)(x)

        logits = tf.keras.layers.Dense(units=num_outputs)(x)
        value = tf.keras.layers.Dense(units=1)(x)

        self.base_model = tf.keras.Model(inputs, [logits, value])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=3, padding="same", kernel_initializer=self.scaled_init)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.Zeros())
        self.bias1 = tf.Variable(initial_value=0.)
        self.bias2 = tf.Variable(initial_value=0.)
        self.bias3 = tf.Variable(initial_value=0.)
        self.bias4 = tf.Variable(initial_value=0.)
        self.scale = tf.Variable(initial_value=1.)

    def scaled_init(self, shape, dtype):
        init = tf.keras.initializers.glorot_uniform()
        return 0.3535 * init(shape, dtype=dtype)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        inputs = tf.keras.layers.ReLU()(inputs)
        out = inputs + self.bias1
        out = self.conv1(out)
        out = out + self.bias2

        out = tf.keras.layers.ReLU()(out)
        out = out + self.bias3
        out = self.conv2(out)

        out = out * self.scale
        out = out + self.bias4

        return inputs + out


# Register model in ModelCatalog
ModelCatalog.register_custom_model("fixup_tf", FixupTF)
