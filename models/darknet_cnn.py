# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.utils.framework import try_import_tf
# from ray.rllib.models import ModelCatalog

# tf = try_import_tf()


# class DarknetCNN(TFModelV2):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)

#         inputs = tf.keras.layers.Input(
#             shape=obs_space.shape, name="observations")
#         scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

#         x = scaled_inputs
