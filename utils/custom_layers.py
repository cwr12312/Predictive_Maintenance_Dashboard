"""
utils/custom_layers.py
======================
Custom layer implementations to handle deserialization of models
saved with newer TensorFlow versions.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import numpy as np


class DenseV2(layers.Dense):
    """
    Custom Dense layer that ignores quantization_config parameter
    for backward compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        # Remove quantization_config if present
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Remove quantization_config from config
        config.pop('quantization_config', None)
        return super().from_config(config)


class Conv2DV2(layers.Conv2D):
    """
    Custom Conv2D layer that ignores quantization_config parameter
    for backward compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)


# Custom objects mapping for deserialization
CUSTOM_LAYERS = {
    'Dense': DenseV2,
    'Conv2D': Conv2DV2,
    'DTypePolicy': tf.keras.mixed_precision.Policy,
    'GlorotUniform': tf.keras.initializers.GlorotUniform,
    'Zeros': tf.keras.initializers.Zeros,
}


# Register custom objects with Keras
for name, layer in CUSTOM_LAYERS.items():
    tf.keras.utils.get_custom_objects()[name] = layer