# Copyright (c) 2025 HVSR authors
# TensorFlow residual fusion block matching:
# out = LayerNormalization()(x + residual)
#
# Architecture:
# Input: concat([CLIP(768), CNN(512)]) -> 1280-D
# MLP: 1024 (ReLU) -> 512 (ReLU) -> 256 (ReLU) -> 512 (linear)
# Residual: Dense(512, use_bias=False) on input
# Output: LayerNorm(x + residual)

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_fusion(clip_dim: int, cnn_feat_dim: int) -> Model:
    """
    Keras Model version of the residual fusion block.

    Args:
        clip_dim: Dimension of CLIP embedding (e.g., 768).
        cnn_feat_dim: Dimension of CNN feature (e.g., 512). Also the fused output dim.

    Returns:
        tf.keras.Model named "Fusion" that takes a (clip_dim + cnn_feat_dim,) vector
        and outputs a (cnn_feat_dim,) fused representation.
    """
    inp = layers.Input(shape=(clip_dim + cnn_feat_dim,), name="fusion_input")
    x = layers.Dense(1024, activation='relu', name='fuse_dense_1024')(inp)
    x = layers.Dense(512, activation='relu', name='fuse_dense_512')(x)
    x = layers.Dense(256, activation='relu', name='fuse_dense_256')(x)
    x = layers.Dense(cnn_feat_dim, activation=None, name='fuse_proj')(x)
    residual = layers.Dense(cnn_feat_dim, activation=None, use_bias=False, name='fuse_residual')(inp)
    out = layers.LayerNormalization(name='fuse_layernorm')(x + residual)
    return Model(inputs=inp, outputs=out, name='Fusion')


class Fusion(layers.Layer):
    """
    Layer version of the residual fusion block. Useful if you want to instantiate
    the block once and call it repeatedly without building a full Model.
    """
    def __init__(self, clip_dim: int, cnn_feat_dim: int, name: str = "FusionLayer"):
        super().__init__(name=name)
        self.clip_dim = clip_dim
        self.cnn_feat_dim = cnn_feat_dim

        self.d1 = layers.Dense(1024, activation='relu', name=f'{name}_dense_1024')
        self.d2 = layers.Dense(512, activation='relu', name=f'{name}_dense_512')
        self.d3 = layers.Dense(256, activation='relu', name=f'{name}_dense_256')
        self.proj = layers.Dense(cnn_feat_dim, activation=None, name=f'{name}_proj')
        self.res = layers.Dense(cnn_feat_dim, activation=None, use_bias=False, name=f'{name}_residual')
        self.norm = layers.LayerNormalization(name=f'{name}_layernorm')

    def call(self, fused_input, training=False):
        """
        Args:
            fused_input: Tensor of shape (batch, clip_dim + cnn_feat_dim).
        """
        x = self.d1(fused_input, training=training)
        x = self.d2(x, training=training)
        x = self.d3(x, training=training)
        x = self.proj(x, training=training)
        r = self.res(fused_input, training=training)
        return self.norm(x + r)
