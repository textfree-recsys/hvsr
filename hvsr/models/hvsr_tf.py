# Copyright (c) 2025 HVSR authors
# Hybrid model wrapper (TensorFlow): DVBPR-CNN + Residual Fusion + user embeddings

import tensorflow as tf
from tensorflow.keras import layers
from .dvbpr_cnn_tf import DVBPRCNN
from .fusion_tf import Fusion

class HybridRecModel(tf.keras.Model):
    """
    TensorFlow implementation of HVSR inference core:
      - Image -> DVBPR CNN -> 512-D
      - Concatenate with frozen CLIP (768-D)
      - Residual fusion -> 512-D item vector
      - User embedding -> 512-D
      - Score = dot(user, item)
    """
    def __init__(self, num_users: int, clip_dim: int = 768, emb_dim: int = 512, dropout: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.clip_dim = int(clip_dim)
        self.emb_dim = int(emb_dim)
        self.cnn = DVBPRCNN(feat_dim=self.emb_dim, dropout=dropout)
        self.fuse = Fusion(clip_dim=self.clip_dim, cnn_feat_dim=self.emb_dim, name="Fusion")
        self.user_emb = layers.Embedding(
            num_users, self.emb_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )

    def score(self, user_vec: tf.Tensor, item_vec: tf.Tensor) -> tf.Tensor:
        """Dot-product relevance score."""
        return tf.reduce_sum(user_vec * item_vec, axis=-1)

    def encode_item(self, img: tf.Tensor, clip_vec: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Encode a batch of items:
          img: (B, H, W, 3)
          clip_vec: (B, clip_dim)
        Returns:
          fused: (B, emb_dim)
        """
        cnn_feat = self.cnn(img, training=training)                             # (B, emb_dim)
        fused_in = tf.concat([clip_vec, cnn_feat], axis=-1)                     # (B, clip_dim + emb_dim)
        fused = self.fuse(fused_in, training=training)                          # (B, emb_dim)
        return fused

    def call(self, u, img_p, clip_p, imgs_n, clips_n, training: bool = False):
        """
        Forward for BPR-K training.
          u: (B,)
          img_p: (B, H, W, 3)
          clip_p: (B, clip_dim)
          imgs_n: (B, K, H, W, 3)
          clips_n: (B, K, clip_dim)
        Returns:
          pos_score: (B,)
          neg_score: (B, K)
        """
        # Positive
        fuse_p = self.encode_item(img_p, clip_p, training=training)             # (B, emb_dim)

        # Negatives
        b = tf.shape(imgs_n)[0]
        k = tf.shape(imgs_n)[1]
        imgs_n_flat = tf.reshape(imgs_n, [b * k, tf.shape(imgs_n)[2], tf.shape(imgs_n)[3], tf.shape(imgs_n)[4]])
        clips_n_flat = tf.reshape(clips_n, [b * k, tf.shape(clips_n)[2]])
        fuse_n_flat = self.encode_item(imgs_n_flat, clips_n_flat, training=training)  # (B*K, emb_dim)
        fuse_n = tf.reshape(fuse_n_flat, [b, k, self.emb_dim])                  # (B, K, emb_dim)

        # User vector
        u_vec = self.user_emb(u)                                                # (B, emb_dim)

        # Scores
        pos_score = self.score(u_vec, fuse_p)                                   # (B,)
        neg_score = tf.reduce_sum(tf.expand_dims(u_vec, 1) * fuse_n, axis=-1)   # (B, K)

        return pos_score, neg_score
