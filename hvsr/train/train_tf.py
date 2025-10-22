import os
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from ..models.hvsr_tf import HybridRecModel
from ..data.loaders import (
    build_tf_dataset, build_triplet_arrays, compute_cold_items,
    items_to_img_bytes_array, find_image_key, load_dvbpr_npy
)

class LRScheduler:
    def __init__(self, optimizer, patience=3, decay=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.patience = patience
        self.decay = decay
        self.min_lr = min_lr
        self.best = -np.inf
        self.wait = 0
    def step(self, metric):
        if metric > self.best + 1e-5:
            self.best = metric; self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old = float(tf.keras.backend.get_value(self.optimizer.lr))
                new = max(self.min_lr, old * self.decay)
                tf.keras.backend.set_value(self.optimizer.lr, new)
                print(f"[LR] ReduceOnPlateau: {old:.2e} -> {new:.2e}")
                self.wait = 0

@tf.function
def train_step(model, optimizer, u, pos_id, img_p, clip_p, imgs_n, clips_n, num_negs):
    with tf.GradientTape() as tape:
        pos_score, neg_score = model(u, img_p, clip_p, imgs_n, clips_n, training=True)
        neg_score_flat = tf.reshape(neg_score, [-1, num_negs])
        loss_bpr = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(tf.expand_dims(pos_score, 1) - neg_score_flat) + 1e-8))
        loss = loss_bpr + tf.add_n(model.losses) if model.losses else loss_bpr
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def evaluate(model, dataset, num_negs, cold_set=None):
    all_scores, all_labels = [], []
    cold_scores, cold_labels = [], []
    for u, pos_id, img_p, clip_p, imgs_n, clips_n in dataset:
        pos_score, neg_score = model(u, img_p, clip_p, imgs_n, clips_n, training=False)
        pos_score = pos_score.numpy()
        neg_score = neg_score.numpy()
        for i in range(pos_score.shape[0]):
            all_scores.append(float(pos_score[i])); all_labels.append(1)
            all_scores.extend([float(x) for x in neg_score[i]]); all_labels.extend([0]*num_negs)
            if cold_set is not None and int(pos_id[i].numpy()) in cold_set:
                cold_scores.append(float(pos_score[i])); cold_labels.append(1)
                cold_scores.extend([float(x) for x in neg_score[i]]); cold_labels.extend([0]*num_negs)
    auc = roc_auc_score(all_labels, all_scores) if len(all_labels) else 0.0
    cold_auc = roc_auc_score(cold_labels, cold_scores) if cold_set and len(cold_labels) else None
    return auc, cold_auc

def prepare_data(npy_path, clip_path, hardneg_mat=None, img_size=(224,224), num_negs=10, cold_thresh=5, batch_items=512):
    user_train, user_val, user_test, Items, user_count, item_count = load_dvbpr_npy(npy_path)
    clip_emb = np.load(clip_path)
    img_key = find_image_key(Items)
    img_bytes = items_to_img_bytes_array(Items, img_key)
    cold_set, _ = compute_cold_items(user_train, item_count, threshold=cold_thresh)

    train_u, train_p, train_n = build_triplet_arrays(user_train, item_count, num_negs, True, hardneg_mat)
    val_u, val_p, val_n = build_triplet_arrays(user_val, item_count, num_negs, False, None)
    test_u, test_p, test_n = build_triplet_arrays(user_test, item_count, num_negs, False, None)

    def _ds(u, p, n):
        ds = build_tf_dataset(u, p, n, img_bytes, clip_emb, img_size=img_size, num_negs=num_negs, batch_items=batch_items)
        return ds.map(lambda u, p, img_p, c_p, imgs_n, clips_n: (u, p, img_p, c_p, imgs_n, clips_n))
    return {
        "train_ds": _ds(train_u, train_p, train_n),
        "val_ds": _ds(val_u, val_p, val_n),
        "test_ds": _ds(test_u, test_p, test_n),
        "user_count": user_count,
        "item_count": item_count,
        "cold_set": cold_set,
        "clip_dim": int(clip_emb.shape[1]),
    }

def run_training(npy_path, clip_path, outdir,
                 hardneg_mat=None, num_negs=10, cold_thresh=5,
                 img_size=(224,224), batch_items=512, epochs=50, lr=3e-4, seed=42):
    os.makedirs(outdir, exist_ok=True)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    data = prepare_data(npy_path, clip_path, hardneg_mat=hardneg_mat, img_size=img_size,
                        num_negs=num_negs, cold_thresh=cold_thresh, batch_items=batch_items)

    model = HybridRecModel(num_users=data['user_count'], clip_dim=data['clip_dim'])
    optimizer = tf.keras.optimizers.Adam(lr)
    sched = LRScheduler(optimizer, patience=3, decay=0.5, min_lr=1e-6)

    for epoch in range(1, epochs+1):
        losses = []
        for u, pos_id, img_p, clip_p, imgs_n, clips_n in data['train_ds']:
            loss = train_step(model, optimizer, u, pos_id, img_p, clip_p, imgs_n, clips_n, num_negs)
            losses.append(float(loss.numpy()))
            if len(losses) % 50 == 0:
                print(f"[Epoch {epoch:02d}] step={len(losses):04d} loss={np.mean(losses[-50:]):.4f}")
        val_auc, val_cold = evaluate(model, data['val_ds'], num_negs, cold_set=data['cold_set'])
        test_auc, test_cold = evaluate(model, data['test_ds'], num_negs, cold_set=data['cold_set'])
        print(f"Epoch {epoch:02d} | loss={np.mean(losses):.4f} | val_auc={val_auc:.4f} val_cold={val_cold} | test_auc={test_auc:.4f} test_cold={test_cold}")
        sched.step(val_auc)

    model.save_weights(os.path.join(outdir, "hvsr_tf.ckpt"))
    print(f"Saved weights to {os.path.join(outdir, 'hvsr_tf.ckpt')}")
    return model
