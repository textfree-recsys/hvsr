import tensorflow as tf

class DVBPRCNN(tf.keras.layers.Layer):
    def __init__(self, feat_dim=512, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = int(feat_dim)
        self.dropout_rate = float(dropout)
        self.W = {
            'wc1': self.add_weight('wc1', shape=[11,11,3,64], initializer='glorot_uniform'),
            'wc2': self.add_weight('wc2', shape=[5,5,64,256], initializer='glorot_uniform'),
            'wc3': self.add_weight('wc3', shape=[3,3,256,256], initializer='glorot_uniform'),
            'wc4': self.add_weight('wc4', shape=[3,3,256,256], initializer='glorot_uniform'),
            'wc5': self.add_weight('wc5', shape=[3,3,256,256], initializer='glorot_uniform'),
            'wd1': self.add_weight('wd1', shape=[7*7*256,4096], initializer='glorot_uniform'),
            'wd2': self.add_weight('wd2', shape=[4096,4096], initializer='glorot_uniform'),
            'wd3': self.add_weight('wd3', shape=[4096,self.feat_dim], initializer='glorot_uniform'),
        }
        self.b = {
            'wc1': self.add_weight('bc1', shape=[64], initializer='zeros'),
            'wc2': self.add_weight('bc2', shape=[256], initializer='zeros'),
            'wc3': self.add_weight('bc3', shape=[256], initializer='zeros'),
            'wc4': self.add_weight('bc4', shape=[256], initializer='zeros'),
            'wc5': self.add_weight('bc5', shape=[256], initializer='zeros'),
            'wd1': self.add_weight('bd1', shape=[4096], initializer='zeros'),
            'wd2': self.add_weight('bd2', shape=[4096], initializer='zeros'),
            'wd3': self.add_weight('bd3', shape=[self.feat_dim], initializer='zeros'),
        }

    def conv(self, x, W, b, stride=1):
        x = tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def pool(self, x, k=2):
        return tf.nn.max_pool2d(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

    def call(self, x, training=False):
        c1 = self.conv(x, self.W['wc1'], self.b['wc1'], stride=4)
        p1 = self.pool(c1, 2)
        c2 = self.conv(p1, self.W['wc2'], self.b['wc2'])
        p2 = self.pool(c2, 2)
        c3 = self.conv(p2, self.W['wc3'], self.b['wc3'])
        c4 = self.conv(c3, self.W['wc4'], self.b['wc4'])
        c5 = self.conv(c4, self.W['wc5'], self.b['wc5'])
        p5 = self.pool(c5, 2)
        flat = tf.reshape(p5, [-1, 7*7*256])
        f1 = tf.nn.relu(tf.matmul(flat, self.W['wd1']) + self.b['wd1'])
        f1 = tf.nn.dropout(f1, rate=self.dropout_rate if training else 0.0)
        f2 = tf.nn.relu(tf.matmul(f1, self.W['wd2']) + self.b['wd2'])
        f2 = tf.nn.dropout(f2, rate=self.dropout_rate if training else 0.0)
        out = tf.matmul(f2, self.W['wd3']) + self.b['wd3']
        return out
