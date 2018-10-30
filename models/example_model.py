from base.base_model import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float64, shape=[None, 4])
        self.y = tf.placeholder(tf.int64, shape=[None, 3])

        # network architecture
        d1 = tf.layers.dense(self.x, 10, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, 3, name="dense2")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy, global_step=self.global_step_tensor)
            self.correct_prediction = tf.equal(tf.argmax(d2), tf.argmax(self.y))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # this is only used for calculating test accuracies
        self.predictions = tf.nn.softmax(d2)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

