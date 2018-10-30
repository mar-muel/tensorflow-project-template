from base.base_test import BaseTest
import tensorflow as tf
import numpy as np
import logging

class ExampleTest(BaseTest):
    def __init__(self, sess, model, data, config, summary_logger):
        super(ExampleTest, self).__init__(sess, model, data, config, summary_logger)
        self.logger = logging.getLogger(__name__)

    def test(self):
        # switch to test data and set batch size to something large
        self.sess.run(self.data.test_init, feed_dict={self.data.batch_size: 10**3})

        with tf.name_scope('test_accuracy'):
            tf_labels = tf.placeholder(tf.int64, [None, self.data.num_classes])
            tf_predictions = tf.placeholder(tf.float64, [None, self.data.num_classes])
            acc, acc_op = tf.metrics.accuracy(tf.argmax(tf_labels, axis=1),
                    tf.argmax(tf_predictions, axis=1))

        # init local variables associated with test accuracy
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="test_accuracy")
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        self.sess.run(running_vars_initializer)

        while True:
            try:
                features, labels = self.sess.run(self.data.next_batch)
                predictions = self.sess.run([self.model.predictions], feed_dict={self.model.x: features, self.model.y: labels})
                # run update accuracy operation
                self.sess.run(acc_op, feed_dict={tf_labels: labels, tf_predictions: predictions[0]})
            except tf.errors.OutOfRangeError:
                # looped through all test data
                break

        # get final accuracy by calling accuracy operation
        accuracy = self.sess.run(acc)
        self.logger.info('Accuracy on test set: {:.3}'.format(accuracy))

        # add to summary
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.summary_logger.summarize(cur_it, summarizer='test', summaries_dict={'accuracy': accuracy})
