import tensorflow as tf

class BaseTrain:
    def __init__(self, sess, model, data, config, summary_logger, tester):
        self.model = model
        self.summary_logger = summary_logger
        self.config = config
        self.sess = sess
        self.data = data
        self.tester = tester
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

        # run on test dataset
        if self.config.run_test:
            self.tester.test()

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
