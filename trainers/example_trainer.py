from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import logging


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, summary_logger, tester):
        super(ExampleTrainer, self).__init__(sess, model, data, config, summary_logger, tester)
        self.logger = logging.getLogger(__name__)

    def train_epoch(self):
        losses = []
        accs = []
        self.sess.run(self.data.train_init, feed_dict={ self.data.batch_size: self.config.batch_size})
        self.logger.info('Running epoch {}'.format(self.model.cur_epoch_tensor.eval(self.sess)))
        loop = tqdm(range(self.config.num_iter_per_epoch))
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.info('Loss: {:.3}, Accuracy: {:.3}'.format(loss, acc))
        self.summary_logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        features, labels = self.sess.run(self.data.next_batch)
        feed_dict = {self.model.is_training: True, self.model.x: features, self.model.y: labels}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
