import numpy as np
import os
import tensorflow as tf


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.batch_size = tf.placeholder(tf.int64)
        self.data_dir = 'data'
        # data specific
        self.num_features = 4
        self.num_classes = 3
        # define tf.datasets
        self.training_dataset, self.test_dataset = self.create_datasets()
        # create iterator of datasets
        self.iterator = self.create_iterator()
        self.next_batch = self.iterator.get_next()
        # create initializer operations for both training and test data
        self.train_init, self.test_init = self.create_initializers()

    def create_datasets(self):
        # load training data
        training_dataset = tf.data.TextLineDataset(os.path.join(self.data_dir, self.config.training_data))
        training_dataset = training_dataset.skip(1).shuffle(buffer_size=1000).repeat()
        training_dataset = training_dataset.map(self.parse_csv).batch(self.batch_size)

        # load test data (make sure not to use .repeat())
        test_dataset = tf.data.TextLineDataset(os.path.join(self.data_dir, self.config.test_data))
        test_dataset = test_dataset.skip(1)
        test_dataset = test_dataset.map(self.parse_csv).batch(self.batch_size)
        return training_dataset, test_dataset

    def parse_csv(self, records):
        """Takes the string input tensor and returns tuple of (features, labels)."""
        columns = tf.decode_csv(records, record_defaults=[[]] * (self.num_features + 1))
        features = columns[:self.num_features]
        labels = tf.one_hot(tf.cast(columns[self.num_features], tf.int64), self.num_classes)
        return features, labels

    def create_iterator(self):
        return tf.data.Iterator.from_structure(self.training_dataset.output_types, 
                self.training_dataset.output_shapes)

    def create_initializers(self):
        training_init = self.iterator.make_initializer(self.training_dataset)
        test_init = self.iterator.make_initializer(self.test_dataset)
        return training_init, test_init
