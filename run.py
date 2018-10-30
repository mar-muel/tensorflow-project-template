import os
import logging
import logging.config
import tensorflow as tf
from utils.config_handler import ConfigHandler
from utils.summary_logger import SummaryLogger
from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from trainers.example_test import ExampleTest

def main():
    # set up logging
    configure_logging()
    logger = logging.getLogger(__name__)
    # Capture the config path from the run arguments then process the json configuration file
    config_handler = ConfigHandler()
    try:
        config_handler.parse_args()
    except:
        logger.exception("Missing or invalid arguments")
        exit(0)
    # Read config file(s)
    config_handler.process_config()
    # Create the experiments dirs
    config_handler.create_config_dirs()
    # Run experiments
    for exp_name in config_handler.experiment_names:
        config = config_handler.config[exp_name]
        logger.info('Start running experiment {}'.format(exp_name))
        tf.reset_default_graph()
        sess = tf.Session()
        try:
            # Create your data generator
            data = DataGenerator(config)
            # Create an instance of the model you want
            model = ExampleModel(config)
            # Create tensorboard logger
            summary_logger = SummaryLogger(sess, config)
            # Create test instance
            tester = ExampleTest(sess, model, data, config, summary_logger)
            # Create trainer and pass all the previous components to it
            trainer = ExampleTrainer(sess, model, data, config, summary_logger, tester)
            # Load model
            model.load(sess)
            # Start training
            trainer.train()
        except:
            logger.exception('An exception occured during training')
            sess.close()
        finally:
            sess.close()
    logger.info('Finished all experiments')

def configure_logging():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils', 'config', 'logging.conf')
    logging.config.fileConfig(fname=config_path, disable_existing_loggers=False)


if __name__ == '__main__':
    main()
