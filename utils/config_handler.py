import json
from munch import Munch
import os
import argparse
import logging
import shutil
import json

class ConfigHandler:
    def __init__(self):
        self.args = None
        self.config = {}
        self.config_path = os.path.abspath(os.path.join('.', 'configs'))
        self.experiment_names = []
        self.logger = logging.getLogger(__name__)

    def parse_args(self):
        """
        Parse command line arguments
        """
        argparser = argparse.ArgumentParser(description=__doc__)
        argparser.add_argument(
            '-c', '--config',
            metavar='C',
            default='config.json',
            help='Name of configuration file in ./config folder. Default: config.json')
        self.args = argparser.parse_args()

    def process_config(self):
        """
        collect configuration options from config file
        :param json_file:
        :return: config(namespace) or config(dictionary)
        """
        config = self._parse_config_file(self.args.config)
        # collect global settings
        global_params = {}
        if 'global_params' in config:
            global_params = config['global_params']

        self.experiment_names = list(config['experiments'].keys())
        if len(self.experiment_names) != len(set(self.experiment_names)):
            raise Exception('Defined experiment names (keys under experiments) are not unique.')
        for experiment_name in self.experiment_names:
            # merge config with global options (experiment options will overwrite global options)
            experiment_config = {**global_params, **config['experiments'][experiment_name]}
            # create Munch object for easier key/value handling
            self.config[experiment_name] = Munch(experiment_config)
            # add additional configs
            self.config[experiment_name].summary_dir = os.path.join('.', 'experiments', experiment_name, 'summary', '')
            self.config[experiment_name].checkpoint_dir = os.path.join('.', 'experiments', experiment_name, 'checkpoint', '')
            self.config[experiment_name].config_file_dump = os.path.join('.', 'experiments', experiment_name, experiment_name + '_params.json')

    def _parse_config_file(self, config_file):
        """
        :param config_file: Name of configuration file
        :return: Content of configuration file
        """
        path_to_config = os.path.join(self.config_path, config_file)
        with open(path_to_config, 'r') as cf:
            config = json.load(cf)
        return config

    def _write_config_dump(self, config, f_path):
        with open(f_path, 'w') as f:
            json.dump(config, f)

    def create_config_dirs(self):
        """
        Creates folders for experiments, deletes old directories if "delete_previous_output"
        """
        try:
            for n in self.experiment_names:
                dirs = [self.config[n].summary_dir, self.config[n].checkpoint_dir]
                for dir_ in dirs:
                    if not os.path.exists(dir_):
                        os.makedirs(dir_)
                    else:
                        if self.config[n].delete_previous_output:
                            shutil.rmtree(dir_)
                            os.makedirs(dir_)
                    self._write_config_dump(self.config[n], self.config[n].config_file_dump)
            return 0
        except Exception as err:
            self.logger.exception("Creating directories error: {0}".format(err))
            exit(-1)
