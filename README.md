# Tensorflow Project Template

This template is an extension of [MrGemy95's tensorflow template](https://github.com/MrGemy95/Tensorflow-Project-Template). You should probably start there and read their README!

It adds the following features:
* Support for multiple experiments (based on a single `config.json`)
* Extended data generator class using the `tf.data.dataset` pipeline
* Test class
* Logging
* Simple example using the Iris dataset


# Installation
* Clone repo
```
git clone git@github.com:mar-muel/tensorflow-project-template.git
cd tensorflow-project-template
```
* Install dependencies with [Anaconda](https://www.anaconda.com/download):
```
conda create -n tf-project-template python=3.6
source activate tf-project-template
pip install -r requirements.txt
```

# Usage
After install, you should be able to run a test network on the Iris dataset by simply
```
python run.py
```
You can specify a different config file from the default (`./configs/config.json`):
```
python run.py --config ./path/to/my/config.json
```
The default config file looks like this:
```
{
  "experiments": {
    "run1": {
      "learning_rate": 1e-4
    },
    "run2": {
      "learning_rate": 1e-5
    }
  },
  "global_params": {
    "training_data": "train.csv",
    "test_data": "test.csv",
    "learning_rate": 1e-4,
    "max_to_keep": 1,
    "num_epochs": 4,
    "num_iter_per_epoch": 1000,
    "batch_size": 10,
    "run_test": true,
    "delete_previous_output": true
  }
}
```
The config contains two experiments with the names `run1` and `run2` containing different learning rates. All parameters under `global_params` will be applied to each individual experiment.
## Data
By default all data resides under `./data` and will be read from there. Specify how you want to load your data in `./data_loader/data_generator.py`. 

## Model
Create your own network architecture by changing the code in the `build_model()` function in `./models/example_model.py`.

## Training
You can change the way the training should happen in `./trainers/example_trainer.py`.

## Testing
You can change the way the training should happen in `./trainers/example_test.py`.

## Tensorboard
You can see all collected summaries in tensorboard by running
```
tensorboard --logdir ./experiments
```

Feel free re-use, extend or adapt for your own purposes!

# Author
Martin Müller (martin.muller@epfl.ch)
