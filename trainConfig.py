from mrcnn.config import Config

# define a configuration for the model
class TrainConfig(Config):
    # define the name of the configuration
    NAME = "kidney_cfg"
    # number of classes (background + kidney)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 131