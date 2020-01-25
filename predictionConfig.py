
from mrcnn.config import Config

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "kidney_cfg"
	# number of classes (background + kidney)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1