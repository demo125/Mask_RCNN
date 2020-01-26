from kidneyDataset import KidneyDataset
from matplotlib import pyplot as plt
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from trainConfig import TrainConfig
from mrcnn.model import MaskRCNN
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


# train set
train_set = KidneyDataset()
train_set.load_dataset('./kidneys', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# test/val set
test_set = KidneyDataset()
test_set.load_dataset('./kidneys', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# prepare config
config = TrainConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./models', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights(
  '/content/Mask_RCNN/models/kidney_cfg20200125T1032/mask_rcnn_kidney_cfg_0063.h5', 
by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=0.01, epochs=100, layers='5+')