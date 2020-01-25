from kidneyDataset import KidneyDataset
from matplotlib import pyplot as plt
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from trainConfig import TrainConfig
from mrcnn.model import MaskRCNN

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
model.load_weights('./mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')