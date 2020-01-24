from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import Dataset
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import skimage.color
import os 
import glob

# class that defines and loads the kangaroo dataset
class KangarooDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "kidney")
        
        dirname = os.path.dirname(__file__)
        dataset_dir = os.path.join(dirname, dataset_dir)

        files = glob.glob(dataset_dir+ r'\**\*', recursive=True)
        image_id = 0
        for f in files:
            
            if f.endswith('.jpg'):
                xml = f.replace('.jpg', '.xml')
                if not os.path.isfile(xml):
                    xml = os.path.join(dataset_dir, 'empty.xml')

                if is_train and int(image_id) >= 180:
                    continue
                
                if not is_train and int(image_id) < 180:
                    continue

                img_path = f
                ann_path = xml
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
                
                image_id += 1
                
                # print(f)
                # print(xml)
                # print()
            
    def old():       
        # define one class
        self.add_class("dataset", 1, "kangaroo")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            # skip bad images
            if image_id in ['00090']:
                continue
            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= 150:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < 150:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        if len(boxes):
            masks = zeros([h, w, len(boxes)], dtype='uint8')
            # create masks
            class_ids = list()
            for i in range(len(boxes)):
                box = boxes[i]
                row_s, row_e = box[1], box[3]
                col_s, col_e = box[0], box[2]
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('kidney'))
        else:
            class_ids = list()
            masks = zeros([h, w], dtype='uint8')
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# train set
train_set = KangarooDataset()
train_set.load_dataset('./kidneys', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# test/val set
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# image_id = 50
# image = train_set.load_image(image_id)
# print(image.shape)
# # load image mask
# mask, class_ids = train_set.load_mask(image_id)
# print(mask.shape)
# pyplot.imshow(image)
# if mask.ndim != 3:
#     mask = skimage.color.gray2rgb(mask)
#     pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
# else:
#     for j in range(mask.shape[2]):
#         pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.2)
    
# pyplot.savefig("./kidney.png")

# plot first few images
# for i,id in enumerate(range(30, 120, 10)):
# 	# define subplot
# 	pyplot.subplot(330 + 1 + i)
# 	# plot raw pixel data
# 	image = train_set.load_image(id)
# 	pyplot.imshow(image)
# 	# plot all masks
# 	mask, _ = train_set.load_mask(id)
# 	if mask.ndim != 3:
# 		mask = skimage.color.gray2rgb(mask)
# 		pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
# 	else:
# 		for j in range(mask.shape[2]):
# 			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.2)
# # show the figure
# pyplot.savefig('./kidneys.png')


image_id = 50
# load the image
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names, save_dir="./instances.png")
