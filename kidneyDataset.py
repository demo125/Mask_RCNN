# detect kidneys in photos with mask rcnn model
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
import os 
import glob
# class that defines and loads the kidney dataset
class KidneyDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		self.add_class("dataset", 1, "kidney")
		
		dirname = os.path.dirname(__file__)
		dataset_dir = os.path.join(dirname, dataset_dir)

		files = glob.glob(os.path.join(dataset_dir, '**','*'), recursive=True)
		
		image_id = 0
		for f in files:
			
			if f.endswith('.jpg'):
				xml = f.replace('.jpg', '.xml')
				if not os.path.isfile(xml):
					xml = os.path.join(dataset_dir, 'empty.xml')

				if (is_train and int(image_id) <= 150) or (not is_train and int(image_id) >= 150):

					img_path = f
					ann_path = xml
					self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
				image_id += 1

	# load all bounding boxes for an image
	def extract_boxes(self, filename):
		# load and parse the file
		root = ElementTree.parse(filename)
		boxes = list()
		# extract each bounding box
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
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('kidney'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
