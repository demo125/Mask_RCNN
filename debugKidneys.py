from kidneyDataset import KidneyDataset
from matplotlib import pyplot as plt
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

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

image_id = 50
image = train_set.load_image(image_id)
print(image.shape)
# load image mask
mask, class_ids = train_set.load_mask(image_id)
print(mask.shape)
plt.imshow(image)
if mask.ndim != 3:
    mask = skimage.color.gray2rgb(mask)
    plt.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
else:
    for j in range(mask.shape[2]):
        plt.imshow(mask[:, :, j], cmap='gray', alpha=0.2)
    
plt.savefig("./kidney.png")

# plot first few images
for i,id in enumerate(range(30, 120, 10)):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	image = train_set.load_image(id)
	plt.imshow(image)
	# plot all masks
	mask, _ = train_set.load_mask(id)
	if mask.ndim != 3:
		mask = skimage.color.gray2rgb(mask)
		plt.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
	else:
		for j in range(mask.shape[2]):
			plt.imshow(mask[:, :, j], cmap='gray', alpha=0.2)
# show the figure
plt.savefig('./kidneys.png')


image_id = 50
# load the image
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names, save_dir="./instances.png")
