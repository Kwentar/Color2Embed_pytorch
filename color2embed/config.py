NORMALIZE_MEDIAN_BGR = [0.485, 0.456, 0.406]
NORMALIZE_STD_BGR = [0.229, 0.224, 0.225]

IMAGE_SIZE = 256

# Path to ImageNet list, list is one path to image per line, sample:
# n09428293/n09428293_23938.JPEG
# n09428293/n09428293_35035.JPEG
# n09428293/n09428293_46724.JPEG
IMAGENET_LIST = '../imagenet.list'
# Prefix to imagenet folder, os.path.join(imagenet_prefix, imagenet_list[i]) must be path to image

IMAGENET_PREFIX = '../datasets/images/ILSVRC/Data/CLS-LOC/train'

TRAIN_BATCH_SIZE_PER_GPU = 2
TRAIN_LR = 0.0001
TRAIN_EPOCHS = 20

COLOR_EMBEDDING_DIM = 512
