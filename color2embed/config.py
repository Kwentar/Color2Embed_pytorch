IMAGE_SIZE = 256

# Path to ImageNet list, list is one path to image per line, sample:
# n09428293/n09428293_23938.JPEG
# n09428293/n09428293_35035.JPEG
# n09428293/n09428293_46724.JPEG
IMAGENET_LIST = '../imagenet.list'

# Prefix to imagenet folder, os.path.join(imagenet_prefix, imagenet_list[i]) must be path to image
IMAGENET_PREFIX = '../datasets/images/ILSVRC/Data/CLS-LOC/train'

TRAIN_BATCH_SIZE_PER_GPU = 32
TRAIN_LR = 0.001
TRAIN_EPOCHS = 10

COLOR_EMBEDDING_DIM = 512
