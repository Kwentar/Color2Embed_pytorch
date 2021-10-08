import os

import albumentations
import albumentations.pytorch
import cv2
from torch.utils.data import Dataset

import config
from tsp import warp_image_cv


class Color2EmbedDataset(Dataset):
    def __init__(self, verbose=True):
        super(Color2EmbedDataset, self).__init__()

        self.prefix = config.IMAGENET_PREFIX
        self.image_size = config.IMAGE_SIZE
        self.imagenet_list = config.IMAGENET_LIST

        self.transform_color_source_aug = albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.GaussNoise(var_limit=0.5),
            ]
        )

        self.transform_torch = albumentations.Compose(
            [
                albumentations.ToFloat(),
                albumentations.pytorch.ToTensorV2(),
            ]
        )

        with open(self.imagenet_list) as f:
            self.lines = [x.strip() for x in f.readlines()]

        if verbose:
            print(f'Init finished, num images = {len(self.lines)}')

    def __getitem__(self, index):
        image_path = os.path.join(self.prefix, self.lines[index])
        l_channel, ground_truth_ab, ground_truth_rgb, color_source = self.process_image(image_path)

        return l_channel, ground_truth_ab, ground_truth_rgb, color_source

    def __len__(self):
        return len(self.lines)

    def process_image(self, image_path):
        lenna = cv2.imread(image_path)
        ground_truth_rgb = cv2.resize(lenna, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(ground_truth_rgb, cv2.COLOR_BGR2LAB)
        tsp = warp_image_cv(ground_truth_rgb)

        color_source = self.transform_color_source_aug(image=tsp)['image']
        color_source = self.transform_torch(image=color_source)['image']
        lab = self.transform_torch(image=lab)['image']
        ground_truth_rgb = self.transform_torch(image=ground_truth_rgb)['image']
        l_channel = lab[0, ...].unsqueeze(0)
        ground_truth_ab = lab[1:, ...]
        return l_channel, ground_truth_ab, ground_truth_rgb, color_source


if __name__ == '__main__':
    ds = Color2EmbedDataset()
    for l_channel_, ground_truth_ab_, ground_truth_rgb_, color_source_ in ds:
        print(l_channel_.shape, ground_truth_ab_.shape, ground_truth_rgb_.shape, color_source_.shape)
        break
