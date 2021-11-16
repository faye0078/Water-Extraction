import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image
from get_args import get_arguments

import sys
sys.path.append("..")
from other.tif_read_write import readTiff

def create_loaders(args, **kwargs):
    train_set = WaterSegmentation(args, split='train')
    val_set = WaterSegmentation(args, split='val')
    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader, num_class

class WaterSegmentation(data.Dataset):
    NUM_CLASSES = 2

    CLASSES = ['other', 'water']

    def __init__(self, args, split="train"):
        self.root = args.base_dir

        self.split = split
        self.args = args
        self.files = {}
        self.mean = args.normalise_params[0]
        self.std = args.normalise_params[1]
        self.crop = self.args.crop_size

        self.images_base = os.path.join(self.root, self.split[:3] + '_scene')
        self.annotations_base = os.path.join(self.root, self.split[:3] + '_truth')

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.tif')


        self.valid_classes = [0, 1]
        self.class_names = ['other', 'water']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-1][:-4].replace('_6Bands', '') + '_Truth.tif')

        _img, _ = readTiff(img_path)
        _img = cv2.normalize(_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)

        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)

        sample = {'image': _img, 'label': _tmp}
        return self.transform(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def get_transform(self):
        if self.split == 'train':
            return transform_tr(self.args, self.mean, self.std)
        elif self.split == 'val':
            return transform_val(self.args, self.mean, self.std)

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class FixedResize(object):
    """change the short edge length to size"""

    def __init__(self, resize=512):
        self.size1 = resize

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.shape[0] * img.shape[1] == mask.shape[0] * mask.shape[1]

        w = img.shape[0]
        h = img.shape[1]
        if w > h:
            oh = self.size1
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.size1
            oh = int(1.0 * h * ow / w)

        img = cv2.resize(
            img, (oh, ow), interpolation=cv2.INTER_NEAREST
        )
        mask = cv2.resize(
            mask, (oh, ow), interpolation=cv2.INTER_NEAREST
        )
        return {'image': img,
                'label': mask}


class RandomCrop(object):
    def __init__(self, crop_size=320):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        h, w = img.shape[:2]
        new_h = min(h, self.crop_size[0])
        new_w = min(w, self.crop_size[1])
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        img = img[top: top + new_h, left: left + new_w]
        mask = mask[top: top + new_h, left: left + new_w]
        return {'image': img,
                'label': mask}

class transform_tr(object):
    def __init__(self, args, mean, std):
        self.composed_transforms = transforms.Compose([
            FixedResize(resize=args.resize),
            RandomCrop(crop_size=args.crop_size),
            Normalize(mean, std),
            ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)


class transform_val(object):
    def __init__(self, args, mean, std):
        self.composed_transforms = transforms.Compose([
            FixedResize(resize=args.resize),
            RandomCrop(crop_size=args.crop_size),
            Normalize(mean, std),
            ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)

# test
if __name__ == '__main__':
    args = get_arguments()
    train_loader, val_loader, num_class = create_loaders(args)
    tbar = tqdm(train_loader)

    for i, sample in enumerate(tbar):
        a = sample
        print(i)
