import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from numpy import int64 as int64
from torch.functional import split
from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image, ImageOps, ImageFilter

def create_loaders(args, **kwargs):
    train_set = WaterSegmentation(args, split='train')
    val_set = WaterSegmentation(args, split='val')
    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader, num_class

class WaterSegmentation(data.Dataset):
    NUM_CLASSES = 19

    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    def __init__(self, args, root='', split="train"):
        self.root = root
        self.root = 'D:/DPCode/data/cityscapes/'
        self.split = split
        self.args = args
        self.files = {}
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.crop = self.args.crop_size

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')


        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']

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
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}
        return self.transform(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
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
            return tr.transform_tr(self.args, self.mean, self.std)
        elif self.split == 'val':
            return tr.transform_val(self.args, self.mean, self.std)

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


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


# resize to 512*1024
class FixedResize(object):
    """change the short edge length to size"""

    def __init__(self, resize=512):
        self.size1 = resize  # size= 512

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        w, h = img.size
        if w > h:
            oh = self.size1
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.size1
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        return {'image': img,
                'label': mask}


# random crop 321*321
class RandomCrop(object):
    def __init__(self, crop_size=320):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': img,
                'label': mask}


class RandomScale(object):
    def __init__(self, scales=(1,)):
        self.scales = scales

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        scale = random.choice(self.scales)
        w, h = int(w * scale), int(h * scale)
        return {'image': img,
                'label': mask}


class Retrain_Preprocess(object):
    def __init__(self, flip_prob, scale_range, crop, mean, std):
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.crop = crop
        self.data_transforms = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=mean, std=std)])

    def __call__(self, sample):
        if self.flip_prob is not None and random.random() < self.flip_prob:
            sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            sample['label'] = sample['label'].transpose(Image.FLIP_LEFT_RIGHT)

        if self.scale_range is not None:
            w, h = sample['image'].size
            rand_log_scale = math.log(self.scale_range[0], 2) + random.random() * \
                (math.log(self.scale_range[1], 2) - math.log(self.scale_range[0], 2))
            random_scale = math.pow(2, rand_log_scale)
            new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
            sample['image'] = sample['image'].resize(new_size, Image.ANTIALIAS)
            sample['label'] = sample['label'].resize(new_size, Image.NEAREST)
        sample['image'] = self.data_transforms(sample['image'])
        sample['label'] = torch.LongTensor(np.array(sample['label']).astype(int64))

        if self.crop:
            image, mask = sample['image'], sample['label']
            h, w = image.shape[1], image.shape[2]
            pad_tb = max(0, self.crop[0] - h)
            pad_lr = max(0, self.crop[1] - w)
            image = nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
            mask = nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

            h, w = image.shape[1], image.shape[2]
            i = random.randint(0, h - self.crop[0])
            j = random.randint(0, w - self.crop[1])
            sample['image'] = image[:, i:i + self.crop[0], j:j + self.crop[1]]
            sample['label'] = mask[i:i + self.crop[0], j:j + self.crop[1]]
        return sample


class transform_tr(object):
    def __init__(self, args, mean, std):
        if args.multi_scale is None:
            self.composed_transforms = transforms.Compose([
                FixedResize(resize=args.resize),
                RandomCrop(crop_size=args.crop_size),
                # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
                # tr.RandomGaussianBlur(),
                Normalize(mean, std),
                ToTensor()])
        else:
            self.composed_transforms = transforms.Compose([
                FixedResize(resize=args.resize),
                RandomScale(scales=args.multi_scale),
                RandomCrop(crop_size=args.crop_size),
                # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
                # tr.RandomGaussianBlur(),
                Normalize(mean, std),
                ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)


class transform_val(object):
    def __init__(self, args, mean, std):
        self.composed_transforms = transforms.Compose([
            FixedResize(resize=args.resize),
            FixScaleCrop(crop_size=args.crop_size),  # TODO:CHECK THIS
            Normalize(mean, std),
            ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)
