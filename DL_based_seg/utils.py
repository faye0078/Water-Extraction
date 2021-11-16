import torch
from torch import nn
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    n_aux_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        if "aux" in name:
            n_aux_params += n_elem
        n_total_params += n_elem
    return n_total_params, n_total_params - n_aux_params

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def validation(args, model, val_loader, epoch, criterion):
    model.eval()
    evaluator = Evaluator(args.num_classes)
    evaluator.reset()
    tbar = tqdm(val_loader, desc='\r')
    losses = AverageMeter()

    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()
        target_var = torch.autograd.Variable(target).long()
        with torch.no_grad():
            output = model(image)

        soft_output = nn.LogSoftmax()(output)
        loss = criterion(soft_output, target_var)
        losses.update(loss.item())

        tbar.set_description('Train loss: %.3f' % (losses.avg))
        pred = soft_output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print('Validation:')
    print("Epoch:{}, Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(epoch, Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % losses.avg)
    new_pred = mIoU
    return new_pred


class imgPatch():
    '''
    author: xin luo, date: 2021.3.19
    description: 1. remote sensing image to multi-scale patches
                 2. patches to remote sensing image
    '''

    def __init__(self, img, patch_size, edge_overlay):
        ''' edge_overlay = left overlay or, right overlay
        edge_overlay should be an even number. '''
        self.patch_size = patch_size
        self.edge_overlay = edge_overlay
        self.img = img[:, :, np.newaxis] if len(img.shape) == 2 else img
        self.img_row = img.shape[0]
        self.img_col = img.shape[1]

    def toPatch(self):
        '''
        description: convert img to patches.
        return:
            patch_list, contains all generated patches.
            start_list, contains all start positions(row, col) of the generated patches.
        '''
        patch_list = []
        start_list = []
        patch_step = self.patch_size - self.edge_overlay
        img_expand = np.pad(self.img, ((self.edge_overlay, patch_step),
                                       (self.edge_overlay, patch_step), (0, 0)), 'constant')
        img_patch_row = (img_expand.shape[0] - self.edge_overlay) // patch_step
        img_patch_col = (img_expand.shape[1] - self.edge_overlay) // patch_step
        for i in range(img_patch_row):
            for j in range(img_patch_col):
                img = img_expand[i * patch_step:i * patch_step + self.patch_size, j * patch_step:j * patch_step + self.patch_size, :]
                img = np.array(img).astype(np.float32).transpose((2, 0, 1))
                img = torch.from_numpy(img).float()
                patch_list.append(img.cuda())
                start_list.append([i * patch_step - self.edge_overlay, j * patch_step - self.edge_overlay])
        return patch_list, start_list, img_patch_row, img_patch_col

    def higher_patch_crop(self, higher_patch_size, start_list):
        '''
        author: xin luo, date: 2021.3.19
        description: crop the higher-scale patch (centered by the given low-scale patch)
        input:
            img, np.array, the original image
            patch_size, int, the lower-scale patch size
            crop_size, int, the higher-scale patch size
            start_list, list, the start position (row,col) corresponding to the original image (generated by the toPatch function)
        return:
            higher_patch_list, list, contains higher-scale patches corresponding to the lower-scale patches.
        '''
        higher_patch_list = []
        radius_bias = higher_patch_size // 2 - self.patch_size // 2
        patch_step = self.patch_size - self.edge_overlay
        img_expand = np.pad(self.img, ((self.edge_overlay, patch_step), (self.edge_overlay, patch_step), (0, 0)),
                            'constant')
        img_expand_higher = np.pad(img_expand, ((radius_bias, radius_bias), (radius_bias, radius_bias), (0, 0)),
                                   'constant')
        start_list_new = list(np.array(start_list) + self.edge_overlay + radius_bias)
        for start_i in start_list_new:
            higher_row_start, higher_col_start = start_i[0] - radius_bias, start_i[1] - radius_bias
            higher_patch = img_expand_higher[higher_row_start:higher_row_start + higher_patch_size,
                           higher_col_start:higher_col_start + higher_patch_size, :]
            higher_patch_list.append(higher_patch)
        return higher_patch_list

    def toImage(self, patch_list, img_patch_row, img_patch_col):
        patch_list = [
            patch.cpu().numpy().transpose((1, 2, 0))[self.edge_overlay // 2:-self.edge_overlay // 2, self.edge_overlay // 2:-self.edge_overlay // 2, :]
            for patch in patch_list]
        patch_list = [np.hstack((patch_list[i * img_patch_col:i * img_patch_col + img_patch_col]))
                      for i in range(img_patch_row)]
        img_array = np.vstack(patch_list)
        img_array = img_array[self.edge_overlay // 2:self.img_row + self.edge_overlay // 2, \
                    self.edge_overlay // 2:self.img_col + self.edge_overlay // 2, :]

        return img_array