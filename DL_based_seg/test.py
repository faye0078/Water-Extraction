import numpy as np
import logging
import torch
import cv2
import torch.nn as nn
from utils import imgPatch
from utils import compute_params
from model import get_model
from get_args import get_arguments

import sys
sys.path.append("..")
from other.tif_read_write import readTiff, writeTiff

def transform(sample, mean, std):
    img = sample['image']

    img = np.array(img).astype(np.float32)
    img /= 255.0
    img -= mean
    img /= std

    return {'image': img}

def main(input_path, output_path, model_path):
    args = get_arguments()
    logger = logging.getLogger(__name__)
    logger.debug(args)
    _img, img_info = readTiff(input_path)
    _img = cv2.normalize(_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    sample = {'image': _img}

    mean = [0.08563971, 0.114898555, 0.11390314, 0.2771133, 0.22948432, 0.1595726]
    std = [0.07971889, 0.0962829, 0.12481037, 0.20359233, 0.20531356, 0.17086406]

    sample = transform(sample, mean, std)

    image = sample['image']
    net = get_model()

    logger.info(
        " Loaded Encoder with #TOTAL PARAMS={:3.2f}M".format(
            compute_params(net)[0] / 1e6
        )
    )
    net.load_state_dict(
        torch.load(model_path), strict=False
    )

    result_patch_list = []
    imgPatch_ins = imgPatch(image, patch_size=512, edge_overlay=80)
    patch_list, start_list, img_patch_row, img_patch_col = imgPatch_ins.toPatch()
    i = 0
    for patch in patch_list:
        with torch.no_grad():
            output = net(patch.unsqueeze(0))
            i = i+1
            print(str(i) + '/' + str(img_patch_row * img_patch_col))
            soft_output = nn.LogSoftmax()(output)
            result_patch_list.append(soft_output)
    result_patch_list = [np.squeeze(patch, axis=0) for patch in result_patch_list]
    pred = imgPatch_ins.toImage(result_patch_list, img_patch_row, img_patch_col)

    pred = np.argmax(pred, axis=2)

    writeTiff(im_data=pred.astype(np.int8),
              im_geotrans=img_info['geotrans'],
              im_geosrs=img_info['geosrs'],
              path_out=output_path)



if __name__ == '__main__':
    input_path = 'D:/LULC_data/after_pre/2002/2002.tif'
    output_path = 'D:/LULC_data/after_pre/2002/test.tif'
    model_paths = "../saved_model/best_hrnet_ach.pth"
    main(input_path, output_path, model_paths)