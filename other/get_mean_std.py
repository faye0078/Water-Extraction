import numpy as np
import cv2
import os
from tif_read_write import readTiff
# img_h, img_w = 32, 32
img_h, img_w = 32, 48  # 根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []

imgs_path = 'D:/LULC_data/dset-s2/tra_scene'
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img, _ = readTiff(os.path.join(imgs_path, item))
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    img = cv2.resize(img, (img_w, img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i, '/', len_)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(6):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))