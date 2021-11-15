import cv2
import os

base_path = 'D:/LULC_data/鄱阳湖_00-16年10-11月影像数据/'
for dirnames in os.listdir(base_path):
    # b2 b3 b4 = G B R

    img_b2_path = base_path + dirnames + '/' + dirnames + '_B2.TIF'
    img_b3_path = base_path + dirnames + '/' + dirnames + '_B3.TIF'
    img_b4_path = base_path + dirnames + '/' + dirnames + '_B4.TIF'

    img_b2 = cv2.imread(img_b2_path, cv2.IMREAD_GRAYSCALE)
    img_b3 = cv2.imread(img_b3_path, cv2.IMREAD_GRAYSCALE)
    img_b4 = cv2.imread(img_b4_path, cv2.IMREAD_GRAYSCALE)

    RGB_img = cv2.merge([img_b2, img_b3, img_b4])
    RGB_img = cv2.resize(RGB_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imwrite(base_path + 'rgb/' + dirnames[17:25] + '.jpg', RGB_img)

cv2.imread('', cv2.IMREAD_GRAYSCALE)