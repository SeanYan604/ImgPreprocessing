import cv2
import glob
from PIL import Image
import numpy as np
import os
import region 
import math

def Maskgeneration(img_num, foldernum):
    for folder_num in range(1, foldernum+1):
        for i in range(1, img_num+1):
            ori_file = '../Dataset/origin_img/{:02d}.png'.format(i)
            def_file = '../Dataset/defect_img/{:02d}/{:02d}.png'.format(folder_num, i)
            mask_file = '../Dataset/Mask_img/{:02d}/{:02d}.png'.format(folder_num, i)

            ori_img = cv2.imread(ori_file)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
            def_img = cv2.imread(def_file)
            def_img = cv2.cvtColor(def_img, cv2.COLOR_BGR2GRAY)

            mask = np.zeros(ori_img.shape, np.uint8)
            ori_img = ori_img.astype(np.float)
            def_img = def_img.astype(np.float)
            sub = ori_img - def_img
            # print(sub.dtype)
            mask[np.abs(sub) > 70] = 255

            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)

            cv2.imwrite(mask_file, mask)

if __name__ == "__main__":
    img_num = 27
    foldernum = 11
    Maskgeneration(img_num, foldernum)

