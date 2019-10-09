#利用加噪声的图像进行降噪自编码器的测试
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST


import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

import cv2
import pySQI
import pyGTemplate
import testAAE
import time
import region
# import Imgprocessing


class AEGenerator(nn.Module):
    def __init__(self):
        super(AEGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32, 5, stride=2, padding=2),
            nn.ReLU(True),# 64*128*128

            nn.Conv2d(32,32, 5, stride=2, padding=2),
            nn.ReLU(True),# 128*64*64

            nn.Conv2d(32,64, 5, stride=2, padding=2),
            nn.ReLU(True),# 256*32*32

            nn.Conv2d(64,64, 5, stride=2, padding=2),
            nn.ReLU(True),# 256*16*16

            nn.Conv2d(64,128, 5, stride=2, padding=2),
            nn.ReLU(True)# 512*8*8
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*8*8, 128),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128 * 8 * 8),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 256 * 16 * 16

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 256 * 32 * 32

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 128 * 64 * 64

            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 64 * 128 * 128

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.Sigmoid() # 1 * 256 * 256            
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.decoder(x)
        return x


def preprocessing(total_num, sample_id, threshold, exposure, write_flag):
    

    sobel_mask_vect = []
    src_vect = []
    sobel_x =np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype=np.float32)
    sobel_y =np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]], dtype=np.float32)
    new_img = np.zeros((256,256), np.uint8)
    for pic_num in range(1, total_num):
        if write_flag:
            src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.jpg'
            output_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.png'

            IN_src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '_IN/' + 'SQI' + '/' + '{:02d}'.format(pic_num) + '.png'
            # output_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '_IN/' + 'TT' + '/' + '{:02d}'.format(pic_num) + '.png'
            # region_file = './roi/region_' + str(pic_num) + '.png'

            print(src_file)
            img = cv2.imread(src_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            m,n = img.shape
            img = img[0:n]
            new_img[3:253,3:253] = img
            cv2.imwrite(output_file, new_img)
            new_img_copy = new_img.copy()

            # IN_img = cv2.imread(IN_src_file)
            # IN_img = cv2.cvtColor(IN_img, cv2.COLOR_BGR2GRAY)
            # src_vect.append(IN_img)
        else:
            src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.png'
            # IN_src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '_IN/' + 'SQI' + '/' + '{:02d}'.format(pic_num) + '.png'
            new_img = cv2.imread(src_file)
            new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
            # IN_img = cv2.imread(IN_src_file)
            # IN_img = cv2.cvtColor(IN_img, cv2.COLOR_BGR2GRAY)
            # src_vect.append(IN_img)


        sobel_mag = np.zeros(new_img.shape, np.float)
        # sobel_angle = np.zeros(new_img.shape, np.float)
        # quantized_angle = np.zeros(new_img.shape, np.uint8)
        sobel_mask = np.zeros(new_img.shape, np.uint8)

        # img_Guassian = cv2.GaussianBlur(new_img,(5,5),0)
        # img_Guassian.astype(np.uint8)
        # m,n = img_Guassian.shape

        # m,n = new_img.shape
        # for i in range(2,m-1):
        #     for j in range(2,n-1):
        #         Gx = np.sum(new_img[i-1:i+2, j-1:j+2] * sobel_x)
        #         Gy = np.sum(new_img[i-1:i+2, j-1:j+2] * sobel_y) 
        #         sobel_mag[i,j] = math.sqrt(math.pow(Gx,2) + math.pow(Gy,2))
        #         sobel_angle[i,j] = math.atan2(Gy, Gx) * 180 / math.pi
        #         # quantized_angle[i,j] = quantizeAngle(sobel_angle[i,j])
        #         if sobel_mag[i,j] >= threshold:
        #             sobel_mask[i,j] = 1
        # contour = angleFilter(sobel_mask, quantized_angle)
        # contour = cv2.blur(contour, (3,3))
        # sobelx = cv2.Sobel(new_img,cv2.CV_32F,1,0)   #默认ksize=3
        # sobely = cv2.Sobel(new_img,cv2.CV_32F,0,1)

        sobelx = cv2.filter2D(new_img, cv2.CV_32F, sobel_x)
        sobely = cv2.filter2D(new_img, cv2.CV_32F, sobel_y)
        sobel_mag = np.sqrt(pow(sobelx,2) + pow(sobely,2))
        # sobel_angle = np.arctan2(sobely,sobelx) * 180 /math.pi
        sobel_mag = cv2.convertScaleAbs(sobel_mag)
        _, sobel_mask = cv2.threshold(sobel_mag, threshold, 255, 0)

        # contour = angleFilter(sobel_mask, sobel_angle)
        # contour = cv2.blur(contour, (3,3))
        # sobel_mask = cv2.blur(sobel_mask, (3,3))
        # contour_vect.append(contour)

        # cv2.imshow('sobel', sobel_mask)
        # cv2.waitKey(0)
        sobel_mask_vect.append(sobel_mask)

    return sobel_mask_vect

def Contour_extraction(img_files):
    width = 256
    height = 256
    x_truth = np.reshape(img_files, (len(img_files), width, height, 1))  # adapt this if using `channels_first` image data format
    #先增加一个维度
    # user_emb_dims = np.expand_dims(self.user_emb, axis=0)
    # user_emb_dims.shape
    x_test = x_truth
    x_truth = np.array(x_truth)
    x_truth = x_truth.astype('float32') / 255.
    x_test = np.array(x_test)
    x_test = x_test.astype('float32') / 255.
    x_truth = np.reshape(x_truth, (len(x_truth),1, width, height))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test),1, width, height))  # adapt this if using `channels_first` image data format

    batch_test=torch.Tensor(x_test)
    img = Variable(batch_test).cuda()
    # ===================forward=====================
    output = model(img)
    output_imgs = output.cpu().data.numpy()
    noise_imgs = img.cpu().data.numpy()

    output_imgs = output_imgs * 255
    output_imgs = output_imgs.transpose(0,2,3,1)

    noise_imgs = noise_imgs * 255
    noise_imgs = noise_imgs.transpose(0,2,3,1)

    contours = []
    for i,singleimg in enumerate(output_imgs):
        _,singleimg = cv2.threshold(singleimg, 170, 255, 0)
        contours.append(singleimg)
    return contours


if __name__ == "__main__":

    total_num = 2
    sample_id = 0  
    threshold = 160
    exposure = 6
    write_flag = False

    W = 40
    H = 60
    ParaName = 'parameter.npy'

# ---------------- Load model ----------------------
    
    model_id = 802
    model_is_trained_parallel = True

    if not os.path.exists('../Test_Image'):
        os.mkdir('../Test_Image')
    if not os.path.exists('../Test_Image/input'):
        os.mkdir('../Test_Image/input')
    if not os.path.exists('../Test_Image/output'):
        os.mkdir('../Test_Image/output')
    # Setting Image Propertie
    model = AEGenerator().cuda()
    if model_is_trained_parallel:    #如果使用服务器并行训练的模型需要加上以下的步骤
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model,device_ids=[0])
        model.to(device)
    # model.load_state_dict(torch.load('./model/aug/conv_aae_epoch_2990.pth'))
    
    checkpoint = torch.load('../Model/GAN/aegan_epoch_{}.pth'.format(model_id))
    # here, checkpoint is a dict with the keys you defined before
    model.load_state_dict(checkpoint['model'])




    st = time.time()
    IN_img_vect = []
    for pic_num in range(1, total_num):
        src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.png'
        img_in = cv2.imread(src_file)
        imgout = pySQI.SQI(img_in)
        IN_img_vect.append(imgout)

    ed = time.time()
    print("IN_Cost:{:.5f}".format(ed-st))

    time_start = time.time()
    sobel_mask_vect = preprocessing(total_num, sample_id, threshold, exposure, write_flag)
    time_end = time.time()
    print('Proprecessing time cost:{:.3f}'.format(time_end - time_start))

    contour_vect = Contour_extraction(sobel_mask_vect)
    print('AAE time cost:{:.3f}'.format(time.time() - time_end))

    for i, singleimg in enumerate(contour_vect):
        # singleimg = np.squeeze(singleimg, axis=(2,))
        singleimg = singleimg.astype(np.uint8)
        src = IN_img_vect[i]
        # cv2.imshow('src',src)
        # cv2.waitKey(0)
        # region_file = '../roi/region_{:02d}'.format(i) + '.png'
        # mask_file = '../Template/bin_mask/region_{:02d}'.format(i) + '.png'
        mask = region.regionGenerate(singleimg)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        eroded = cv2.erode(mask,kernel)
        eroded_2 = cv2.erode(eroded,kernel)
        eroded_3 = cv2.erode(eroded_2,kernel)
        roi = cv2.bitwise_and(src, src, mask=eroded)
        sub = eroded - eroded_3

        # print(roi.shape, eroded.shape)
        # m,n = sub.shape
        # for row in range(m):
        #     for col in range(n):
        #         if sub[row, col] and roi[row, col] < 80:
        #             roi[row,col] = 0
        #             eroded[row, col] = 0
        roi[(sub>0)*(roi<80)] = 0
        eroded[(sub>0)*(roi<80)] = 0

        # background = cv2.bitwise_not(eroded)            
        # cv2.imwrite(region_file, roi)
        # cv2.imwrite(mask_file, eroded)
        # cv2.imshow('region', roi+background)
        # cv2.waitKey(0)

        #--------------------defect detection 
        # cv2.imshow('roi', roi) 
        # cv2.imshow('eroded', eroded)
        # cv2.waitKey(0)


        start = time.time()
        defect_mask, defect_rgb = pyGTemplate.TempGenAndDetection(ParaName, W, H, roi, eroded)
        end = time.time()
        print('Detection cost:{:.4f}'.format(end - start))

        result_file = '../Results/defect_rgb_{:02d}'.format(i) + '.png'
        mask_file = '../Results/defect_mask_{:02d}'.format(i) + '.png'

        cv2.imwrite(result_file, defect_rgb)
        # cv2.imwrite(mask_file, defect_mask)

    print('Totally time cost:{:.3f}'.format(time.time() - st)) 

    