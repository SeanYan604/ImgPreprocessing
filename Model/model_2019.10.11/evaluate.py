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
#from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
#from SSIM_PIL import compare_ssim as ssim
#import pytorch_ssim
import os
from imgaug import augmenters as iaa
import cv2
import scipy.io as scio


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

#AEGenerator_SK == AutoEncoderForGeneratorWithSmallKernel(3*3 Kernel),同时修改激活函数为:nn.LeakyRelu(0.2,True)
class AEGenerator_SK(nn.Module):
    def __init__(self):
        super(AEGenerator_SK, self).__init__()
        self.encoder = nn.Sequential( #input 1*256*256
            nn.Conv2d(1,32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 32*128*128
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32,32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 32*64*64
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32,64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 64*32*32
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64,64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 64*16*16
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64,128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 128*8*8
            nn.MaxPool2d((2,2))
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
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 256 * 16 * 16

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 256 * 32 * 32

            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 128 * 64 * 64
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 64 * 128 * 128

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),  # b, 16, 5, 5
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

def Contour_extraction(img_array):
    width = 256
    height = 256
    x_test = np.reshape(img_array, (1, 1, width, height))  # adapt this if using `channels_first` image data format
    #先增加一个维度

    batch_test=torch.Tensor(x_test)
    img = Variable(batch_test).cuda()
    # ===================forward=====================
    output = model(img)
    output_imgs = output.cpu().data.numpy()
    # noise_imgs = img.cpu().data.numpy()

    output_imgs = output_imgs * 255
    output_imgs = output_imgs.transpose(0,2,3,1)

    # noise_imgs = noise_imgs * 255
    # noise_imgs = noise_imgs.transpose(0,2,3,1)

    # contours = []
    for i,singleimg in enumerate(output_imgs):
        _,singleimg = cv2.threshold(singleimg, 170, 255, 0)
        # contours.append(singleimg)
    return singleimg

def regionGenerate(img):
    findcontourimg = img.copy()
    # if img.ndim < 3:
    #     findcontourimg = np.expand_dims(findcontourimg,axis=2)
    findcontourimg = np.clip(findcontourimg, 0, 255)# 归一化也行
    findcontourimg = np.array(findcontourimg,np.uint8)

    ret, binary = cv2.threshold(findcontourimg,127,255,cv2.THRESH_BINARY)
    contours,hierarchy= cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    regionOut = np.zeros((binary.shape[0],binary.shape[1]),np.uint8)*255

    isHoleExist = True
    c_max = []
    c_min = []
    lenContours = len(contours)
    if lenContours == 2:
        isHoleExist = False
    #print(lenContours)
    hierarchy0 = hierarchy[0]
    for i in range(lenContours):
        hierarchyI = hierarchy0[i]
        if  hierarchyI[3] == -1: #hierarchyI[0] == -1 and hierarchyI[1] == -1 and
            cnt = contours[i]
            c_max.append(cnt)
        if  hierarchyI[2] == -1:#hierarchyI[0] == -1 and hierarchyI[1] == -1 and
            cnt = contours[i]
            c_min.append(cnt)
    cv2.drawContours(regionOut, c_max, -1,  (255,255,255), cv2.FILLED)
    if isHoleExist:
        cv2.drawContours(regionOut, c_min, -1,  (0,0,0), cv2.FILLED)
        cv2.drawContours(regionOut, c_min, -1,  (255,255,255), 1)

    return regionOut


if __name__ == "__main__":
    
    gt_root = '../../DefectDataset/gt'
    noise_root = '../../DefectDataset/noise'
    pic_num = 27
    model_id = 400
    model_is_trained_parallel = True

    model = AEGenerator().cuda()
    if model_is_trained_parallel:    #如果使用服务器并行训练的模型需要加上以下的步骤
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model,device_ids=[0])
        model.to(device)
    # model.load_state_dict(torch.load('./model/aug/conv_aae_epoch_2990.pth'))
    
    checkpoint = torch.load('./gan/ae_epoch_{}.pth'.format(model_id))
    # here, checkpoint is a dict with the keys you defined before
    model.load_state_dict(checkpoint['model'])

    gt_files = [x.path for x in os.scandir(gt_root) if x.name.endswith(".png")]
    noise_files = [x.path for x in os.scandir(noise_root) if x.name.endswith(".png")]

    seg_gt_files = []
    seg_noise_files = []
    for i in range(pic_num):
        temp = [x for x in gt_files if x.split("_")[1]==str(i)]
        temp.sort(key=lambda x:int(x[-8:-4]))
        seg_gt_files.append(temp)
        temp = [x for x in noise_files if x.split("_")[1]==str(i)]
        temp.sort(key=lambda x:int(x[-8:-4]))
        seg_noise_files.append(temp)

    IoU_vect = []
    Precise_vect = []
    Recall_vect = []
    F1_vect = []
    for i, img_files in enumerate(seg_noise_files):
        gt_img_files = seg_gt_files[i]
        IoU = []
        Precise = []
        Recall = []
        F1 = []

        for idx in range(len(img_files)):
            gt_path = gt_img_files[idx]
            noise_path = img_files[idx]

            img = Image.open(noise_path)
            img = img.convert("L")
            if(img.width != 256 or img.height != 256):
                img = img.resize((256, 256), 1)
            data = np.asarray(img)
            data = data.astype('float32') / 255.
            noise_img = data.copy()

            img = Image.open(gt_path)
            img = img.convert("L")
            if(img.width != 256 or img.height != 256):
                img = img.resize((256, 256), 1)
            data = np.asarray(img)
            data = data.astype('float32') / 255.
            gt_img = data.copy()

            out_img = Contour_extraction(noise_img).astype('float32') / 255.

            one = np.ones(gt_img.shape, np.uint8)
            TP = np.sum(one[(out_img>0)*(gt_img>0)])
            FP = np.sum(one[(out_img-gt_img)>0])
            FN = np.sum(one[(gt_img-out_img)>0])

            gt_roi = regionGenerate(255*gt_img.astype(np.uint8))
            rec_roi = regionGenerate(255*out_img.astype(np.uint8))

            iou = np.sum(one[(rec_roi>0)*(gt_roi>0)])/np.sum(one[(rec_roi+gt_roi)>0])            
            precise = TP/(TP+FP)
            recall = TP/(TP+FN)
            f1 = 2*(precise*recall)/(precise + recall)

            IoU.append(iou)
            Precise.append(precise)
            Recall.append(recall)
            F1.append(f1)
            # print(noise_img.shape, noise_img.dtype, gt_img.shape, gt_img.dtype)
        
        IoU_mean = np.mean(IoU)
        Precise_mean = np.mean(Precise)
        Recall_mean = np.mean(Recall)
        F1_mean = np.mean(F1)

        IoU_vect.append(IoU_mean)
        Precise_vect.append(Precise_mean)
        Recall_vect.append(Recall_mean)
        F1_vect.append(F1_mean)

        print('IoU : {:.4f}   Precise : {:.4f}  Recall : {:.4f}  F1 : {:.4f}'.format(IoU_mean, Precise_mean, Recall_mean, F1_mean))
    
    IoU_vect = np.array(IoU_vect)
    Precise_vect = np.array(Precise_vect)
    Recall_vect = np.array(Recall_vect)
    F1_vect = np.array(F1_vect)

    data = np.vstack((IoU_vect,Precise_vect,Recall_vect,F1_vect))
    print(data.shape)

    mat = 'evaluate_data_dae.mat'
    scio.savemat(mat, {'data': data})