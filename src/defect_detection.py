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

import scipy.io as scio
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

            # print(src_file)
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
            src_file = '../Dataset/defect_img/{:02}.png'.format(pic_num)
            # src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.png'
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

def Contour_extraction(img_files, model):
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

def Template_method():

    total_num = 28
    foldernum = 11
    # sample_id = 0  
    threshold = 160
    # exposure = 6
    # write_flag = False
    evaluate_flag = False
    extract_CF = False

    # W = 30
    # H = 20
    Wh = 0.3
    Wl = 0.5

    Wh_vect = np.array([Wh])
    Wl_vect = np.array([Wl])
    if(evaluate_flag):
        Wh_vect = np.linspace(0.1,0.9, 85, endpoint= True)
        Wl_vect = np.linspace(0.1,0.9, 85, endpoint= True)
        # print(Wh_vect)

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
    # if not os.path.exists('../Detection_results'):
    #     os.mkdir('../Detection_results')
        
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

    F1_evaluate = []
    Precise_evaluate = []
    Recall_evaluate = []

    index = 0
    total_time = []
    sqi_time = []
    ROI_time = []
    detection_time = []

    initial_template = pyGTemplate.inittempGeneration(ParaName, [256,256])

    for Wl_idx, Wl in enumerate(Wl_vect):

        # st = time.time()
        IN_img_vect = []
        GT_img_vect = []
        sobel_mask_vect = []
        sobel_x =np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype=np.float32)
        sobel_y =np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]], dtype=np.float32)
        # new_img = np.zeros((256,256), np.uint8)

        Precise = []
        Recall = []
        F1 = []
        for folder_num in range(1, foldernum+1):

            Precise_mean = []
            Recall_mean = []
            F1_mean = []

            for pic_num in range(1, total_num):
                # src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.png'
                if extract_CF:
                    src_file = '../Dataset/origin_img/{:02}.png'.format(pic_num)
                else:
                    src_file = '../Dataset/defect_img/{:02}/{:02}.png'.format(folder_num, pic_num)
                
                index += 1
                gt_file = '../Dataset/Mask_img/{:02}/{:02}.png'.format(folder_num, pic_num)
                

                img_in = cv2.imread(src_file)
                img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
                img_gt = cv2.imread(gt_file)
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)

                sqi_start = time.time()
                imgout = pySQI.SQI(img_in)
                sqi_end = time.time()
                sqi_time.append(sqi_end - sqi_start)

                # IN_img_vect.append(imgout)
                # GT_img_vect.append(img_gt)

                sobel_mag = np.zeros(img_in.shape, np.float)
                sobel_mask = np.zeros(img_in.shape, np.uint8)

                sobelx = cv2.filter2D(img_in, cv2.CV_32F, sobel_x)
                sobely = cv2.filter2D(img_in, cv2.CV_32F, sobel_y)
                sobel_mag = np.sqrt(pow(sobelx,2) + pow(sobely,2))
                sobel_mag = cv2.convertScaleAbs(sobel_mag)
                _, sobel_mask = cv2.threshold(sobel_mag, threshold, 255, 0)
                sobel_mask_vect.append(sobel_mask)

                contour = Contour_extraction([sobel_mask], model)
                single_img = contour[0].astype(np.uint8)
                mask = region.regionGenerate(single_img)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
                eroded = cv2.erode(mask,kernel)
                eroded_2 = cv2.erode(eroded,kernel)
                eroded_3 = cv2.erode(eroded_2,kernel)
                roi = cv2.bitwise_and(imgout, imgout, mask=eroded)
                sub = eroded - eroded_3

                roi[(sub>0)*(roi<80)] = 0
                eroded[(sub>0)] = 0
                
                if extract_CF:
                    region_file = '../roi/region_{:02d}'.format(pic_num) + '.png'
                    mask_file = '../Template/bin_mask/region_{:02d}'.format(pic_num) + '.png'     
                    cv2.imwrite(region_file, roi)
                    cv2.imwrite(mask_file, eroded)

                detect_start = time.time()
                ROI_time.append(detect_start - sqi_end)
                defect_mask, defect_rgb = pyGTemplate.TempGenAndDetection(ParaName, Wh, Wl, initial_template, imgout, roi, eroded)
                detect_end = time.time()
                detection_time.append(detect_end - detect_start)
                total_time.append(detect_end - sqi_start)
                # print('Detection cost:{:.4f}'.format(end - start))

                # result_file = '../Results/defect_rgb_{:02d}'.format(pic_num) + '.png'
                mask_file = '../Results/mask/{:03d}.png'.format(index)
                result_file = '../Results/rgb/{:03d}.png'.format(index)
                
                if not (extract_CF and evaluate_flag):
                    cv2.imwrite(mask_file, defect_mask)
                    cv2.imwrite(result_file, defect_rgb)
                # cv2.imwrite(result_file, defect_rgb)
                # cv2.imwrite(mask_file, defect_mask)
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
                # defect_mask = cv2.dilate(defect_mask, kernel)
                # result = np.zeros(defect_mask.shape, np.uint8)
                # result[(defect_mask_>0)*(sobel_mask>0)+(defect_mask>0)] = 255

                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
                # result = cv2.dilate(result, kernel)
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
                # result = cv2.erode(result, kernel)

                one = np.ones(defect_mask.shape, np.uint8)
                Inter = one[(defect_mask>0)*(img_gt>0)]
                Union = one[(defect_mask>0)+(img_gt>0)]

                IoU = np.sum(Inter)/np.sum(Union)
                if np.sum(Inter) == 0 and np.sum(Union) == 0 :
                    IoU = 1
                
                TP = np.sum(one[(defect_mask>0)*(img_gt>0)])
                FP = np.sum(one[(defect_mask>0)*(img_gt==0)])
                FN = np.sum(one[(defect_mask==0)*(img_gt>0)])

                precise = TP/(TP+FP)
                recall = TP/(TP+FN)
                if (TP+FP)==0:
                    precise = 0
                if (TP+FN)==0:
                    recall = 0
                if recall==0 and precise==0:
                    F1 = 0
                else:
                    F1 = 2*(precise*recall)/(precise + recall)
                print('IoU : {:.4f}   Precise : {:.4f}  Recall : {:.4f}  F1 : {:.4f}'.format(IoU, precise, recall, F1))

                # if precise:
                Precise_mean.append(precise)
                # if recall:
                Recall_mean.append(recall)
                # if F1:
                F1_mean.append(F1)


                # cv2.imshow('mask', defect_mask)
                # cv2.imshow('defect', defect_rgb)
                # cv2.waitKey(0)


            # ed = time.time()
            # print("IN_Cost:{:.5f}".format(ed-st))

            # time_start = time.time()
            # sobel_mask_vect = preprocessing(total_num, sample_id, threshold, exposure, write_flag)
            # time_end = time.time()
            # print('Proprecessing time cost:{:.3f}'.format(time_end - time_start))

            # contour_vect = Contour_extraction(sobel_mask_vect)
            # print('AAE time cost:{:.3f}'.format(time.time() - time_end))

            # for i, singleimg in enumerate(contour_vect):
            #     # singleimg = np.squeeze(singleimg, axis=(2,))
            #     singleimg = singleimg.astype(np.uint8)
            #     src = IN_img_vect[i]
            #     # cv2.imshow('src',src)
            #     # cv2.waitKey(0)
            #     region_file = '../roi/region_{:02d}'.format(i) + '.png'
            #     mask_file = '../Template/bin_mask/region_{:02d}'.format(i) + '.png'
            #     mask = region.regionGenerate(singleimg)
                
            #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
            #     eroded = cv2.erode(mask,kernel)
            #     eroded_2 = cv2.erode(eroded,kernel)
            #     eroded_3 = cv2.erode(eroded_2,kernel)
            #     roi = cv2.bitwise_and(src, src, mask=eroded)
            #     sub = eroded - eroded_3

            #     roi[(sub>0)*(roi<80)] = 0
            #     eroded[(sub>0)*(roi<80)] = 0

            #     # background = cv2.bitwise_not(eroded)            
            #     # cv2.imwrite(region_file, roi)
            #     # cv2.imwrite(mask_file, eroded)
            #     # cv2.imshow('region', roi+background)
            #     # cv2.waitKey(0)

            #     #--------------------defect detection 
            #     # cv2.imshow('roi', roi) 
            #     # cv2.imshow('eroded', eroded)
            #     # cv2.waitKey(0)


            #     start = time.time()
            #     defect_mask, defect_rgb = pyGTemplate.TempGenAndDetection(ParaName, Wh, Wl, roi, eroded)
            #     end = time.time()
            #     print('Detection cost:{:.4f}'.format(end - start))

            #     result_file = '../Results/defect_rgb_{:02d}'.format(i) + '.png'
            #     mask_file = '../Results/defect_mask_{:02d}'.format(i) + '.png'

            #     # cv2.imwrite(result_file, defect_rgb)
            #     # cv2.imwrite(mask_file, defect_mask)
            #     cv2.imshow('mask', defect_mask)
            #     cv2.waitKey(0)

            #  print('Totally time cost:{:.3f}'.format(time.time() - st)) 
            Precise_mean = np.mean(Precise_mean)
            Recall_mean = np.mean(Recall_mean)
            F1_mean = np.mean(F1_mean)

            print('Mean Precise : {:.4f}  Mean Recall : {:.4f}  Mean F1 : {:.4f}'.format(Precise_mean, Recall_mean, F1_mean))

            Precise_evaluate.append(Precise_mean)
            Recall_evaluate.append(Recall_mean)
            F1_evaluate.append(F1_mean)
        
        Precise = np.max(Precise_mean)
        Recall = np.max(Recall_mean)
        F1 = np.max(F1_mean)

    Precise_evaluate = np.array(Precise)
    Recall_evaluate = np.array(Recall)
    F1_evaluate = np.array(F1)
    data = np.vstack((Precise_evaluate, Recall_evaluate, F1_evaluate))
    mean_data = np.mean(data,1)

    sqi_mean = np.mean(sqi_time)
    ROI_mean = np.mean(ROI_time)
    detection_mean = np.mean(detection_time)
    total_mean = np.mean(total_time)

    if(evaluate_flag):
        mat = 'Wl_evaluate_data.mat'
        scio.savemat(mat, {'data': data})
    else:
        with open('result.txt', 'w') as f:
            f.write('Mean Precise : {:.4f}  Mean Recall : {:.4f}  Mean F1 : {:.4f} \n'.format(mean_data[0], mean_data[1], mean_data[2]))
            f.write('Avg SQI time_cost: {:.4f}   Avg RoI_extract time_cost: {:.4f}  Avg detection time_cost: {:.4f} \n'.format(sqi_mean, ROI_mean, detection_mean))
            f.write('Total time cost: {:.4f}'.format(total_mean))

if __name__ == "__main__":

    Template_method()
    
