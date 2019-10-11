
import numpy as np
from numba import jit
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from scipy.optimize import curve_fit
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

temp_num = 15

def gaussian(x, *param):  
    return param[0]*np.exp(-np.power(x-param[3],2.)/np.power(param[6], 2.))\
        +param[1]*np.exp(-np.power(x-param[4],2.)/np.power(param[7],2.))\
        +param[2]*np.exp(-np.power(x-param[5],2.)/np.power(param[8],2.))
    # +a2*np.exp(-np.square((x-b2)/c2))\
    #     +a3*np.exp(-np.square((x-b3)/c3))

def Chara_func (temp_num):
    bg_sum = np.array([])
    for i in range(1,temp_num):
        img_bg = cv2.imread('../roi/region_{:02d}.png'.format(i))
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
        pix_vect = img_bg[img_bg > 0]
        bg_sum = np.hstack((bg_sum, pix_vect))
    
    values, edges = np.histogram(bg_sum, bins=256, density=False)
    # n, bins, patches = plt.hist(bg_sum, 256, normed=1, facecolor='blue', alpha=0.5)

    edges = edges[:-1] + 0.25
    nvalues = values[values != 0] 
    nedges = edges[values != 0] 

    pixel_sum = np.sum(values)

    # centerpoints = np.vstack((nedges, nvalues))
    # print(centerpoints.shape)
    # plt.bar(centerpoints[0,:], height=centerpoints[1,:]/pixel_sum, width=0.5)
    # plt.xlabel('Values')  
    # plt.ylabel('Probability')
    # plt.show()

    start = time.time()
    x = nedges
    y = nvalues / pixel_sum
    popt, pcov = curve_fit(gaussian, x, y, p0=[0.03,0.02,0.02,133,133,149,6,20,10])
    end = time.time()
    print('Time Cost:{:.5f}'.format(end-start))
    # plt.plot(x,y,'b+:',label='data')  
    # plt.plot(x,gaussian(x,*popt),'ro:',label='fit')  
    # plt.legend()  
    # plt.show()

    alpha = popt[0:3] * 1.772453 * popt[6:]
    mu = popt[3:6]
    sigma = popt[6:] / 1.414

    para = np.vstack((alpha, mu, sigma))
    np.save('parameter.npy', para)

def ExpandImg(src_img, mask_img):
    [m,n] = src_img.shape
    img_expanded = np.zeros(src_img.shape, np.uint8)
    img_map = np.zeros(src_img.shape, np.uint8)

    for i in range(n):
        temp_ve = np.zeros(m,np.uint8)
        temp_vm = np.zeros(m,np.uint8)
        column_src = src_img[:,i]
        column_mask = mask_img[:,i]
        vect_pos = np.arange(0,256)
        vect = column_src[column_mask > 0]
        vect_pos = vect_pos[column_mask > 0]

        if(len(vect)):
            vect_sorted = np.sort(vect, kind='mergesort')
            # index = np.argsort(vect)
            pos = vect_pos[np.argsort(vect)]
            pad = np.floor(256/len(vect)).astype(np.uint8)
            for j in range(len(vect)-1):
                temp_ve[j*pad:(j+1)*pad] = vect_sorted[j]
                temp_vm[j*pad] = pos[j]
            
            temp_ve[pad*(len(vect)-1):] = vect_sorted[-1]
            temp_vm[pad*(len(vect)-1)] = pos[-1]

        img_expanded[:,i] = temp_ve
        img_map[:,i] = temp_vm
        # print(img_map)

            
    
    # cv2.imshow('expanded',img_expanded)
    # cv2.waitKey(0)
    return img_expanded, img_map

def ExpandImg_simple(src_img, mask_img):
    [m,n] = src_img.shape
    img_expanded = np.zeros(src_img.shape, np.uint8)
    img_map = np.zeros(src_img.shape, np.uint8)

    for i in range(n):
        temp_ve = np.zeros(m,np.uint8)
        temp_vm = np.zeros(m,np.uint8)
        column_src = src_img[:,i]
        column_mask = mask_img[:,i]
        vect_pos = np.arange(0,256)
        vect = column_src[column_mask > 0]
        vect_pos = vect_pos[column_mask > 0]

        if(len(vect)):
            vect_sorted = np.sort(vect, kind='mergesort')
            # index = np.argsort(vect)
            pos = vect_pos[np.argsort(vect)]
            # pad = np.floor(256/len(vect)).astype(np.uint8)
            # for j in range(len(vect)-1):
            #     temp_ve[j*pad:(j+1)*pad] = vect_sorted[j]
            #     temp_vm[j*pad] = pos[j]
            
            # temp_ve[pad*(len(vect)-1):] = vect_sorted[-1]
            # temp_vm[pad*(len(vect)-1)] = pos[-1]
            temp_ve[:len(vect_sorted)] = vect_sorted
            temp_vm[:len(pos)] = pos

        img_expanded[:,i] = temp_ve
        img_map[:,i] = temp_vm
        # print(img_map)

            
    
    # cv2.imshow('expanded',img_expanded)
    # cv2.imshow('map', img_map)
    # cv2.waitKey(0)
    return img_expanded, img_map


def CalculateXi_tri(para, Q):
    Xi = np.zeros(Q, np.uint8)
    if(Q == 0):
        return Xi
    alpha = para[0]
    mu = para[1]
    sigma = para[2]
    sum_ = 0
    i = 1
    for t in range(1,256):
        if(sum_ >= i/(Q+1)):
            Xi[i-1] = t
            i += 1
            while (sum_ >= (i/(Q+1))):
                Xi[i-1] = t
                i += 1
            if(i == Q+1):
                break
            sum_ = sum_+(alpha[0]*mlab.normpdf(t, mu[0], sigma[0])+alpha[1]*mlab.normpdf(t, mu[1], sigma[1])+alpha[2]*mlab.normpdf(t, mu[2], sigma[2]))/np.sum(alpha)

        else:
            sum_ = sum_+(alpha[0]*mlab.normpdf(t, mu[0], sigma[0])+alpha[1]*mlab.normpdf(t, mu[1], sigma[1])+alpha[2]*mlab.normpdf(t, mu[2], sigma[2]))/np.sum(alpha)
    return Xi

def sub_operation(para, initial_template, img_expanded):
    [m,n] = img_expanded.shape
    # alpha = para[0]
    # mu = para[1]
    # sigma = para[2]

    L = 0
    R = 256
    gap_pix = 1
    vl = initial_template[gap_pix-1, 0]
    vh = initial_template[m-gap_pix, 0]

    for k in range(n-1):
        if(img_expanded[m-1, k]==0 and img_expanded[m-1, k+1]>0):
            L = k+1
        if(img_expanded[m-1, k]>0 and img_expanded[m-1, k+1]==0):
            R = k+1

    sub_template = np.zeros([m,n], np.uint8)
    sub_template[:,L:R] = initial_template[:,L:R]
    temp_mask = np.zeros([m,n], np.uint8)
    Qj = np.zeros(n, np.uint8)
    C = sub_template
    # Xjv = np.zeros(m, np.uint8)  
    # ------------------------------------
    temp_mask_1 = (img_expanded>=vl)+0
    temp_mask_2 = (img_expanded<=vh)+0
    temp_mask = temp_mask_1*temp_mask_2
    Qj = np.sum(temp_mask, axis=0)
    # ------------------------------------
    # print(Qj)

    st = time.time()
    for i in range(n):
        if(Qj[i] == 0):
            continue
        index = np.arange(m)
        a = time.clock()
        Xjv = CalculateXi_tri(para, Qj[i])
        # print('Xi_cost:{:.4f}'.format(time.clock() - a))
        index_ = index[temp_mask[:,i]==1]
        C[index_[0]:index_[-1]+1,i] = Xjv
    # print("Guidance_template_Cost:{:.5f}".format(time.time()-st))
    D = C.astype(np.float) - img_expanded.astype(np.float)

    return D

def sub_operation_simple(para, initial_template, img_expanded, img_map):
    [m,n] = img_expanded.shape
    # L = 0
    # R = 256
    gap_pix = 1
    vl = initial_template[gap_pix-1, 0]
    vh = initial_template[m-gap_pix, 0]
    print(vl,vh)
    # for k in range(n-1):
    #     if(img_expanded[m-1, k]==0 and img_expanded[m-1, k+1]>0):
    #         L = k+1
    #     if(img_expanded[m-1, k]>0 and img_expanded[m-1, k+1]==0):
    #         R = k+1

    # sub_template = np.zeros([m,n], np.uint8)
    # sub_template[:,L:R] = initial_template[:,L:R]
    # temp_mask = np.zeros([m,n], np.uint8)
    Qj = np.zeros(n, np.uint8)
    C = np.zeros([m,n], np.uint8)
    temp_mask = (img_map>0)+0
    num = np.sum(temp_mask, axis=0)
    for i in range(n):
        index = np.linspace(0,255,num=num[i],endpoint=True)
        index = index.astype(np.uint8)
        temp = initial_template[index, i]
        C[:len(temp),i] = temp

    # cv2.imshow('C',C)
    # cv2.imshow('expanded', img_expanded)
    # cv2.waitKey(0)
    # Xjv = np.zeros(m, np.uint8)  
    # ------------------------------------
    temp_mask_1 = (img_expanded>=vl)+0
    temp_mask_2 = (img_expanded<=vh)+0
    temp_mask_ = temp_mask_1*temp_mask_2

    # cv2.imshow('mask', temp_mask_.astype(np.uint8)*255)
    # cv2.waitKey(0)
    Qj = np.sum(temp_mask_, axis=0)
    # ------------------------------------
    # print(Qj)

    st = time.time()
    for i in range(n):
        if(Qj[i] == 0):
            continue
        index = np.arange(m)
        idx = np.linspace(0,255,num=Qj[i],endpoint=True)
        idx = idx.astype(np.uint8)
        Xjv = initial_template[idx, i]
        # Xjv = CalculateXi_tri(para, Qj[i])
        # print('Xi_cost:{:.4f}'.format(time.clock() - a))
        index_ = index[temp_mask_[:,i]==1]
        # print(C[index_[0]:index_[-1]+1,i], Xjv)
        C[index_[0]:index_[-1]+1,i] = Xjv
    # print("Guidance_template_Cost:{:.5f}".format(time.time()-st))

    
    D = C.astype(np.float) - img_expanded.astype(np.float)

    return D

def TempGenAndDetection(ParaName, Wh, Wl, roi_src_img, roi_mask_img):

    para = np.load(ParaName)
    # THl = W*np.log(np.mean(para[2]))/np.log(np.mean(para[1]))
    # THl = np.max(THl, 0)
    # THh = THl + H
    # print(THh, THl)
    THh = Wh*(np.sum(np.dot(para[0],para[1])))/np.sum(para[0])
    THl = -Wl*(255 - np.sum(np.dot(para[0],para[1])))/np.sum(para[0])
    # print(THh, THl)

    st = time.time()
    [m,n] = roi_mask_img.shape
    img_expanded, img_map = ExpandImg_simple(roi_src_img, roi_mask_img)
    # img_expanded, img_map = ExpandImg(roi_src_img, roi_mask_img)
    Xi = CalculateXi_tri(para, m)
    initial_template = np.zeros([m,n], np.uint8)
    for i in range(n):
        initial_template[:,i] = Xi
    ed = time.time()
    # print("Initial_template_Cost:{:.5f}".format(ed-st))


    D = sub_operation_simple(para, initial_template, img_expanded, img_map)
    # D = sub_operation(para, initial_template, img_expanded)
    R = np.zeros([m,n], np.float)
    defect_mask = np.zeros([m,n], np.uint8)
    defect_rgb = cv2.merge([roi_src_img, roi_src_img, roi_src_img])
    ed_2 = time.time()
    print("Substraction_Cost:{:.5f}".format(ed_2 - ed))

    for i in range(n):
        for j in range(m):
            if(img_map[j,i]):
                R[img_map[j,i], i] = D[j,i]
                if(R[img_map[j,i], i] > THh or R[img_map[j,i], i] < THl):
                    defect_mask[img_map[j,i], i] = 255
                    # defect_rgb[img_map[j,i], i, :] = np.array([0, 0, 255]) 

    for j in range(1,n-1):
        for i in range(1,m-1):
            if(defect_mask[i,j]):
                adjecent = float(defect_mask[i-1,j-1])+float(defect_mask[i-1,j])+float(defect_mask[i-1,j+1])+float(defect_mask[i,j-1]+defect_mask[i,j+1])+float(defect_mask[i+1,j-1])+float(defect_mask[i+1,j])+float(defect_mask[i+1,j+1])
                if(adjecent == 0):
                    defect_mask[i,j] = 0
                else:
                    defect_rgb[i,j,:] = np.array([0,0,255])
    print("Defect_loc_Cost:{:.5f}".format(time.time() - ed_2))
    
    # defect_mask = cv2.medianBlur(defect_mask, 3)
    # defect_rgb[defect_mask > 0] = np.array([0,0,255])

    # defect_mask[R>THl] = 255
    # defect_mask[R<-THh] = 255

    # cv2.imshow('mask', defect_mask)
    # cv2.waitKey(0)
    # cv2.imshow('rgb', defect_rgb)
    # cv2.waitKey(0)
    return defect_mask, defect_rgb


if __name__ == "__main__":
    # Chara_func(temp_num)
    ParaName = 'parameter.npy'

    test_num = 0
    Wh = 0.35
    Wl = 0.7
    roi_src_img = cv2.imread('../roi/region_{:02d}.png'.format(test_num))
    roi_mask_img = cv2.imread('../Template/bin_mask/region_{:02d}.png'.format(test_num))
    roi_src_img = cv2.cvtColor(roi_src_img, cv2.COLOR_BGR2GRAY)
    roi_mask_img = cv2.cvtColor(roi_mask_img, cv2.COLOR_BGR2GRAY)

    start = time.time()
    defect_mask, defect_rgb = TempGenAndDetection(ParaName, Wh, Wl, roi_src_img, roi_mask_img)
    # expanded, map = ExpandImg_simple(roi_src_img, roi_mask_img)
    end = time.time()
    print('Total cost:{:.4f}'.format(end - start))