 
import cv2
import numpy as np
import math
import time
import testAAE
import region

def quantizeAngle(angle):
    if angle >= 0:
        if angle >= 90:
            if angle >= 45:
                quantized = 2
            else:
                quantized = 1
        elif angle >= 135:
            quantized = 4
        else:
            quantized = 8
    elif angle <= -90:
        if angle <= -135:
            quantized = 16
        else:
            quantized = 32
    elif angle <= -45:
        quantized = 64
    else:
        quantized = 128
    return int(quantized)

def angleFilter(mask, quantized, quantized_flag = False):
    temp_angle = mask*quantized
    kernal = 9
    m,n = mask.shape
    hist = {}
    hist_sorted = []
    strong_angle = np.zeros(mask.shape, np.uint8)
    contour = np.zeros(mask.shape, np.uint8)
    score_map = np.array([[5,3,1,0,0,0,1,3],
                 [3,5,3,1,0,0,0,1],
                 [1,3,5,3,1,0,0,0],
                 [0,1,3,5,3,1,0,0],
                 [0,0,1,3,5,3,1,0],
                 [0,0,0,1,3,5,3,1],
                 [1,0,0,0,1,3,5,3],
                 [3,1,0,0,0,1,3,5]])
    
    bias = math.floor(kernal /2)
    qt_angle = np.array([1,2,4,8,16,32,64,128])
    for i in range(m):
        for j in range(n):
            if mask[i,j] > 0:
                if i-bias < 0: 
                    h_t=0
                else: 
                    h_t = i-bias
                if i+bias > m-1: 
                    h_b=m-1
                else: 
                    h_b = i+bias
                if j-bias < 0: 
                    w_l=0
                else: 
                    w_l=j-bias
                if j+bias > m-1:
                    w_r=m-1
                else: 
                    w_r=j+bias
                temp = temp_angle[h_t:h_b+1,w_l:w_r+1]
                a,b = temp.shape
                temp = temp.flat[:]
                if not quantized_flag:
                    for k in range(a*b):
                        if temp[k] > 0:
                            temp[k] = quantizeAngle(temp[k])
                    temp = temp.astype(np.uint8)

                temp_ = temp[temp.nonzero()]
                bcounts = np.bincount(temp_)
                strong_temp = np.zeros(a*b)
                score_temp = np.zeros(a*b)
                hist.clear
                hist_sorted.clear
                hist = dict(zip(np.unique(temp_),bcounts[bcounts.nonzero()]))
                hist_sorted = sorted(hist.items(), key=lambda x: x[1], reverse=True) 
                max_count = hist_sorted[0][1]
                strong_angle[i,j] = hist_sorted[0][0]
                count = 0
                for c in range(a*b):
                    if temp[c] > 0:
                        score_temp[c] = score_map[int(math.log2(quantizeAngle(temp_angle[i,j]))),int(math.log2(temp[c]))]
                        strong_temp[c] = score_map[int(math.log2(strong_angle[i,j])),int(math.log2(temp[c]))]
                        count+=1
                pix_score = np.sum(score_temp)/count
                strong_score = np.sum(strong_temp)/count
                if max_count > 5 and (pix_score > 2 or strong_score > 2):
                    contour[i,j] = 1
    return contour

def preprocessing():
    total_num = 28
    sample_id = 0  
    threshold = 160
    exposure = 6
    write_flag = False

    sobel_mask_vect = []
    src_vect = []
    sobel_x =np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype=np.float32)
    sobel_y =np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]], dtype=np.float32)
    new_img = np.zeros((256,256), np.uint8)
    for pic_num in range(1, total_num):
        if write_flag:
            src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.jpg'
            output_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.png'

            IN_src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '_IN/' + 'TT' + '/' + '{:02d}'.format(pic_num) + '.png'
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

            IN_img = cv2.imread(IN_src_file)
            IN_img = cv2.cvtColor(IN_img, cv2.COLOR_BGR2GRAY)
            src_vect.append(IN_img)
        else:
            src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.png'\
            IN_src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '_IN/' + 'TT' + '/' + '{:02d}'.format(pic_num) + '.png'
            new_img = cv2.imread(src_file)
            new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
            IN_img = cv2.imread(IN_src_file)
            IN_img = cv2.cvtColor(IN_img, cv2.COLOR_BGR2GRAY)
            src_vect.append(IN_img)


        sobel_mag = np.zeros(new_img.shape, np.float)
        sobel_angle = np.zeros(new_img.shape, np.float)
        quantized_angle = np.zeros(new_img.shape, np.uint8)
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
        sobel_angle = np.arctan2(sobely,sobelx) * 180 /math.pi
        sobel_mag = cv2.convertScaleAbs(sobel_mag)
        _, sobel_mask = cv2.threshold(sobel_mag, threshold, 255, 0)

        # contour = angleFilter(sobel_mask, sobel_angle)
        # contour = cv2.blur(contour, (3,3))
        # sobel_mask = cv2.blur(sobel_mask, (3,3))
        # contour_vect.append(contour)

        # cv2.imshow('sobel', sobel_mask)
        # cv2.waitKey(0)
        sobel_mask_vect.append(sobel_mask)

    return sobel_mask_vect, src_vect

if __name__ == "__main__":

    time_start = time.time()
    sobel_mask_vect, src_vect = preprocessing()
    time_end = time.time()
    print('Proprecessing time cost:{:.3f}'.format(time_end - time_start))
    # for sobel_mask in sobel_mask_vect:
    #     # cv2.imshow("sobel",255*sobel_mask.astype(np.uint8))
    #     cv2.imshow("sobel",sobel_mask)
    #     # cv2.imshow("extend", 255*contour.astype(np.uint8))
    #     # cv2.imshow("sub",255*(sobel_mask - contour).astype(np.uint8))
    #     cv2.waitKey(0)
    


    output_img_vect = testAAE.AEprocessing(sobel_mask_vect)
    print('AAE time cost:{:.3f}'.format(time.time() - time_end))

    for i, singleimg in enumerate(output_img_vect):
        # singleimg = np.squeeze(singleimg, axis=(2,))
        singleimg = singleimg.astype(np.uint8)
        src = src_vect[i]
        # cv2.imshow('src',src)
        # cv2.waitKey(0)
        region_file = '../roi/region_{:02d}'.format(i) + '.png'
        mask = region.regionGenerate(singleimg)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        eroded = cv2.erode(mask,kernel)
        eroded_2 = cv2.erode(eroded,kernel)
        eroded_3 = cv2.erode(eroded_2,kernel)
        roi = cv2.bitwise_and(src, src, mask=eroded)
        sub = eroded - eroded_3
        m,n = sub.shape
        for row in range(m):
            for col in range(n):
                if sub[row, col] and roi[row, col] < 80:
                    roi[row,col] = 0
                    eroded[row, col] = 0


        background = cv2.bitwise_not(eroded)            
        cv2.imwrite(region_file, roi)
        cv2.imshow('region', roi+background)
        cv2.waitKey(0)

    print('Totally time cost:{:.3f}'.format(time.time() - time_start))    

        
    
        