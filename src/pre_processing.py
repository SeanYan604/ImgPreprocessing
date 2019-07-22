 
import cv2
import numpy as np
import math

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
    return quantized

def angleFilter(mask, quantized):
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
                        score_temp[c] = score_map[int(math.log2(temp_angle[i,j])),int(math.log2(temp[c]))]
                        strong_temp[c] = score_map[int(math.log2(strong_angle[i,j])),int(math.log2(temp[c]))]
                        count+=1
                pix_score = np.sum(score_temp)/count
                strong_score = np.sum(strong_temp)/count
                if max_count > 5 and (pix_score > 2 or strong_score > 2):
                    contour[i,j] = 1
    return contour



if __name__ == "__main__":

    total_num = 28
    sample_id = 0
    exposure = 6
    threshold = 100
    write_flag = True

    sobel_x =np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
    sobel_y =np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])
    for pic_num in range(1, total_num):
        src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.jpg'
        output_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.png'
        region_file = './roi/region_' + str(pic_num) + '.png'

        img = cv2.imread(src_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m,n = img.shape
        img = img[0:n]

        new_img = np.zeros((256,256), np.uint8)
        new_img[3:253,3:253] = img
        if(write_flag):
            cv2.imwrite(output_file, new_img)

        sobel_mag = np.zeros(new_img.shape, np.float)
        sobel_angle = np.zeros(new_img.shape, np.float)
        quantized_angle = np.zeros(new_img.shape, np.uint8)
        sobel_mask = np.zeros(new_img.shape, np.uint8)

        # img_Guassian = cv2.GaussianBlur(new_img,(5,5),0)
        # img_Guassian.astype(np.uint8)
        # m,n = img_Guassian.shape
        m,n = new_img.shape
        for i in range(2,m-1):
            for j in range(2,n-1):
                Gx = np.sum(new_img[i-1:i+2, j-1:j+2] * sobel_x)
                Gy = np.sum(new_img[i-1:i+2, j-1:j+2] * sobel_y) 
                sobel_mag[i,j] = math.sqrt(math.pow(Gx,2) + math.pow(Gy,2))
                sobel_angle[i,j] = math.atan2(Gy, Gx) * 180 / math.pi
                quantized_angle[i,j] = quantizeAngle(sobel_angle[i,j])
                if sobel_mag[i,j] >= threshold:
                    sobel_mask[i,j] = 1

        # sobel_mask = cv2.medianBlur(sobel_mask, 5)
        contour = angleFilter(sobel_mask, quantized_angle)

        cv2.imshow("extend", 255*contour.astype(np.uint8))
        cv2.waitKey(0)
        