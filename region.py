import cv2
import numpy as np

if __name__ == "__main__":

    total_num = 28
    for pic_num in range(1, total_num):
        mask_file = './bin_contour/sobel_mask_' + str(pic_num) + '.png'
        src_file = './pic/' + str(pic_num) + '.png'
        region_file = './roi/region_' + str(pic_num) + '.png'

        img = cv2.imread(mask_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        findcontourimg = img

        ret, binary = cv2.threshold(findcontourimg,127,255,cv2.THRESH_BINARY)
        # print(binary.shape)
        contours,hierarchy= cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        imgcontour = img.copy()
        imgcontour[:] = 0
        # 需要搞一个list给cv2.drawContours()才行！！！！！

        temp = np.zeros((256,256),np.uint8)
        #画出轮廓：temp是黑色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度 cv2.FILLED 表示对轮廓内部区域进行填充

        c_max = []
        c_min = []
        lenContours = len(contours)
        if lenContours == 2:
            fill_flage = False
        else:
            fill_flage = True

        hierarchy0 = hierarchy[0]
        for i in range(lenContours):
            hierarchyI = hierarchy0[i]
            if  hierarchyI[3] == -1: #hierarchyI[0] == -1 and hierarchyI[1] == -1 and
                cnt = contours[i]
                c_max.append(cnt)
            if  hierarchyI[2] == -1:#hierarchyI[0] == -1 and hierarchyI[1] == -1 and
                cnt = contours[i]
                c_min.append(cnt)


        cv2.drawContours(temp, c_max, -1,  (255), cv2.FILLED)
        if fill_flage:
            cv2.drawContours(temp, c_min, -1,  (0), cv2.FILLED)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        eroded = cv2.erode(temp,kernel)
        # cv2.imshow('region',eroded)
        # cv2.waitKey(0)

        src_img = cv2.imread(src_file)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        region = cv2.bitwise_and(src_img, src_img, mask=eroded)
        cv2.imshow('region', region)
        cv2.waitKey(0)

        cv2.imwrite(region_file,region)
