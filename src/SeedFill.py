# -*- coding: utf-8 -*-

import cv2
import numpy as np
import collections


def SeedFill(point, src, label):
    queue = collections.deque()
    queue.append(point)
    
    while len(queue) != 0:
        pt = queue.popleft()
        label[pt[0],pt[1]] = 255
        if src[pt[0]-1,pt[1]] == src[pt[0],pt[1]] and label[pt[0]-1,pt[1]] == 0 and queue.count([pt[0]-1,pt[1]]) == 0:
            queue.append([pt[0]-1,pt[1]])
        if src[pt[0],pt[1]-1] == src[pt[0],pt[1]] and label[pt[0],pt[1]-1] == 0 and queue.count([pt[0],pt[1]-1]) == 0:
            queue.append([pt[0],pt[1]-1])
        if src[pt[0]+1,pt[1]] == src[pt[0],pt[1]] and label[pt[0]+1,pt[1]] == 0 and queue.count([pt[0]+1,pt[1]]) == 0:
            queue.append([pt[0]+1,pt[1]])
        if src[pt[0],pt[1]+1] == src[pt[0],pt[1]] and label[pt[0],pt[1]+1] == 0 and queue.count([pt[0],pt[1]+1]) == 0:
            queue.append([pt[0],pt[1]+1])
        # print(queue)

if __name__ == "__main__":
    img = cv2.imread('./bin_contour/sobel_mask_10.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = img_gray.shape
    label = np.zeros((size[0],size[1]), dtype= np.uint8)
    endflag = 0
    for i in range(1,size[0]-1):
        for j in range(1,size[1]-1):
            if img_gray[i,j] == 0:
                if img_gray[i-1,j] == 255 and img_gray[i,j+1] == 255:
                    start_point = [i,j]
                    endflag = 1
                    break
            if endflag == 1:
                break

    SeedFill(start_point, img_gray, label)        
    cv2.imshow("gray", img_gray)
    cv2.waitKey(0)
    cv2.imshow("label", label)
    cv2.waitKey(0)


