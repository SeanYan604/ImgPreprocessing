
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  

temp_num = 15

def Chara_func (temp_num):
    bg_sum = np.array([])
    for i in range(1,temp_num):
        img_bg = cv2.imread('../roi/region_{:02d}.png'.format(i))
        # cv2.imshow('img',img_bg)
        # cv2.waitKey(0)
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
        pix_vect = img_bg[img_bg > 0]
        bg_sum = np.hstack((bg_sum, pix_vect))
    
    values, edges = np.histogram(bg_sum, bins=256, density=True)
    # n, bins, patches = plt.hist(bg_sum, 256, normed=1, facecolor='blue', alpha=0.5)

    edges = edges[:-1] + 0.5
    nvalues = values[values != 0] 
    nedges = edges[values != 0] 

    print(nvalues.shape, nedges.shape)
    centerpoints = np.vstack((nedges, nvalues))
    # print(centerpoints.shape)
    plt.bar(centerpoints[0,:], height=centerpoints[1,:], width=0.5)
    plt.xlabel('Values')  
    plt.ylabel('Probability')
    plt.show()

if __name__ == "__main__":
    Chara_func(temp_num)