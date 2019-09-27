
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from scipy.optimize import curve_fit

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
        # cv2.imshow('img',img_bg)
        # cv2.waitKey(0)
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
        pix_vect = img_bg[img_bg > 0]
        bg_sum = np.hstack((bg_sum, pix_vect))
    
    values, edges = np.histogram(bg_sum, bins=256, density=False)
    # n, bins, patches = plt.hist(bg_sum, 256, normed=1, facecolor='blue', alpha=0.5)

    edges = edges[:-1] + 0.25
    nvalues = values[values != 0] 
    nedges = edges[values != 0] 

    pixel_sum = np.sum(values)

    print(nvalues.shape, nedges.shape)
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
    print('Cost:{:.5f}'.format(end-start))
    plt.plot(x,y,'b+:',label='data')  
    plt.plot(x,gaussian(x,*popt),'ro:',label='fit')  
    plt.legend()  
    plt.show()

    alpha = popt[0:3] * 1.772453 * popt[6:]
    mu = popt[3:6]
    sigma = popt[6:] / 1.414

    para = np.vstack((alpha, mu, sigma))
    np.save('parameter.npy', para)

if __name__ == "__main__":
    Chara_func(temp_num)