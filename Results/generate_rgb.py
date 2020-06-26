import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def transition():
    ori_file = Path('../../Dataset/ori_gray')
    rgb_file = Path('./rgb')
    mask_file = Path('./mask')

    total_num = 297
    for i in tqdm(range(total_num)):
        file_name = str(ori_file/f'{i+1:03d}.png')
        save_name = str(rgb_file/f'{i+1:03d}.png')
        mask_name = str(mask_file/f'{i+1:03d}.png')

        img = cv2.imread(file_name)
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        img[mask > 0] = np.array([0,0,255])
        cv2.imwrite(save_name, img)
        # cv2.imshow('rgb', img)
        # cv2.waitKey(0)

if __name__ == '__main__':
    transition()