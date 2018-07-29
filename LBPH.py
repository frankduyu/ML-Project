# CS 542: Machine Learning
# Final Project: Face Detection and Recongnition
# Part 1: Image Processing to Pattern

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class Image:
    def __init__(self):
        self.IMG_LEN = 0
        self.IMG_WID = 0
        self.image = []
        self.img_lbp = []

    def _img_show(self):
        plt.imshow(self.img_lbp, cmap=plt.get_cmap('gray'))
        plt.show()

    def _img_pre(self):
        img_1d = []
        img_lbp_1d = []
        img_init_ind = np.array([-self.IMG_WID-1, -self.IMG_WID, -self.IMG_WID+1, -1, 1, self.IMG_WID-1, self.IMG_WID, self.IMG_WID+1])
        # image preprocess
        for row in self.image:
            row_list = list(row)
            img_1d += row_list
        for item_ind in range(self.IMG_WID-1, len(img_1d)-self.IMG_WID-1):
            bin_num = 0
            for ind in img_init_ind:
                if img_1d[item_ind+ind] > img_1d[item_ind]:
                    bin_num = bin_num * 2 + 1
                else:
                    bin_num = bin_num * 2
            img_lbp_1d.append(bin_num)
        # reshape image
        self.img_lbp = np.reshape(img_lbp_1d, (int(len(img_lbp_1d)/self.IMG_WID), self.IMG_WID))
        # show LBP
        self._img_show()

    def extract_histogram(self, K = 3):
        step_len = int(self.IMG_LEN // K)
        step_wid = int(self.IMG_WID // K)
        hist = []
        # Hist Initialization
        for i in range(K**2):
            for j in range(256):
                hist.append(0)
        print(len(hist))
        for i in range(step_len * K):
            for j in range(step_wid * K):
                index = (j // step_wid) + (i // step_len) * K
                # print(index, self.img_lbp[i][j], i, j)
                index = index * 256 + int(self.img_lbp[i][j])
                hist[index] = hist[index] + 1
        return hist


    def pattern_return(self, img_name):
        # parameter initial
        self.image = mpimg.imread(img_name)
        self.image = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140])
        self.IMG_LEN = len(self.image)
        self.IMG_WID = len(self.image[0])
        self._img_pre()


if __name__ == "__main__":
    img = Image()
    img.pattern_return('test.png')
    print(img.extract_histogram())
