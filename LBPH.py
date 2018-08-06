"""
CS 542: Machine Learning
Final Project: Face Detection and Recongnition
Part 1: Image Processing to Pattern through LBPH Algorithm.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class Image:
    def __init__(self):
        self.IMG_LEN = 0
        self.IMG_WID = 0
        self.image = []
        self.pattern = []
        self.img_lbp = []

    def _img_show(self):
        plt.imshow(self.img_lbp, cmap=plt.get_cmap('gray'))
        plt.show()

    def _img_pre(self):
        img_1d = []
        img_lbp_1d = []
        img_init_ind = np.array([-self.IMG_WID-1, -self.IMG_WID, -self.IMG_WID+1, -1,
                                 1, self.IMG_WID-1, self.IMG_WID, self.IMG_WID+1])

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
        self.img_lbp = np.reshape(img_lbp_1d, (int(len(img_lbp_1d) / self.IMG_WID), self.IMG_WID))
        # self._img_show()

    def _hist_pattern(self, k=4):
        sep_len = (self.IMG_LEN-2)//k
        sep_wid = self.IMG_WID//k
        hist = np.zeros(256*(k**2))

        for i in range(sep_len*k):
            for j in range(sep_wid*k):
                index = (j // sep_wid) + (i // sep_len) * k
                index = index * 256 + int(self.img_lbp[i][j])
                hist[index] = hist[index] + 1
        self.pattern = hist

    def _normalize(self):
        self.pattern = np.array(self.pattern)
        sum_item = np.sum(self.pattern)
        self.pattern = self.pattern / sum_item

    def pattern_return(self, img_name):
        # parameter initial
        self.image = mpimg.imread(img_name)
        self.image = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140])
        self.IMG_LEN = len(self.image)
        self.IMG_WID = len(self.image[0])
        self._img_pre()
        self._hist_pattern()
        self._normalize()
        return self.pattern
