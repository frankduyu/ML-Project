"""
CS 542: Machine Learning
Final Project: Face Detection and Recongnition
Part 1: Image Processing to Pattern through LBPH Algorithm.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import pickle


class Image:
    def __init__(self):
        self.IMG_LEN = 0
        self.IMG_WID = 0

        self.image = []
        self.pattern = []
        self.img_lbp = []

        self.para_dict = pickle.load(open('face_detection_parameter.pickle', 'rb'))
        self.train_patterns = self.para_dict['train_data_pattern']
        self.dist_bar = self.para_dict['dist_bar']
        self.prob_bar = self.para_dict['proba_bar']

    def _img_pre(self):
        """
        Use LBP algorithm to preprocess image.
        """
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

    def _dist(self, patt1, patt2):
        """
        Calculate distance between two histogram patterns.
        """
        diff = np.array(patt1)-np.array(patt2)
        distance = sum(diff**2)
        return distance

    def _face_detect(self, pattern):
        """
        Judge whether image in slide window is a face.
        """
        test_dist = []
        for patt in self.train_patterns[:100]:
            distance = self._dist(pattern, patt)
            test_dist.append(distance)

        # calculate test data probability
        test_prob = sum(dist <= self.dist_bar for dist in test_dist) / len(test_dist)
        print(test_prob)

        if test_prob >= self.prob_bar:
            print("This is a face")
            return test_prob
        else:
            return 0

    def _slid_wind(self, interval=4, win_end=60):
        """
        Scan image using different slide window to find candidate faces.
        """
        img_lbp = self.img_lbp
        win_len = len(img_lbp)
        win_wid = len(img_lbp[0])
        win_side = min(win_wid, win_len)
        moving_step = win_side // interval
        crops = []
        proba = []

        # start slide window
        while win_side >= win_end:
            print('start')
            for i in range(0, int(win_len - win_side), moving_step):
                for j in range(0, int(win_wid - win_side), moving_step):
                    win_data = img_lbp[i: int(i + win_side), j: int(j + win_side)]
                    filter_patt = self._hist_pattern(win_data)
                    test_prob = self._face_detect(filter_patt)
                    if test_prob > self.prob_bar:
                        crops.append([i, j, win_side, win_side])
                        proba.append(test_prob)

            win_side = win_side * interval / (interval + 1)
            moving_step = int(win_side // interval)

        # select face candidate that has highest probability
        if crops != []:
            crops = crops[proba.index(max(proba))]

        return crops

    def _normalize(self, hist):
        """
        Normalize histogram pattern with dividing number of pixel.
        """
        hist = np.array(hist)
        sum_item = np.sum(hist)
        hist = hist / sum_item

        return hist

    def _hist_pattern(self, img_lbp, k=4):
        """
        Transfer LBP pattern into histogram. The last step of LBPH algorithm.
        """
        win_len = len(img_lbp)
        win_wid = len(img_lbp[0])
        sep_len = win_len//k
        sep_wid = win_wid//k
        hist = np.zeros(256*(k**2))

        for i in range(sep_len*k):
            for j in range(sep_wid*k):
                index = (j // sep_wid) + (i // sep_len) * k
                index = index * 256 + int(img_lbp[i][j])
                hist[index] = hist[index] + 1
        hist = self._normalize(hist)
        return hist

    def _draw_crops(self, crops):
        """
        Show image with face contained rectangle using pyplot package.
        """
        im = self.image
        fig, ax = plt.subplots(1)
        ax.imshow(im)

        # add face rectangle in image
        rect = patches.Rectangle((crops[0], crops[1]), crops[2], crops[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.show()

    def pattern_return(self, img_name):
        # parameter initial
        self.image = mpimg.imread(img_name)
        self.image = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140])
        self.IMG_LEN = len(self.image)
        self.IMG_WID = len(self.image[0])

        # preprocess face into histogram with LBP algorithm
        self._img_pre()

        # find face in image
        crops = self._slid_wind()

        # show image with rectangle
        if crops != []:
            self._draw_crops(crops)
        else:
            print('Oops, there is no face in image!')
