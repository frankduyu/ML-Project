# CS 542: Machine Learning
# Final Project: Face Detection and Recongnition
# Part 1: Image Processing to Pattern

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import stats


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
        self.img_lbp = np.reshape(img_lbp_1d, (int(len(img_lbp_1d) / self.IMG_WID), self.IMG_WID))
        # self._img_show()

    def _hist_pattern(self, k = 3):
        sep_len = (self.IMG_LEN-2)//k
        sep_wid = self.IMG_WID//k
        hist = np.zeros(256*(k**2))
        for i in range(sep_len*k):
            for j in range(sep_wid*k):
                index = (j // sep_wid) + (i // sep_len) * k
                index = index * 256 + int(self.img_lbp[i][j])
                hist[index] = hist[index] + 1
        # for ind in range(0,len(hist),256):
        #     hist[ind] = 0
        self.pattern = hist

    def _normalize(self):
        max_item = max(self.pattern)
        for ind in range(len(self.pattern)):
            self.pattern[ind] = self.pattern[ind]/max_item

    def pattern_return(self, img_name):
        # parameter initial
        self.image = mpimg.imread(img_name)
        self.image = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140])
        self.IMG_LEN = len(self.image)
        self.IMG_WID = len(self.image[0])
        self._img_pre()
        self._hist_pattern()
        # self._normalize()
        return self.pattern


if __name__ == "__main__":
    img = Image()
    img_name1 = 'obama_4.png'
    img_name2 = 'dog_1.png'
    pattern1 = img.pattern_return('face_dataset/'+img_name1)
    pattern2 = img.pattern_return('face_dataset/'+img_name2)
    # p_coef
    p_coef = np.cov(pattern1,pattern2)[0][1]/(np.std(pattern1)*np.std(pattern2))
    # t_test
    t_test = stats.ttest_ind(pattern1,pattern2)
    # distance
    diff = np.array(pattern1)-np.array(pattern2)
    dist = sum(diff**2)
    # # chi-square
    # chi_sq = stats.chisquare(pattern1)
    # output parameters
    print(img_name1,img_name2)
    print('p_coef is: ', p_coef)
    print('t_test score is: ', t_test[0],t_test[1])
    # print('distance score is: ',dist)
    print('distance score is: ', dist/100000000)

    # # save patterns
    # with open('training_database_pattern','a') as f:
    #     f.write(img_name+'\t')
    #     for item in pattern:
    #         f.write(str(int(item))+' ')
    #     f.write('\n')




