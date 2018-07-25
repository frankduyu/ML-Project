# CS 542: Machine Learning
# Final Project: Face Detection and Recongnition
# Part 1: LBPH

import matplotlib.pyplot as plt  # plt - for showing image
import matplotlib.image as mping  # mpimg - for reading image
import numpy

class Image():
    def __init__(self, imgName):
        self.name = imgName
        self.image = mping.imread(imgName)

    def shownImage(self, axis = False, gray = False):
        plt.imshow(self.image)
        if axis:  # Showing with axis or not
            plt.axis('on')
        else:
            plt.axis('off')
        if gray:
            plt.imshow(self.image, cmap =plt.get_cmap('gray'))
        plt.show()

    def rgbToGray(self): # Transform the RBG picture into Gray Picture
        self.image = numpy.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140])

class LBPH(Image):
    def __init__(self, imgName):
        super(LBPH, self).__init__(imgName)
        self.N = len(self.image)
        self.M = len(self.image[0])
        pass

    def doLBP(self):
        LBPimg = []
        for i in range(1, self.N-1):
            LBPimg.append([])
            for j in range(1, self.M-1):
                slideWimdowValue = 0
                for wrow in range(1, -2, -1):
                    for wcol in range(1, -2, -1):
                        if wrow != 0 and wcol != 0:
                            if self.image[i+wrow][j+wcol] > self.image[i][j]:
                                slideWimdowValue = slideWimdowValue * 2 + 1
                            else:
                                slideWimdowValue = slideWimdowValue * 2
                LBPimg[i-1].append(slideWimdowValue)
        return LBPimg



if __name__ == "__main__":
    img = LBPH("3-160P4111639.jpg")
    #img.shownImage()
    img.rgbToGray()
    #img.shownImage(gray = True) # Shown gray
    LBP = img.doLBP()
    plt.imshow(LBP, cmap = plt.get_cmap('gray'))
    plt.show()




