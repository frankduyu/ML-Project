import numpy as np
import pickle


class FaceDetect:
    def __init__(self, img):
        # parameter initialize
        self.para_dict = pickle.load(open('face_detection_parameter.pickle', 'rb'))
        self.train_patterns = self.para_dict['train_data_pattern']
        self.dist_bar = self.para_dict['dist_bar']
        self.prob_bar = 0.8
        self.img = img

    def _dist(self, patt1, patt2):
        diff = np.array(patt1)-np.array(patt2)
        distance = sum(diff**2)
        return distance

    def face_detect(self, img_path):
        test_dist = []
        test_data = self.img.pattern_return(img_path)

        for patt in self.train_patterns:
            distance = self._dist(test_data, patt)
            test_dist.append(distance)

        # calculate test data probability
        test_prob = sum(dist <= self.dist_bar for dist in test_dist) / len(test_dist)
        print(test_prob)

        if test_prob >= self.prob_bar:
            print("This is a face")
        else:
            print("This is NOT a face")
