import numpy as np
import pickle


class FaceDetect:
    def __init__(self, img):
        # parameter initialize

        self.img = img

    def _dist(self, patt1, patt2):
        diff = np.array(patt1)-np.array(patt2)
        distance = sum(diff**2)
        return distance

    def face_detect(self, img_path):

        test_patterns = self.img.pattern_return(img_path)

        for pattern in test_patterns:
            test_dist = []
            for patt in self.train_patterns[:100]:
                distance = self._dist(pattern, patt)
                test_dist.append(distance)

            # calculate test data probability
            test_prob = sum(dist <= self.dist_bar for dist in test_dist) / len(test_dist)
            print(test_prob)

            if test_prob >= self.prob_bar:
                print("This is a face")
            else:
                # print("This is NOT a face")
                pass
