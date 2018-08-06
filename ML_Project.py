"""
Group Name: Power Machine
Group Members: YU DU, Bincheng Huang, Siyi Liu
Description: CS 542 Course Project, Face Recognition Algorithm.
"""

from LBPH import *
from face_detection import *


def main():
    img = Image()
    face_detect = FaceDetect(img)
    face_detect.face_detect('test_data/people_4.png')


if __name__ == "__main__":
    main()
