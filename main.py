"""
Group Name: Power Machine
Group Members: YU DU, Bincheng Huang, Siyi Liu
Description: CS 542 Course Project, Face Recognition Algorithm.
"""

from LBPH import *
import sys


def main():
    img = Image()
    img.pattern_return('test_data/' + sys.argv[1])


if __name__ == "__main__":
    main()
