from __future__ import division, print_function
import sys
from scripts import classify, classify_with_pca, classify_with_std


def main(argv):
    if len(argv) < 8:
        classify.main(argv)
    else:
        # classify_with_pca.main(argv)
        classify_with_std.main(argv)

if __name__ == '__main__':
    main(sys.argv)
