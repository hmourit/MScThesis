from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *
import json

import os
from os.path import join
import sys

sys.path.append('..')
from glob import glob
from pprint import pprint
from time import time

DATA_PATH = '../../bucket/data/'
RESULTS_PATH = '../../bucket/results/'
TABLES_FOLDET = '~/Dropbox/MSc Thesis/tables'

