from __future__ import division

import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
#from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

def do_myfashion_evaluation(dataset, predictions, output_folder, logger):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    print("do_myfashion_evaluation")
    