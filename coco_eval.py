from utils.init import *
from utils.python_utils import *
import sys
import pdb
import re
import numpy as np
import os

COCO_EVAL_PATH = coco_eval_path
sys.path.insert(0,COCO_EVAL_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def score_generation(gt_filename=None, generation_result=None):

  coco_dict = read_json(generation_result)
  coco = COCO(gt_filename)
  generation_coco = coco.loadRes(generation_result)
  coco_evaluator = COCOEvalCap(coco, generation_coco)
  #coco_image_ids = [self.sg.image_path_to_id[image_path]
  #                  for image_path in self.images]
  coco_image_ids = [j['image_id'] for j in coco_dict]
  coco_evaluator.params['image_id'] = coco_image_ids
  results = coco_evaluator.evaluate(return_results=True)
  return results

