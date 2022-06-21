import neurox
import numpy as np
from numpy import loadtxt
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from os import mkdir, path
import pickle
import sys
import torch
sys.path.append("..")

parser = argparse.ArgumentParser(description='Extract representations')
parser.add_argument('-c_sentences', action='store', dest='sentences', help='Store sentence corpus destination')
results = parser.parse_args()

dirName = results.sentences

try:
    mkdir(dirName)
    print("Directory " , dirName ,  "created") 
except FileExistsError:
    print("Directory " , dirName ,  "already exists")

sentences = dirName+".txt"

##############################
# EXTRACTING REPRESENTATIONS #
##############################

import neurox.data.extraction.transformers_extractor as transformers_extractor

if path.isfile(dirName+'/activations.json') == False:
    transformers_extractor.extract_representations('bert-base-multilingual-cased',
        sentences,
        dirName+'/activations.json',
        aggregation="average"
    )
