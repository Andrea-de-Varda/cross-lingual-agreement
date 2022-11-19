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

parser = argparse.ArgumentParser(description='Probing on mBERT')
parser.add_argument('-sent', action='store', dest='sentences', help='Store sentence corpus destination')
in_data = parser.parse_args()

dirName = in_data.sentences

try:
    mkdir(dirName+"-XLM")
    print("Directory " , dirName ,  "created") 
except FileExistsError:
    print("Directory " , dirName ,  "already exists")
    
class data:
   def __init__(self, sentences):
      self.sentences = sentences
      self.tags = sentences+"_tags"
      self.positions = sentences+"_position"

data = data(dirName)

file_train_sentences = data.sentences+"_train.txt"
file_test_sentences = data.sentences+"_test.txt"
file_train_tags = data.tags+"_train.txt"
file_test_tags = data.tags+"_test.txt"
file_train_positions = data.positions+"_train.txt"
file_test_positions = data.positions+"_test.txt"

###########################
# LOADING REPRESENTATIONS #
###########################

dirName = dirName+"-XLM"

import neurox.data.loader as data_loader
activations_train, num_layers_train = data_loader.load_activations(dirName+'/activations_train.json', 768, is_brnn=True)

activations_test, num_layers_test = data_loader.load_activations(dirName+'/activations_test.json', 768, is_brnn=True)

# only extract representation in the verb position
positions_train = loadtxt(file_train_positions, delimiter="\n", unpack=False).astype(int)
positions_test = loadtxt(file_test_positions, delimiter="\n", unpack=False).astype(int)

act_train = []
for array, index in zip(activations_train, positions_train):
    v = array[index]
    act_train.append(v.reshape(1, 9984))
activations_train = act_train

act_test = []
for array, index in zip(activations_test, positions_test):
    v = array[index]
    act_test.append(v.reshape(1, 9984))
activations_test = act_test

tags_train = [[item] for item in open(file_train_tags).read().split("\n")[:-1]]
tags_test = [[item] for item in open(file_test_tags).read().split("\n")[:-1]]

source_train = [[item] for item in open(file_train_sentences).read().split("\n")[:-1]]
source_test = [[item] for item in open(file_test_sentences).read().split("\n")[:-1]]

tokens_train = {'source': source_train, 'target': tags_train}
tokens_test = {'source': source_test, 'target': tags_test}

import neurox.interpretation.utils as utils
X_train, y_train, mapping_train = utils.create_tensors(tokens_train, activations_train, "True")
label2idx_train, idx2label_train, src2idx_train, idx2src_train = mapping_train

X_test, y_test, mapping_test = utils.create_tensors(tokens_test, activations_test, "True")
label2idx_test, idx2label_test, src2idx_test, idx2src_test = mapping_test

X_train_balanced, y_train_balanced = neurox.interpretation.utils.balance_binary_class_data(X_train, y_train) 

#############
# PROBELESS #
#############

import neurox.interpretation.probeless as probeless

ranking = probeless.get_neuron_ordering(X_train_balanced, y_train_balanced)

with open(dirName+'/ordering_probeless', 'wb') as f:
    pickle.dump(ranking, f)

 
