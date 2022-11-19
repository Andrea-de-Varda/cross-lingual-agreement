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

parser = argparse.ArgumentParser(description='Probe on XLM')
parser.add_argument('-c_sentences', action='store', dest='sentences', help='Store sentence corpus destination')
parser.add_argument('-c_tags', action='store', dest='tags', help='Store tag corpus destination')
parser.add_argument('-c_positions', action='store', dest='positions', help='Store position corpus destination')
results = parser.parse_args()

l1 = 0.001
l2 = 0.001
epochs = 30

dirName = results.sentences+"-XLM"
try:
    mkdir(dirName)
    print("Directory " , dirName ,  "created") 
except FileExistsError:
    print("Directory " , dirName ,  "already exists")

file_train_sentences = results.sentences+"_train.txt"
file_test_sentences = results.sentences+"_test.txt"
file_train_tags = results.tags+"_train.txt"
file_test_tags = results.tags+"_test.txt"
file_train_positions = results.positions+"_train.txt"
file_test_positions = results.positions+"_test.txt"

##############################
# EXTRACTING REPRESENTATIONS #
##############################


import neurox.data.extraction.transformers_extractor as transformers_extractor

if path.isfile(dirName+'/activations_train.json') == False:
    transformers_extractor.extract_representations('xlm-roberta-base',
        file_train_sentences,
        dirName+'/activations_train.json',
        aggregation="average"
    )

if path.isfile(dirName+'/activations_test.json') == False:
    transformers_extractor.extract_representations('xlm-roberta-base',
        file_test_sentences,
        dirName+'/activations_test.json',
        aggregation="average"
    )

import neurox.data.loader as data_loader
activations_train, num_layers_train = data_loader.load_activations(dirName+'/activations_train.json', 768, is_brnn=False)

activations_test, num_layers_test = data_loader.load_activations(dirName+'/activations_test.json', 768, is_brnn=False)

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

# load labels
# cannot use default data_loader since I use a subpart of the activations (only in the verb position). The output is a dict {'source': [['the', 'author', 'runs'], [...]], 'target': [['OK', 'OK'], [...]]}.

tags_train = [[item] for item in open(file_train_tags).read().split("\n")[:-1]]
tags_test = [[item] for item in open(file_test_tags).read().split("\n")[:-1]]

source_train = [[item] for item in open(file_train_sentences).read().split("\n")[:-1]]
source_test = [[item] for item in open(file_test_sentences).read().split("\n")[:-1]]

tokens_train = {'source': source_train, 'target': tags_train}
tokens_test = {'source': source_test, 'target': tags_test}
# tokens_train = data_loader.load_data(file_train_sentences, file_train_tags, activations_train, 512)
# tokens_test = data_loader.load_data(file_test_sentences, file_test_tags, activations_test, 512)


import neurox.interpretation.utils as utils
X_train, y_train, mapping_train = utils.create_tensors(tokens_train, activations_train, "True")
label2idx_train, idx2label_train, src2idx_train, idx2src_train = mapping_train

X_test, y_test, mapping_test = utils.create_tensors(tokens_test, activations_test, "True")
label2idx_test, idx2label_test, src2idx_test, idx2src_test = mapping_test


X_train_balanced, y_train_balanced = neurox.interpretation.utils.balance_binary_class_data(X_train, y_train)

############################
# TRAIN PROBING CLASSIFIER #
############################

def save_results_classifier(item, probe_output, output_list=True):
    if output_list == True:
        return [item, probe_output["__OVERALL__"], probe_output["False"], probe_output["True"]]
    else:
        return " ".join([item, str(probe_output["__OVERALL__"]), str(probe_output["False"]), str(probe_output["True"])])

import neurox.interpretation.linear_probe as linear_probe
probe = linear_probe.train_logistic_regression_probe(X_train_balanced, y_train_balanced, lambda_l1=l1, lambda_l2=l2, num_epochs=epochs)

probe_eval = linear_probe.evaluate_probe(probe, X_test, y_test, idx_to_class=idx2label_test)

######################
# layer-wise probing #
######################

import neurox.interpretation.ablation as ablation

layerwise = []
for l in range(0, 13):
    layer_train = ablation.filter_activations_by_layers(X_train_balanced, [l], 13)
    layer_test = ablation.filter_activations_by_layers(X_test, [l], 13)
    probe_layer = linear_probe.train_logistic_regression_probe(layer_train, y_train_balanced, lambda_l1=l1, lambda_l2=l2, num_epochs=epochs)#, learning_rate = l_rate)
    out = linear_probe.evaluate_probe(probe_layer, layer_test, y_test, idx_to_class=idx2label_test)
    print(out)
    layerwise.append(save_results_classifier(l, out))

layerwise = pd.DataFrame(layerwise, columns=["l", "acc", "acc_no", "acc_ok"])
layerwise.to_csv(dirName+"/layerwise.csv", index=False)

plt.plot(layerwise["acc"])
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.savefig(dirName+"/Acc_by_layer.png")
plt.clf()

######################
# Get NEURON RANKING #
######################

ordering, cutoffs = linear_probe.get_neuron_ordering(probe, label2idx_train)

with open(dirName+'/ordering', 'wb') as f:
    pickle.dump(ordering, f)
with open(dirName+'/cutoffs', 'wb') as f:
    pickle.dump(cutoffs, f)

 
