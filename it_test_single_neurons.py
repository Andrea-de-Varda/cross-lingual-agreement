import neurox
import neurox.interpretation.ablation as ablation
import neurox.interpretation.utils as utils
import neurox.data.loader as data_loader
import numpy as np
from numpy import loadtxt
import statsmodels.api as sm
import argparse
import sys
import torch
import pandas as pd
from scipy.stats import binom
import researchpy as rp
from sklearn.metrics import accuracy_score
import re
from pymer4.models import Lmer
import matplotlib.pyplot as plt
import statistics
import random
sys.path.append("..")

class data:
   def __init__(self, sentences):
      self.sentences = sentences
      self.tags = sentences+"_tags"
      self.positions = sentences+"_position"
      
it = data("it_prep")

d_y = {"False":0, "True":1}

outcome = []
for results in [it]:
    dirName = results.sentences
    print(dirName)
    
    sentences = results.sentences+".txt"
    tags = results.tags+".txt"
    positions = results.positions+".txt"
    
    activations, num_layers = data_loader.load_activations(dirName+'/activations.json', 768, is_brnn=False)
    
    positions = loadtxt(positions, delimiter="\n", unpack=False).astype(int)
    
    act = []
    for array, index in zip(activations, positions):
        v = array[index]
        act.append(v.reshape(1, 9984))
    activations = act
    
    tags = [[item] for item in open(tags, encoding="utf8").read().split("\n")[:-1]]
    tags_uniform = [d_y[i[0]] for i in tags] 
    source = [[item] for item in open(sentences, encoding="utf8").read().split("\n")[:-1]]
    tokens = {'source': source, 'target': tags}

    
    X, y, mapping = utils.create_tensors(tokens, activations, "True")
    label2idx, idx2label, src2idx, idx2src = mapping
      
    neuron_list = [[6533],[7674]]
    
    for neuron in neuron_list:
        print(neuron)
        test = ablation.filter_activations_keep_neurons(X, neuron)
        for activation, label in zip(test, tags_uniform):
            outcome.append([neuron[0], activation[0], label])
            
outcome_df = pd.DataFrame(outcome, columns=["neuron", "act", "label"])
long = outcome_df[outcome_df["neuron"] == 7674]
short = outcome_df[outcome_df["neuron"] == 6533]

np.mean(long[long.label==0].act)
np.mean(long[long.label==1].act)
rp.ttest(long[long.label==0].act, long[long.label==1].act)

np.mean(short[short.label==0].act)
np.mean(short[short.label==1].act)
rp.ttest(short[short.label==0].act, short[short.label==1].act)

# note: unlike in previous condition, here the first 50% of data is singular, second is plural. Need to sample randomly train-test (while keeping balance T-F)
threshold = int(3200*0.8)

def binarize_n(n):
    out = 0
    if n > 0.5:
        out = 1
    return out

def acc(pred, target, num=True, return_acc=False):
    out = []
    for tup in zip(pred, target):
        if int(tup[0]) == int(tup[1]):
            out.append(1)
        else:
            out.append(0)
    if return_acc:
        return out
    else:
        return sum(out)/len(out), sum(out)

def pbinom(num):
    return sum([binom.pmf(n, 640, 0.5) for n in range(num, 640)])

results = []
for df, dfname in [(long, "long"), (short, "short")]:#, (long1, "long1"), (long2, "long2"), (short1, "short1"), (short2, "short2")]:
    print("\n", dfname.upper(), "\n")
    df = df.sample(frac=1, random_state=1) # shuffle
    
    X_train = df[["act"]][:threshold]
    y_train = df[["label"]][:threshold]
    log_reg = sm.Logit(y_train, X_train).fit()
    
    X_test = df[["act"]][threshold:]
    y_test = df[["label"]][threshold:]
    y_pred = log_reg.predict(X_test)
    y_pred = [binarize_n(n) for n in y_pred]
    print(len(y_test) == len(y_pred))
          
    score = acc(y_test.label,y_pred)
    p = pbinom(score[1])
    results.append([dfname, score[0], p])
    #print(log_reg.summary())
    print(score[0])

results = pd.DataFrame(results, columns=["cond", "acc", "p"])
results

outcome = []
for results in [it]:
    dirName = results.sentences
    print(dirName)
    
    sentences = results.sentences+".txt"
    tags = results.tags+".txt"
    positions = results.positions+".txt"
    activations, num_layers = data_loader.load_activations(dirName+'_xlm/activations.json', 768, is_brnn=False)
    positions = loadtxt(positions, delimiter="\n", unpack=False).astype(int)
    
    act = []
    for array, index in zip(activations, positions):
        v = array[index]
        act.append(v.reshape(1, 9984))
    activations = act
    
    tags = [[item] for item in open(tags, encoding="utf8").read().split("\n")[:-1]]
    tags_uniform = [d_y[i[0]] for i in tags] 
    source = [[item] for item in open(sentences, encoding="utf8").read().split("\n")[:-1]]
    tokens = {'source': source, 'target': tags}

    X, y, mapping = utils.create_tensors(tokens, activations, "True")
    label2idx, idx2label, src2idx, idx2src = mapping
      
    neuron_list = [[5621],[5066]]
    
    for neuron in neuron_list:
        print(neuron)
        test = ablation.filter_activations_keep_neurons(X, neuron)
        
        language = results.sentences[:2]
        for activation, label in zip(test, tags_uniform):
            outcome.append([neuron[0], activation[0], label])


outcome_df_xlm = pd.DataFrame(outcome, columns=["neuron", "act", "label"])

long_xlm = outcome_df_xlm[outcome_df_xlm["neuron"] == 5066]
short_xlm = outcome_df_xlm[outcome_df_xlm["neuron"] == 5621]
    
results_xlm = []
for df, dfname in [(long_xlm, "long"), (short_xlm, "short")]:#, (long1, "long1"), (long2, "long2"), (short1, "short1"), (short2, "short2")]:
    print("\n\n\n", dfname.upper(), "\n\n")
    df = df.sample(frac=1, random_state=1) # shuffle
    
    X_train = df[["act"]][:threshold]
    y_train = df[["label"]][:threshold]
    log_reg = sm.Logit(y_train, X_train).fit()
    
    X_test = df[["act"]][threshold:]
    y_test = df[["label"]][threshold:]
    y_pred = log_reg.predict(X_test)
    y_pred = [binarize_n(n) for n in y_pred]
    print(len(y_test) == len(y_pred))
    score = acc(y_test.label,y_pred)
    
    p = pbinom(score[1])
    
    results_xlm.append([dfname, score[0], p])
    #print(log_reg.summary())
    print(score[0])
        
results_xlm = pd.DataFrame(results_xlm, columns=["cond", "acc", "p"])
results_xlm

##################
# RANDOM NEURONS #
##################

threshold_0 = 768*1 
threshold_1 = 768*2
threshold_2 = 768*3
threshold_3 = 768*4
threshold_4 = 768*5
threshold_5 = 768*6
threshold_6 = 768*7
threshold_7 = 768*8 
threshold_8 = 768*9
threshold_9 = 768*10
threshold_10 = 768*11
threshold_11 = 768*12

random.seed(0)
layer6 = random.sample([i for i in range(threshold_5, threshold_6)], 30)
random.seed(0)
layer7 = random.sample([i for i in range(threshold_6, threshold_7)], 30)
random.seed(0)
layer8 = random.sample([i for i in range(threshold_7, threshold_8)], 30)
random.seed(0)
layer9 = random.sample([i for i in range(threshold_8, threshold_9)], 30)


outcome = []
for results in [it]:
    dirName = results.sentences
    print(dirName)
    sentences = results.sentences+".txt"
    tags = results.tags+".txt"
    positions = results.positions+".txt"
    activations, num_layers = data_loader.load_activations(dirName+'/activations.json', 768, is_brnn=False)
    positions = loadtxt(positions, delimiter="\n", unpack=False).astype(int)
    act = []
    for array, index in zip(activations, positions):
        v = array[index]
        act.append(v.reshape(1, 9984))
    activations = act
    
    tags = [[item] for item in open(tags, encoding="utf8").read().split("\n")[:-1]]
    tags_uniform = [d_y[i[0]] for i in tags] 
    source = [[item] for item in open(sentences, encoding="utf8").read().split("\n")[:-1]]
    tokens = {'source': source, 'target': tags}

    X, y, mapping = utils.create_tensors(tokens, activations, "True")
    del activations, act
    label2idx, idx2label, src2idx, idx2src = mapping
      
    neuron_list = layer8+layer9
    
    for neuron in neuron_list:
        print(neuron)
        test = ablation.filter_activations_keep_neurons(X, neuron)
        
        language = results.sentences[:2]
        for activation, label in zip(test, tags_uniform):
            outcome.append([neuron, activation, label])

outcome_df_random = pd.DataFrame(outcome, columns=["neuron", "act", "label"])

results_random = []
mean_acc = []
mean_tot = []
for neuron in layer9:
    temp_df = outcome_df_random[outcome_df_random.neuron == neuron]
    temp_df = temp_df.sample(frac=1, random_state=1) # shuffle
    X_train = temp_df[["act"]][:threshold]
    y_train = temp_df[["label"]][:threshold]
    log_reg = sm.Logit(y_train, X_train).fit()
    
    X_test = temp_df[["act"]][threshold:]
    y_test = temp_df[["label"]][threshold:]
    y_pred = log_reg.predict(X_test)
    y_pred = [binarize_n(n) for n in y_pred]
    score = acc(y_test.label,y_pred)
    tot = score[1]
    
    mean_acc.append(score[0])
    mean_tot.append(tot)
accuracy = np.mean(mean_acc)
p = pbinom(int(np.mean(mean_tot)))
results_random.append(["layer 9", accuracy, p])

mean_acc = []
mean_tot = []
for neuron in layer8:
    temp_df = outcome_df_random[outcome_df_random.neuron == neuron]
    temp_df = temp_df.sample(frac=1, random_state=1) # shuffle
    X_train = temp_df[["act"]][:threshold]
    y_train = temp_df[["label"]][:threshold]
    log_reg = sm.Logit(y_train, X_train).fit()
    
    X_test = temp_df[["act"]][threshold:]
    y_test = temp_df[["label"]][threshold:]
    y_pred = log_reg.predict(X_test)
    y_pred = [binarize_n(n) for n in y_pred]
    score = acc(y_test.label,y_pred)
    tot = score[1]
    
    mean_acc.append(score[0])
    mean_tot.append(tot)
    
accuracy = np.mean(mean_acc)
p = pbinom(int(np.mean(mean_tot)))
results_random.append(["layer 8", accuracy, p])

results_random = pd.DataFrame(results_random, columns=["cond", "acc", "p"])
results_random

#################
# RANDOM, XLM-R ###############################################################
#################

outcome = []
for results in [it]:
    dirName = results.sentences
    print(dirName)
    sentences = results.sentences+".txt"
    tags = results.tags+".txt"
    positions = results.positions+".txt"
    activations, num_layers = data_loader.load_activations(dirName+'_xlm/activations.json', 768, is_brnn=False)
    positions = loadtxt(positions, delimiter="\n", unpack=False).astype(int)
    act = []
    for array, index in zip(activations, positions):
        v = array[index]
        act.append(v.reshape(1, 9984))
    activations = act
    
    tags = [[item] for item in open(tags, encoding="utf8").read().split("\n")[:-1]]
    tags_uniform = [d_y[i[0]] for i in tags] 
    source = [[item] for item in open(sentences, encoding="utf8").read().split("\n")[:-1]]
    tokens = {'source': source, 'target': tags}

    X, y, mapping = utils.create_tensors(tokens, activations, "True")
    label2idx, idx2label, src2idx, idx2src = mapping
      
    neuron_list = layer6+layer7
    for neuron in neuron_list:
        print(neuron)
        test = ablation.filter_activations_keep_neurons(X, neuron)
        for activation, label in zip(test, tags_uniform):
            outcome.append([neuron, activation, label])

outcome_df_random_xlm = pd.DataFrame(outcome, columns=["neuron", "act", "label"])

results_random_xlm = []
mean_acc = []
mean_tot = []
for neuron in layer6:
    temp_df = outcome_df_random_xlm[outcome_df_random_xlm.neuron == neuron]
    temp_df = temp_df.sample(frac=1, random_state=1) # shuffle
    X_train = temp_df[["act"]][:threshold]
    y_train = temp_df[["label"]][:threshold]
    log_reg = sm.Logit(y_train, X_train).fit()
    
    X_test = temp_df[["act"]][threshold:]
    y_test = temp_df[["label"]][threshold:]
    y_pred = log_reg.predict(X_test)
    y_pred = [binarize_n(n) for n in y_pred]
    score = acc(y_test.label,y_pred)
    tot = score[1]
    
    mean_acc.append(score[0])
    mean_tot.append(tot)
accuracy = np.mean(mean_acc)
p = pbinom(int(np.mean(mean_tot)))
results_random_xlm.append(["layer 6", accuracy, p])
    
mean_acc = []
mean_tot = []
for neuron in layer7:
    temp_df = outcome_df_random_xlm[outcome_df_random_xlm.neuron == neuron]
    temp_df = temp_df.sample(frac=1, random_state=1) # shuffle
    X_train = temp_df[["act"]][:threshold]
    y_train = temp_df[["label"]][:threshold]
    log_reg = sm.Logit(y_train, X_train).fit()
    
    X_test = temp_df[["act"]][threshold:]
    y_test = temp_df[["label"]][threshold:]
    y_pred = log_reg.predict(X_test)
    y_pred = [binarize_n(n) for n in y_pred]
    score = acc(y_test.label,y_pred)
    tot = score[1]
    
    mean_acc.append(score[0])
    mean_tot.append(tot)
accuracy = np.mean(mean_acc)
p = pbinom(int(np.mean(mean_tot)))
results_random_xlm.append(["layer 7", accuracy, p])

results_random_xlm = pd.DataFrame(results_random_xlm, columns=["cond", "acc", "p"])
