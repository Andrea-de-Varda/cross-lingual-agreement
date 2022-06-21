import pickle
from os import chdir
from random import sample
import pandas as pd
import matplotlib.pyplot as plt
chdir("/path/")

# visualize layers, long vp coordination
ru_layer_vp = pd.read_csv('ru_long_vp_coord/layerwise.csv')
en_layer_vp = pd.read_csv('en_long_vp_coord/layerwise.csv')
he_layer_vp = pd.read_csv('he_long_vp_coord/layerwise.csv')
de_layer_vp = pd.read_csv('de_long_vp_coord/layerwise.csv')
fr_layer_vp = pd.read_csv('fr_long_vp_coord/layerwise.csv')

plt.figure(dpi=1200)
for layer, name in zip([ru_layer_vp, en_layer_vp, he_layer_vp, de_layer_vp, fr_layer_vp], ["Russian", "English", "Hebrew", "German", "French"]):
    plt.plot(layer.l, layer.acc, label=name)
plt.axvline(7, linestyle='--', color="black")
plt.xlabel("Layer")
plt.ylabel('Accuracy')
plt.legend(loc=2)
plt.show()

out = pd.DataFrame()
for name, layer in zip(["ru", "en", "he", "de", "fr"],[ru_layer_vp, en_layer_vp, he_layer_vp, de_layer_vp, fr_layer_vp]):
    out[name] = layer.acc

out['mean'] = out.mean(axis=1)  

# visualize layers, simple agreement
ru_layer_vp = pd.read_csv('ru_simple_agreement/layerwise.csv')
en_layer_vp = pd.read_csv('en_simple_agreement/layerwise.csv')
he_layer_vp = pd.read_csv('he_simple_agreement/layerwise.csv')
de_layer_vp = pd.read_csv('de_simple_agreement/layerwise.csv')
fr_layer_vp = pd.read_csv('fr_simple_agreement/layerwise.csv')

fig = plt.figure(dpi=1200)
for layer, name in zip([ru_layer_vp, en_layer_vp, he_layer_vp, de_layer_vp, fr_layer_vp], ["Russian", "English", "Hebrew", "German", "French"]):
    plt.plot(layer.l, layer.acc, label=name)
plt.axvline(7, linestyle='--', color="black")
plt.xlabel("Layer")
plt.ylabel('Accuracy')
plt.legend(loc=2)
plt.show()

out_s = pd.DataFrame()
for name, layer in zip(["ru", "en", "he", "de", "fr"],[ru_layer_vp, en_layer_vp, he_layer_vp, de_layer_vp, fr_layer_vp]):
    out_s[name] = layer.acc

out_s['mean'] = out_s.mean(axis=1) 
