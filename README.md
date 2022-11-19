## cross-lingual-agreement

In this set of studies, we analyze the behaviour of individual neurons in two massively multilingual models ([mBERT](https://arxiv.org/abs/1810.04805) and [XLM-R](https://arxiv.org/abs/1911.02116)) searching for single units responding to subject-verb number agreement in five languages (English, German, French, Hebrew, Russian, see [CLAMS](https://aclanthology.org/2020.acl-main.490/)). We found a significant cross-lingual overlap in the units encoding syntactic agreement, that peaked in the middle-to-deep layers of the networks. 

We focus on two conditions, namely *short distance agreement* (1) and *long-distance VP coordination* (2).
1. <ins>The authors</ins> <ins>smile/\*smiles</ins>
2. <ins>The author</ins> knows many different foreign languages and <ins>\*like/likes</ins> to watch television shows

We train a logistic classifier to predict the grammaticality of a token given the phrasal context relying on the internal activation of the model in the verb position. Then, we use the weights learned by the classifier to identify the relevant hidden units, and the [super exact test](https://www.nature.com/articles/srep16923) to identify significant intersection between relevant neurons across languages.



*Note: our scripts rely on [NeuroX](https://www.semanticscholar.org/paper/NeuroX%3A-A-Toolkit-for-Analyzing-Individual-Neurons-Dalvi-Nortonsmith/3c8d7c5a9eb3bf84c0ea47e3416f79d5a49f71fd), a toolkit for analyzing individual neurons in neural networks, which has an [excellent documentation](https://neurox.qcri.org/docs/) for those that may be interested in replicating our analyses. Also note that the intersection analyses are performed with R, while everything else is in Python.*


### Files - main experiment

#### Constructing the datasets
The file `make_data.py` creates the train-test sets for the probing from the [CLAMS](https://aclanthology.org/2020.acl-main.490/) dataset. It also performs some checks on the obtained data. It needs to be run from the same folder as the CLAMS dataset, as downloaded from the [GitHub repository](https://github.com/aaronmueller/clams).

#### Probing
The files `probe_verb_mBERT.py` and `probe_verb_mBERT.py` perform the probing experiments on the data obtained in the previous step. They should be run as follows:
```
python3 probe_verb_mBERT.py -c_sentences <path/to/sentences> -c_tags <path/to/tags> -c_positions <path/to/positions>
```
Where the tags correspond to the grammaticality of the sentence (grammatical-ungrammatical) and the positions correspond to the indexes of the verb tokens, on which the probing is performed.

We also experiment with the _probeless_ ranking method, proposed by [Antverg & Belinkov (2022)](https://arxiv.org/abs/2110.07483). The scripts for performing the _probeless_ analysis are `probeless.py` and `probeless-XLM.py`.

#### Intersection analysis
The R script for performing the intersection analysis is in `Plot_intersections.R`. The intersection analyses test the statistical significance of the set intersections between the top-100 neurons identified with the probing task. 



### Files - follow-up analyses

#### 

#### Constructing the Italian dataset
In a follow-up analysis, we test the cross-lingual neurons identified before in Italian, a language that is not present in the [CLAMS](https://aclanthology.org/2020.acl-main.490/) dataset. To do so, we create four agreement conditions (`make_italian_dataset.py`) by computing the Cartesian product between determiners (D), nouns (N), and verbs (V).
1. Singular agreement: $D$ × $N$ × $V$
2. Plural agreement: $D_p$ × $N_p$ × $V_p$
3. Violated singular agreement: $D$ × $N$ × $V_p$
4. Violated plural agreement: $D_p$ × $N_p$ × $V$

We then test the cross-lingual neurons identified before with the script `it_test_single_neurons.py`


