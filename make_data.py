import pandas as pd
import random

def list_to_txt(l, filename):
    with open(filename, 'w', encoding="utf8") as f:
        for item in l:
            f.write("%s\n" % item)

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def get_data(location, name, n, n1):
    l = [l.split("\t") for l in open(location, encoding="utf8").read().split("\n")]#[:-1]
    if l[-1] == ['']:
        del l[-1]
    df = pd.DataFrame([[i[0][0], i[0][1], i[1][0], i[1][1]] for i in chunks(l, 2)], columns = ["grammatical", "sent", "grammatical_next", "sent_next"])
    sentences, tags, position = [], [], []
    for index, row in df.iterrows():
        sentences.append(row["sent"]); tags.append(row["grammatical"])
        sentences.append(row["sent_next"]); tags.append(row["grammatical_next"])
        sent = row["sent"].split()
        sent1 = row["sent_next"].split()
        for index, pair in enumerate(zip(sent, sent1)):
            if pair[0] != pair[1]:
                #print(index, pair)
                position.append(index)
                position.append(index)
                break
    #print([i for i in zip(sentences, tags, position)]) # check
    print(len(sentences), len(tags), len(position))
    
    random.seed(1)
    randomize = list(zip(sentences, tags, position))
    random.shuffle(randomize)
    sentences, tags, position = zip(*randomize)
    
    sentences = sentences[n:n1]; tags = tags[n:n1]; position = position[n:n1]
    
    threshold_test = int(0.8*len(sentences))
    sentences_train = sentences[:threshold_test]
    sentences_test = sentences[threshold_test:]
    tags_train = tags[:threshold_test]
    tags_test = tags[threshold_test:]
    position_train = position[:threshold_test]
    position_test = position[threshold_test:]
    
    if len(sentences_test) == len(tags_test) == len(position_test):
        list_to_txt(sentences_test, name+"_test.txt")
        list_to_txt(tags_test, name+"_tags_test.txt")
        list_to_txt(position_test, name+"_position_test.txt")
    else:
        print("Lenght (test) does not match!")
        
    if len(sentences_train) == len(tags_train) == len(position_train):
        list_to_txt(sentences_train, name+"_train.txt")
        list_to_txt(tags_train, name+"_tags_train.txt")
        list_to_txt(position_train, name+"_position_train.txt")
    else:
        print("Lenght (train) does not match!")
        
        
def get_data_all(location, name, n, n1): # to use in case of prepositional phrase, where we do not want to split the dataset
    l = [l.split("\t") for l in open(location, encoding="utf8").read().split("\n")]#[:-1]
    if l[-1] == ['']:
        del l[-1]
    df = pd.DataFrame([[i[0][0], i[0][1], i[1][0], i[1][1]] for i in chunks(l, 2)], columns = ["grammatical", "sent", "grammatical_next", "sent_next"])
    sentences, tags, position = [], [], []
    for index, row in df.iterrows():
        sentences.append(row["sent"]); tags.append(row["grammatical"])
        sentences.append(row["sent_next"]); tags.append(row["grammatical_next"])
        sent = row["sent"].split()
        sent1 = row["sent_next"].split()
        for index, pair in enumerate(zip(sent, sent1)):
            if pair[0] != pair[1]:
                #print(index, pair)
                position.append(index)
                position.append(index)
                break
    #print([i for i in zip(sentences, tags, position)]) # check
    print(len(sentences), len(tags), len(position))
    sentences = sentences[n:n1]; tags = tags[n:n1]; position = position[n:n1]
    
    randomize = list(zip(sentences, tags, position))
    random.shuffle(randomize)
    sentences, tags, position = zip(*randomize)
    
    if len(sentences) == len(tags) == len(position):
        list_to_txt(sentences, name+".txt")
        list_to_txt(tags, name+"_tags.txt")
        list_to_txt(position, name+"_position.txt")
    else:
        print("Lenght (test) does not match!")


# short distance: simple agreement
get_data("clams/en_evalset/simple_agrmt.txt", "en_simple_agreement", 0, 280)
get_data("clams/ru_evalset/simple_agrmt.txt", "ru_simple_agreement", 0, 280)
get_data("clams/de_evalset/simple_agrmt.txt", "de_simple_agreement", 0, 280)
get_data("clams/fr_evalset/simple_agrmt.txt", "fr_simple_agreement", 0, 280)
get_data("clams/he_evalset/simple_agrmt.txt", "he_simple_agreement", 0, 280)

# long distance: long VP coordination
get_data("clams/en_evalset/long_vp_coord.txt", "en_long_vp_coord", 0, 280)
get_data("clams/ru_evalset/long_vp_coord.txt", "ru_long_vp_coord", 0, 280)
get_data("clams/de_evalset/long_vp_coord.txt", "de_long_vp_coord", 0, 280)
get_data("clams/fr_evalset/long_vp_coord.txt", "fr_long_vp_coord", 0, 280)
get_data("clams/he_evalset/long_vp_coord.txt", "he_long_vp_coord", 0, 280)

# medium distance: across a prepositional phrase
get_data_all("clams/en_evalset/prep_anim.txt", "en_prep", 0, 11200) #11200 max len prep_anim he_evalset
get_data_all("clams/ru_evalset/prep_anim.txt", "ru_prep", 0, 11200)
get_data_all("clams/de_evalset/prep_anim.txt", "de_prep", 0, 11200)
get_data_all("clams/fr_evalset/prep_anim.txt", "fr_prep", 0, 11200)
get_data_all("clams/he_evalset/prep_anim.txt", "he_prep", 0, 11200)


# check if the sentence pairs actually differ only by the verb in agreement
def check_data(location):
    l = [l.split("\t") for l in open(location, encoding="utf8").read().split("\n")]#[:-1]
    if l[-1] == ['']:
        del l[-1]
    df = pd.DataFrame([[i[0][0], i[0][1], i[1][0], i[1][1]] for i in chunks(l, 2)], columns = ["grammatical", "sent", "grammatical_next", "sent_next"])
    for index, row in df.iterrows():
        sent = row["sent"].split()
        sent1 = row["sent_next"].split()
        c = 0
        for index, pair in enumerate(zip(sent, sent1)):
            if pair[0] != pair[1]:
                #print(index, pair)
                c += 1
                if c> 1:
                    print(sent, sent1, c)

# short distance: simple agreement
check_data("clams/en_evalset/simple_agrmt.txt")
check_data("clams/ru_evalset/simple_agrmt.txt")
check_data("clams/de_evalset/simple_agrmt.txt")
check_data("clams/fr_evalset/simple_agrmt.txt") # also differs by the N that forms a nominal predicate with the verb 
check_data("clams/he_evalset/simple_agrmt.txt")

# long distance: long VP coordination
check_data("clams/en_evalset/long_vp_coord.txt")
check_data("clams/ru_evalset/long_vp_coord.txt")
check_data("clams/de_evalset/long_vp_coord.txt")
check_data("clams/fr_evalset/long_vp_coord.txt")
check_data("clams/he_evalset/long_vp_coord.txt")

# avg distance: PP
check_data("clams/en_evalset/prep_anim.txt")
check_data("clams/ru_evalset/prep_anim.txt")
check_data("clams/de_evalset/prep_anim.txt")
check_data("clams/fr_evalset/prep_anim.txt")
check_data("clams/he_evalset/prep_anim.txt")
