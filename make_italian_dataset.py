# CREATING DATASET of AGREEMENT IN ITALIAN
from itertools import product
from random import sample

def list_to_txt(l, filename):
    with open(filename, 'w', encoding="utf8") as f:
        for item in l:
            f.write("%s\n" % item)

det_sing = ["Un", "Il"]
det_plur = ["I", "Dei"]

noun_sing = "politico maggiordomo cameriere criminale ministro cuoco giudice cantante musicista regista soldato poeta parrucchiere barista sindaco presidente pilota camionista calciatore cacciatore".split()
noun_plur = "politici maggiordomi camerieri criminali ministri cuochi giudici cantanti musicisti registi soldato poeti parrucchieri baristi sindaci presidenti piloti camionisti calciatori cacciatori".split()


verb_sing = ["mangia una mela", "beve una bibita", "scrive un articolo", "gioca una partita", "saluta un amico", "assaggia un boccone", "compra un vestito", "colora un disegno", "esprime un'idea", "racconta una storia", "corre", "salta", "pensa", "gioca", "dorme", "muore", "cucina", "scappa", "esce", "cammina"] # 10 trans, 10 intrans
verb_plur = ["mangiano una mela", "bevono una bibita", "scrivono un articolo", "giocano una partita", "salutano un amico", "assaggiano un boccone", "comprano un vestito", "colorano un disegno", "esprimono un'idea", "raccontano una storia", "corrono", "saltano", "pensano", "giocano", "dormono", "muoiono", "cucinano", "scappano", "escono", "camminano"]

mod= ["di martedì", "molto spesso", "ogni lunedì", "in inverno", "in autunno", "in estate", "in primavera", "ogni mattina", "di sera", "di giorno", "a pranzo", "a cena", "al tramonto", "di notte", "in vacanza"]

a = list(list(w) for w in product(det_sing, noun_sing, verb_sing)) # 840 items
b = list(list(w) for w in product(det_plur, noun_plur, verb_plur))
c = list(list(w) for w in product(det_plur, noun_plur, verb_sing))
d = list(list(w) for w in product(det_sing, noun_sing, verb_plur))

def add_modifier(alist):
    out = []
    for item in alist:
        item.insert(2, sample(mod, 1)[0])
        out.append(" ".join(item)+".")
    return out

a1 = add_modifier(a) 
b1 = add_modifier(b) 
c1 = add_modifier(c) 
d1 = add_modifier(d) 

correct = a1+b1
incorrc = c1+d1

sentences, labels, positions = [], [], []
for correctsent, incorrectsent in zip(correct, incorrc):
    sentences.extend([correctsent, incorrectsent])
    labels.extend(["True", "False"])
    positions.extend([4,4])
    
list_to_txt(sentences, "it_prep.txt")
list_to_txt(labels, "it_prep_tags.txt")
list_to_txt(positions, "it_prep_position.txt")
