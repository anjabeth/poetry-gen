import sys
from annoy import AnnoyIndex
import numpy as np

t = AnnoyIndex(50, metric='euclidean')
words = list()
lookup = dict()

print("loading...")
for i, row in enumerate(open("phonetic_vectors_every2.txt")):
    spl = row.find("' [")
    if spl > 0: #skip lines that don't have ' [
        line = row[0:spl+1]
        stripped_line = line[2:-1] #skip the b''
        vec = row[spl+3:-2]
        vals = np.array([float(val) for val in vec.split(", ")])
        if stripped_line.lower() in lookup:
            phon.add_item(i, vals) #problem: skipping is is bad
            lookup[stripped_line.lower()] = i
            lines.append(stripped_line.lower())
    if i % 50000 == 0:
        print("{0} vectors loaded".format(i))
t.build(50)
print("done.")

try:
    vec = lookup["he bows his little head."]
    print([words[i[0]] for i in t.get_nns_by_vector(vec, 10)])
except KeyError:
    print("not found")