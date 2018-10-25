#read in both semantic and phonetic vectors
#create annoy index for each type
#create dict of line id -> line - line ids should be same across both annoy indices
#hmm, what to do about the ones that are different across semantic/phonetic? there are some that had a phonetic transc
#figure out what distance metric I want to use
#play with annoy parameters (50 and 100) to see how big/good I can make them - does it matter if they're different sizes?
from collections import Counter 
from annoy import AnnoyIndex
import numpy as np

def main():
    sem = AnnoyIndex(99, metric="euclidean")
    phon = AnnoyIndex(50, metric="euclidean")
    lines = list()
    lookup = dict()
    j = 0
    for i, row in enumerate(open("semantic_vectors.txt")):
        spl = row.find("[")
        line = row[0:spl-1]
        vec = row[spl+1:-2]
        vals = np.array([float(val) for val in vec.split(", ")])
        sem.add_item(i, vals)
        lines.append(line.lower())
        lookup[line.lower()] = i
        j += 1
        if j > 200000:
            break
    sem.build(100)
    print(sem.get_n_items())

    k = 0
    for i, row in enumerate(open("phonetic_vectors_every2.txt")):
        spl = row.find("' [")
        if spl > 0: #skip lines that don't have ' [
            line = row[0:spl+1]
            stripped_line = line[2:-1] #skip the b''
            vec = row[spl+3:-2]
            vals = np.array([float(val) for val in vec.split(", ")])
            if stripped_line.lower() in lookup:
                phon.add_item(i, vals) #problem: skipping is is bad
        k += 1
        if k > 200000:
            break
    phon.build(100)
    print(phon.get_n_items())


    print(phon.get_n_items())

    print("Semantic similarity")
    print([lines[i[0]] for i in nn_lookup(sem, sem.get_item_vector(lookup["he bows his little head."]))])
    print("Phonetic similarity")
    print([lines[i[0]] for i in nn_lookup(phon, phon.get_item_vector(lookup["he bows his little head."]))])

def nn_lookup(an, vec, n=10):
    res = an.get_nns_by_vector(vec, n)
    batches = []
    current_batch = []
    last_vec = None
    for item in res:
        if last_vec is None or an.get_item_vector(item) == last_vec:
            current_batch.append(item)
            last_vec = an.get_item_vector(item)
        else:
            batches.append(current_batch[:])
            current_batch = []
            last_vec = None
    if len(current_batch) > 0:
        batches.append(current_batch[:])
    output = []
    for batch in batches:
        output.append(batch)
    return output

if __name__ == '__main__':
    main()