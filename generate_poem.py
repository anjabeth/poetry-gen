#read in both semantic and phonetic vectors
#create annoy index for each type
#create dict of line id -> line - line ids should be same across both annoy indices
#hmm, what to do about the ones that are different across semantic/phonetic? there are some that had a phonetic transc
#figure out what distance metric I want to use
#play with annoy parameters (50 and 100) to see how big/good I can make them - does it matter if they're different sizes?
#maybe use pickle to save the indices (minus the prompt) so that they don't have to be recreated every time?
from collections import Counter 
from annoy import AnnoyIndex
import numpy as np
from datetime import datetime


def main():
    prompt_word = raw_input("Choose a word to base your poem on: ")
    prompt_vec = find_glove_vector(prompt_word)
    sem, phon = build_annoy_indices(prompt_word, prompt_vec)
    

    print("Semantic similarity")
    print([lines[i[0]] for i in nn_lookup(sem, sem.get_item_vector(lookup["he bows his little head."]))])
    print("Phonetic similarity")
    print([lines[i[0]] for i in nn_lookup(phon, phon.get_item_vector(lookup["he bows his little head."]))])



def find_glove_vector(input_word):
    print("Searching Glove Vectors: {0}".format(datetime.now().time()))
    with io.open("glove.6B.100d.txt", 'r', encoding='utf-8') as glove:
        for line in glove:
            entries = line.split(" ")
            word = entries[0]
            if word == input_word.lower():
                vector = np.array([float(n) for n in entries[1:-1]])
                return word, vector
    print("Sorry, word not found.")


def build_annoy_indices(input_word, input_vector):
    print("Building Annoy Indices: {0}".format(datetime.now().time()))
    sem = AnnoyIndex(99, metric="euclidean")
    phon = AnnoyIndex(100, metric="euclidean")
    lines = list()
    lookup = dict()

    print("Building Semantic Index: {0}".format(datetime.now().time()))
    for i, row in enumerate(open("semantic_vectors.txt")):
        spl = row.find("[")
        line = row[0:spl-1].lower()
        vec = row[spl+1:-2]
        vals = np.array([float(val) for val in vec.split(", ")])
        if line in lookup:
            continue
        sem.add_item(i, vals)
        lines.append(line)
        lookup[line] = i
        if i % 100000 == 0:
            print("......{i} vectors loaded.")

    print("At end of loading i is {0}".format(i))
    sem.add_item(i+1, input_vector) #add input vector so its neighbors can be calculated
    sem.build(100)
    print(sem.get_n_items())

    print("Building Phonetic Index: {0}".format(datetime.now().time()))
    for i, row in enumerate(open("phonetic_vectors_every2_d100.txt")):
        spl = row.find("' [")
        if spl > 0: #skip lines that don't have ' [
            line = row[0:spl+1]
            stripped_line = line[2:-1] #skip the b''
            vec = row[spl+3:-2]
            vals = np.array([float(val) for val in vec.split(", ")])
            if stripped_line.lower() in lookup:
                phon.add_item(i, vals) #problem: skipping is is bad
            if i % 100000 == 0:
                print("......{i} vectors loaded.")
    phon.build(100)

    print("Done Building Annoy Indices: {0}".format(datetime.now().time()))
    return sem, phon


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
