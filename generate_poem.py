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
import io

lookup = dict()
slines = dict()
plines = dict()


def main():
    prompt_word = input("Choose a word to base your poem on: ")
    prompt_vec = find_glove_vector(prompt_word)
    sem, phon = build_annoy_indices(prompt_word, prompt_vec)

    print("Generating Poem: {0}".format(datetime.now().time()))
    print("***")

    sem_similar_lines, sem_distances = nn_lookup(sem, sem.get_item_vector(lookup[prompt_word][0]))
    print("semantically similar lines are: {0}".format([slines[i[0]] for i in sem_similar_lines]))
    print("distances are: {0}".format(sem_distances))
    first_stanza = []
    second_stanza = []
    poem = [first_stanza, second_stanza]

    print("***")
    i = 1 #skip the first one, since it will be the input word itself
    first_stanza_1_index = sem_similar_lines[i][0]
    while first_stanza_1_index > phon.get_n_items(): #not sure why phon has fewer items, but this gets around it
        i += 1
        first_stanza_1_index = sem_similar_lines[i][0]
    first_stanza_1 = slines[first_stanza_1_index]
    print("first line is {0}".format(first_stanza_1))
    first_stanza.append(first_stanza_1)
    phon_similar_lines_1, phon_distances = nn_lookup(phon, phon.get_item_vector(lookup[first_stanza_1][1]))
    print("phonetically similar lines are: {0}".format([plines[i[0]] for i in phon_similar_lines_1]))
    print("distances are: {0}".format(phon_distances))
    for j in phon_similar_lines_1:
        if plines[j[0]] == first_stanza_1:
            continue #skip the one that is the line itself
        first_stanza.append(plines[j[0]])

    print("***")
    i += 1
    second_stanza_1_index = sem_similar_lines[i][0]
    while second_stanza_1_index > phon.get_n_items(): #not sure why phon has fewer items, but this gets around it
        i += 1
        second_stanza_1_index = sem_similar_lines[i][0]
    second_stanza_1 = slines[second_stanza_1_index]
    print("first line is {0}".format(second_stanza_1))
    second_stanza.append(second_stanza_1)
    phon_similar_lines_2, phon_distances = nn_lookup(phon, phon.get_item_vector(lookup[second_stanza_1][1]))
    print("phonetically similar lines are: {0}".format([plines[i[0]] for i in phon_similar_lines_2]))
    print("distances are: {0}".format(phon_distances))
    for j in phon_similar_lines_2:
        if plines[j[0]] == second_stanza_1:
            continue #skip the one that is the line itself
        second_stanza.append(plines[j[0]])
    print("***")

    print("Done Generating Poem: {0}".format(datetime.now().time()))

    for stanza in poem:
        for line in stanza:
            print(line)
        print("\n")

    print("Done! : {0}".format(datetime.now().time()))


def find_glove_vector(input_word):
    print("Searching Glove Vectors: {0}".format(datetime.now().time()))
    with io.open("glove.6B.100d.txt", 'r', encoding='utf-8') as glove:
        for line in glove:
            entries = line.split(" ")
            word = entries[0]
            if word == input_word.lower():
                vector = np.array([float(n) for n in entries[1:-1]])
                return vector
    print("Sorry, word not found.")


def build_annoy_indices(input_word, input_vector):
    print("Building Annoy Indices: {0}".format(datetime.now().time()))
    sem = AnnoyIndex(99, metric="euclidean")
    phon = AnnoyIndex(100, metric="euclidean")

    index = 0
    print("Reading Data for Semantic Index: {0}".format(datetime.now().time()))
    for row in open("semantic_vectors.txt"):
        spl = row.find("[")
        line = row[0:spl-1].lower()
        vec = row[spl+1:-2]
        vals = np.array([float(val) for val in vec.split(", ")])
        if line not in lookup:
            sem.add_item(index, vals)
            slines[index] = line
            lookup[line] = [index]
            index += 1
        if index % 100000 == 0:
            print("......{0} vectors loaded.".format(index))

    last_index = index+1
    sem.add_item(last_index, input_vector) #add input vector so its neighbors can be calculated
    lookup[input_word] = [last_index]
    slines[last_index] = input_word

    print("Building Semantic Index: {0}".format(datetime.now().time()))
    sem.build(100)
    print("Built: {0}".format(datetime.now().time()))
    print("Num items in semantic index: {0}".format(sem.get_n_items()))

    print("Reading Data for Phonetic Index: {0}".format(datetime.now().time()))
    pindex = 0
    for row in open("phonetic_vectors_every2_d100.txt"):
        spl = row.find("' [")
        spl2 = row.rfind("' [")
        if spl > 0 and spl == spl2: #make sure brackets are correct
            line = row[0:spl+1]
            stripped_line = line[2:-1] #skip the b''
            vec = row[spl+3:-2]
            vals = np.array([float(val) for val in vec.split(", ")])
            if stripped_line.lower() in lookup:
                phon.add_item(pindex, vals)
                lookup[stripped_line.lower()].append(pindex)
                plines[pindex] = stripped_line.lower()
                pindex += 1
            if pindex % 100000 == 0:
                print("......{0} vectors loaded.".format(pindex))

    print("Building Phonetic Index: {0}".format(datetime.now().time()))
    phon.build(100)
    print("Built: {0}".format(datetime.now().time()))
    print("Num items in phonetic index: {0}".format(phon.get_n_items()))

    print("Done Building Annoy Indices: {0}".format(datetime.now().time()))
    return sem, phon


def nn_lookup(an, vec, n=20):
    res, distances = an.get_nns_by_vector(vec, n, include_distances=True)
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
    return output, distances

if __name__ == '__main__':
    main()
