from collections import Counter 
from annoy import AnnoyIndex
import numpy as np
from datetime import datetime
import io
import nltk

lookup = dict()
slines = dict()
plines = dict()

def main():
    num_poems = int(input("How many poems would you like to generate? "))
    prompt_words = []
    prompt_vecs = []
    for i in range(num_poems):
        prompt_words.append(input("Choose a word or phrase to base poem {0} on: ".format(i+1)).lower())
    prompt_vecs = find_glove_vectors(prompt_words)

    sem, phon = build_annoy_indices(prompt_words, prompt_vecs)

    for i in range(num_poems):
        create_poem(sem, phon, prompt_words[i])

    print("All poems generated at: {0}".format(datetime.now().time()))


def create_stanza(sem_lines, idx, phon, unique_words=False):
    stanza = []
    stanza_line1_idx = sem_lines[idx][0]
    while stanza_line1_idx > phon.get_n_items(): #not sure why phon has fewer items, but this gets around it
        idx += 1
        stanza_line1_idx = sem_lines[idx][0]
    stanza_line1 = slines[stanza_line1_idx]
    line1_words = stanza_line1.split(" ")
    stanza.append(stanza_line1)
    phon_similar_lines, phon_distances = nn_lookup(phon, phon.get_item_vector(lookup[stanza_line1][1]))
    index = 1
    orig_distance = phon_distances[index] #distance to first phonetic neighbor not itself
    while orig_distance == 0:
        index += 1
        orig_distance = phon_distances[index]
    for j in range(1, len(phon_similar_lines)): 
        k = phon_similar_lines[j]
        perc = phon_distances[j] / orig_distance
        if perc < 1.1 or perc > 1.4:
            continue
        if unique_words:
            #eliminate if contains same word as first line (approximate, since using naive space split to define words)
            contains = False
            line_words = plines[k[0]].split(" ")
            for word in line_words:
                if word in line1_words:
                    contains = True
            if contains:
                continue

        stanza.append(plines[k[0]])

    return stanza, idx #return idx so we know where to start the second stanza


def create_poem(sem, phon, prompt_word):
    print("Generating Poem: {0}".format(datetime.now().time()))

    sem_similar_lines, sem_distances = nn_lookup(sem, sem.get_item_vector(lookup[prompt_word][0]))
    # print("semantically similar lines are: {0}".format([slines[i[0]] for i in sem_similar_lines]))
    # print("distances are: {0}".format(sem_distances))
    sem_idx = 1
    first_stanza, new_idx = create_stanza(sem_similar_lines, sem_idx, phon)
    new_idx += 1
    second_stanza, throwaway = create_stanza(sem_similar_lines, new_idx, phon)

    poem = [first_stanza, second_stanza]

    print("Done Generating Poem: {0}".format(datetime.now().time()))

    for stanza in poem:
        for line in stanza:
            print(line)
        print("\n")


def find_glove_vectors(input_words):
    print("Searching Glove Vectors: {0}".format(datetime.now().time()))
    all_vectors = []
    for w_or_p in input_words:
        words = nltk.word_tokenize(w_or_p)
        num_words = len(words)
        matching_vectors = [None] * len(words)
        found = [0] * len(words)
        with io.open("glove.6B.100d.txt", 'r', encoding='utf-8') as glove:
            for line in glove:
                entries = line.split(" ")
                word = entries[0]
                if word in words:
                    idx = words.index(word)
                    vector = np.array([float(n) for n in entries[1:-1]])
                    matching_vectors[idx] = vector
                    found[idx] = 1
        if all(found):
            avgd_vec = sum(matching_vectors) / num_words
            summed_vec = sum(matching_vectors)
            total_vec = 0.9 * avgd_vec + 0.1 * summed_vec
            all_vectors.append(total_vec)
        else:
            not_found_index = found.index(0)
            print(words)
            print("Sorry, one of your prompt words {0} could not be found.".format(words[not_found_index]))
            return
    return all_vectors


def build_annoy_indices(input_words, input_vectors):
    print("Building Annoy Indices: {0}".format(datetime.now().time()))
    sem = AnnoyIndex(99, metric="euclidean")
    phon = AnnoyIndex(100, metric="euclidean")

    index = 0
    print("Reading Data for Semantic Index: {0}".format(datetime.now().time()))
    for row in open("semantic_vectors_weighted82.txt"):
        spl = row.find("@@@")
        line = row[0:spl-1].lower()
        vec = row[spl+3:-1]
        vals = np.array([float(val) for val in vec.split(", ")])
        if line not in lookup:
            sem.add_item(index, vals)
            slines[index] = line
            lookup[line] = [index]
            index += 1
        if index % 100000 == 0:
            print("......{0} vectors loaded.".format(index))

    last_index = index+1
    for i in range(len(input_words)):
        sem.add_item(last_index, input_vectors[i]) #add input vector so its neighbors can be calculated
        lookup[input_words[i]] = [last_index]
        slines[last_index] = input_words[i]
        last_index += 1

    print("Building Semantic Index: {0}".format(datetime.now().time()))
    sem.build(150)
    print("Built: {0}".format(datetime.now().time()))
    print("Num items in semantic index: {0}".format(sem.get_n_items()))

    print("Reading Data for Phonetic Index: {0}".format(datetime.now().time()))
    pindex = 0
    for row in open("phonetic_vectors_every2_d100_reformatted.txt"):
        spl = row.find("@@@")
        line = row[0:spl-1]
        stripped_line = line[2:-1].lower() #skip the b''
        vec = row[spl+3:-1]
        vals = np.array([float(val) for val in vec.split(", ")])
        if stripped_line in lookup:
            phon.add_item(pindex, vals)
            lookup[stripped_line].append(pindex)
            plines[pindex] = stripped_line
            pindex += 1
        if pindex % 100000 == 0:
            print("......{0} vectors loaded.".format(pindex))

    print("Building Phonetic Index: {0}".format(datetime.now().time()))
    phon.build(150)
    print("Built: {0}".format(datetime.now().time()))
    print("Num items in phonetic index: {0}".format(phon.get_n_items()))

    print("Done Building Annoy Indices: {0}".format(datetime.now().time()))
    return sem, phon


def nn_lookup(an, vec, n=30):
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
