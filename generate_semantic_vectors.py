import io
import csv
import nltk
import numpy as np
from collections import defaultdict
from datetime import datetime

word_vectors = defaultdict(lambda: None)

print("Loading Glove Vectors: {0}".format(datetime.now().time()))
with io.open("glove.42B.300d.txt", 'r', encoding='utf-8') as glove:
    for line in glove:
        entries = line.split(" ")
        word = entries[0]
        vector = np.array([float(n) for n in entries[1:-1]])
        word_vectors[word] = vector
lines_and_vecs = dict()

print("Processing Phonetic Vectors: {0}".format(datetime.now().time()))
with open('phonetic_vectors_every2_d100_reformatted.txt') as phonetic_vectors:
    i = 0
    num_entries = 0
    for row in phonetic_vectors:
        spl = row.find("@@@")
        raw_line = row[0:spl-1]
        line = raw_line[2:-1].lower() #skip the b''
        vec = row[spl+3:-1]
        vals = np.array([float(val) for val in vec.split(", ")])
        words = nltk.word_tokenize(line)
        has_all_words = True
        all_vecs = []
        num_words = 0
        for word in words:
            num_words += 1
            v = word_vectors[word]
            if v is None:
                has_all_words = False
            all_vecs.append(v)
        if has_all_words and num_words > 0:
            num_entries += 1
            if num_entries % 10000 == 0:
                print("num_entries is {0}".format(num_entries))
            avgd_vec = sum(all_vecs) / num_words
            summed_vec = sum(all_vecs)
            line_vec = 0.9 * avgd_vec + 0.1 * summed_vec
            lines_and_vecs[line] = line_vec
        i += 1
        if i % 10000 == 0:
            print("i is {0}".format(i))

print("Done Processing Phonetic Vectors: {0}".format(datetime.now().time()))

print("Writing to File: {0}".format(datetime.now().time()))
with open('semantic_vectors_42b_300d_weighted91.txt', mode='w') as out_file:
    for line, vec in lines_and_vecs.items():
        vals = ", ".join([str(val) for val in vec])
        out_file.write("{0} @@@ {1}\n".format(line, vals))
print("Written to file: {0}".format(datetime.now().time()))
