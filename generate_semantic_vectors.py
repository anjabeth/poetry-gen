import io
import csv
import nltk
import numpy as np
from collections import defaultdict
from datetime import datetime

word_vectors = defaultdict(lambda: None)

print("Loading Glove Vectors: {0}".format(datetime.now().time()))
with io.open("glove.6B.100d.txt", 'r', encoding='utf-8') as glove:
    for line in glove:
        entries = line.split(" ")
        word = entries[0]
        vector = np.array([float(n) for n in entries[1:-1]])
        word_vectors[word] = vector
lines_and_vecs = dict()

print("Processing Phonetic Vectors: {0}".format(datetime.now().time()))
with open('phonetic_vectors_every2.txt') as phonetic_vectors:
    i = 0
    num_entries = 0
    for row in phonetic_vectors:
        spl = row.find("[")
        line = row[0:spl-1]
        stripped_line = line[2:-1] #skip the b''
        vec = row[spl+1:-1]
        line = stripped_line.lower()
        words = nltk.word_tokenize(line)
        has_all_words = True
        all_vecs = []
        for word in words:
            v = word_vectors[word]
            if v is None:
                has_all_words = False
            all_vecs.append(v)
        if has_all_words and len(all_vecs) > 1:
            num_entries += 1
            if num_entries % 10000 == 0:
                print("num_entries is {0}".format(num_entries))
            line_vec = sum(all_vecs).tolist()
            lines_and_vecs[stripped_line] = line_vec

        i += 1
        if i % 10000 == 0:
            print("i is {0}".format(i))
print("Done Processing Phonetic Vectors: {0}".format(datetime.now().time()))
print("Writing to File: {0}".format(datetime.now().time()))
with open('semantic_vectors.txt', mode='w') as out_file:
    for line, vec in lines_and_vecs.items():
        out_file.write("{0} {1}\n".format(line, vec))
print("Written to file: {0}".format(datetime.now().time()))
