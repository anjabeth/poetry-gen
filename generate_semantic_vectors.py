import io
import csv
import nltk
import numpy as np
from collections import defaultdict
from datetime import datetime
#TO DO
# Read in whole glove file (~331 MB) to create dictionary to look up matching vectors
# Read in phonetic vectors one line at a time so it doesn't overwhelm the memory
# Only keep the text part of the phonetic vector, that's the part we care about
# Might need to tokenize the text differently to match glove? They include punctuation, 's is a separate word, etc. wonder if they have inflected forms there or not - I think probably yes

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
with open('phonetic_vectors_1p5M_every2.csv') as phonetic_vectors:
    rdr = csv.reader(phonetic_vectors)
    i = 0
    for row in rdr:
        orig_line = row[0]
        if i % 100 == 0:
            print(orig_line)
        #strip the bs from the strings - might need to disable this later if I regenerate the data
        j = line.find('\'')
        k = line.rfind('\'')
        stripped_line = orig_line[j+1:k]
        if i % 100 == 0:
            print(stripped_line)
        line = stripped_line.lower()
        words = nltk.word_tokenize(line)
        has_all_words = True
        all_vecs = []
        for word in words:
            v = word_vectors[word]
            if v is None:
                has_all_words = False
            all_vecs.append(v)
        if has_all_words:
            line_vec = sum(all_vecs)
            lines_and_vecs[stripped_line] = line_vec
        i += 1
        if i % 100 == 0:
            print("i is {0}".format(i))
        if i > 5000:
            break
print("Done Processing Phonetic Vectors: {0}".format(datetime.now().time()))
#TODO: maybe write to text file instead of csv? Would then have to rerun the phonetic vectors script so they match, but it might be worth it to avoid the weird parsing issues
print("Writing to File: {0}".format(datetime.now().time()))
with open('semantic_vectors.txt', mode='w') as out_file:
    for line, vec in lines_and_vecs.items():
        out_file.write("{0} {1}".format(line, vec.astype(str)))
print("Written to file: {0}".format(datetime.now().time()))