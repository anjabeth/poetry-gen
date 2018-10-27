""" Code adapted from https://github.com/aparrish/phonetic-similarity-vectors/blob/master/generate.py """

#TO FIX: 
# usage: python generate.py <cmudict-0.7b >cmudict-0.7b-embeddings
# reads a CMUDict-formatted text file on standard input, prints embeddings
# for each word on standard output.

import sys
import csv
import re

import numpy as np
from collections import Counter
from datetime import datetime
from sklearn.decomposition import PCA
from featurephone import feature_bigrams

def normalize(vec):
    """Return unit vector for parameter vec.
    >>> normalize(np.array([3, 4]))
    array([ 0.6,  0.8])
    """
    if np.any(vec):
        norm = np.linalg.norm(vec)
        return vec / norm
    else:
        return vec

def main():
    print("Program Start: {0}".format(datetime.now().time()))

    all_features = Counter()
    entries = []

    with open('transcribed_data.csv') as data:
        rdr = csv.reader(data)
        i = 0
        num_entries = 0
        for row in rdr:
            if len(row) > 0: #skip any empty ones
                if i % 2 == 0: #6.3M lines total, but memory can't handle all of that, so try ~600K
                    line = row[0]
                    phones = [p for p in row[1].split(" ") if p not in ["-","!","+","/","#",":", "name", "abbrev"]] #ignore special ARPAbet symbols
                    if len(phones) < 2: #skip any with empty phones
                        continue
                    features = Counter(feature_bigrams(phones))
                    entries.append((line, features))
                    all_features.update(features.keys())
                    num_entries += 1
                    if num_entries % 10000 == 0:
                        print("num_entries is {0}".format(num_entries))
                i += 1
                if i % 20000 == 0:
                    print("i is {0}".format(i))

    print("Entries calculated at {0}".format(datetime.now().time()))
    print("Num entries {0}".format(len(entries)))

    filtfeatures = sorted([f for f, count in all_features.items() \
            if count >= 2])

    print("Feature count:", len(filtfeatures))
    print("Starting PCA at {0}".format(datetime.now().time()))

    arr = np.array([normalize([i.get(j, 0) for j in filtfeatures]) \
        for line, i in entries])
    pca = PCA(n_components=200, whiten=True) #could play with # of components
    transformed = pca.fit_transform(arr)

    print("PCA finished at {0}".format(datetime.now().time()))

    with open('phonetic_vectors_every2_d200.txt', mode='w') as out_file:
        for i in range(len(entries)):
            line = entries[i][0]
            nums = [num for num in transformed[i]]
            out_file.write("{0} {1}\n".format(line, nums))
    print("Written to file: {0}".format(datetime.now().time()))

if __name__ == '__main__':
    main()
