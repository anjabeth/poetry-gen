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
    all_features = Counter()
    entries = []

    with open('transcribed_data.csv') as data:
        rdr = csv.reader(data)
        i = 0
        for row in rdr:
            if len(row) > 0: #skip any empty ones
                line = row[0]
                phones = row[1].split(" ")
                print("phones")
                print(phones)
                i += 1
                if i > 10:
                    break
                features = Counter(feature_bigrams(phones))
                entries.append((line, features))
                all_features.update(features.keys())

    print("Entries calculated at {0}".format(datetime.now().time()))

    filtfeatures = sorted([f for f, count in all_features.items() \
            if count >= 2])

    print("Feature count:", len(filtfeatures))
    print("Etarting PCA at {0}".format(datetime.now().time()))

    arr = np.array([normalize([i.get(j, 0) for j in filtfeatures]) \
            for line, i in entries])
    pca = PCA(n_components=50, whiten=True) #could play with # of components
    transformed = pca.fit_transform(arr)

    print("PCA finished at {0}".format(datetime.now().time()))

    with open('vectors.csv', mode='w') as out_file:
        out_writer = csv.writer(out_file, delimiter=',', quotechar='"')
        for i in range(len(entries)):
            line = entries[i][0]
            nums = [num for num in transformed[i]]
            out_writer.writerow([line.encode('utf-8'), nums])
    print("Written to file: {0}".format(datetime.now().time()))

if __name__ == '__main__':
    main()