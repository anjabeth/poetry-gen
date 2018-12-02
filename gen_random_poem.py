import sys
from annoy import AnnoyIndex
import numpy as np
from datetime import datetime
import io
import random

def main():
    t = AnnoyIndex(99, metric='euclidean')
    lines = dict()
    lookup = dict()

    print("loading...")
    index = 0
    for row in open("semantic_vectors_weighted91.txt"):
        spl = row.find("@@@")
        line = row[0:spl-1].lower()
        vec = row[spl+3:-1]
        vals = np.array([float(val) for val in vec.split(", ")])
        if line in lookup:
            continue
        t.add_item(index, vals)
        lines[index] = line
        lookup[line] = [index]
        index += 1
        if index % 50000 == 0:
            print(line)
            print("{0} vectors loaded".format(index))
    print("building")
    t.build(100)
    print("done.")

    nums1 = [random.randint(1, t.get_n_items()) for i in range(5)]
    nums2 = [random.randint(1, t.get_n_items()) for i in range(5)]
    
    poem = [nums1, nums2]

    for s in poem:
        for line in s:
            print(lines[line])
        print("\n")


if __name__ == "__main__":
    main()
