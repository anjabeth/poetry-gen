import sys
from annoy import AnnoyIndex
import numpy as np
from datetime import datetime
import io

def main():
    t = AnnoyIndex(99, metric='euclidean')
    lines = dict()
    lookup = dict()

    prompt_word = input("Get the nearest semantic neighbors of: ")
    prompt_vec = find_glove_vector(prompt_word)

    print("loading...")
    index = 0
    for row in open("semantic_vectors_avgd.txt"):
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

    last_index = index+1
    t.add_item(last_index, prompt_vec) #add input vector so its neighbors can be calculated
    lookup[prompt_word] = [last_index]
    lines[last_index] = prompt_word

    t.build(100)
    print("done.")

    print("Num dict items: {0}".format(len(lookup)))
    print("Num list items: {0}".format(len(lines)))
    print("Num index items: {0}".format(t.get_n_items()))

    try:
        vec = prompt_vec
        print(nn_lookup(t, vec))
        print([lines[i[0]] for i in nn_lookup(t, vec)])
    except KeyError:
        print("not found")

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

def nn_lookup(an, vec, n=20):
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

if __name__ == "__main__":
    main()
