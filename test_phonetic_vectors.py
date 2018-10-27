import sys
from annoy import AnnoyIndex
import numpy as np

def main():
    t = AnnoyIndex(200, metric='euclidean')
    lines = list()
    lookup = dict()

    print("loading...")
    for i, row in enumerate(open("phonetic_vectors_every2_d200.txt")):
        spl = row.find("' [")
        spl2 = row.rfind("' [")
        if spl != spl2: #skip this one if there are weird extra brackets
            continue
        if spl > 0: #make sure there are brackets at all
            line = row[0:spl+1]
            stripped_line = line[2:-1].lower() #skip the b''
            vec = row[spl+3:-2]
            vals = np.array([float(val) for val in vec.split(", ")])
            if stripped_line in lookup:
                continue
            lookup[stripped_line] = i
            lines.append(stripped_line)
            t.add_item(i, vals)
        if i % 50000 == 0:
            print(stripped_line.lower())
            print("{0} vectors loaded".format(i))
    t.build(200)
    print("done.")

    print("Num dict items: {0}".format(len(lookup)))
    print("Num list items: {0}".format(len(lines)))
    print("Num index items: {0}".format(t.get_n_items()))

    try:
        vec = lookup["i think the very violets"]
        print(vec)
        print(t.get_item_vector(vec))
        print(nn_lookup(t, t.get_item_vector(vec)))
        print([lines[i[0]] for i in nn_lookup(t, t.get_item_vector(vec))])
    except KeyError:
        print("not found")

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

if __name__ == "__main__":
    main()
