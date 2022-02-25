# CONVERT EMBEDDING FILES INTO PICKLE FILES
import pickle

import numpy as np

aligned_embeddings_es = dict()

with open('../data/multi-lingual-aligned-vectors/wiki.pl.align.vec', 'r') as f:
    f.readline()
    line = f.readline()
    while line:
        splitted_words = line.split()

        try:
            np.float16(splitted_words[1:])
        except:
            pass

        token = line[:line.find(splitted_words[-300])-1].replace(chr(160), ' ').strip()
        embeddings = np.float16(splitted_words[-300:])

        if len(embeddings) != 300:
            AttributeError('Vector length did not match!')

        aligned_embeddings_es[token] = embeddings
        line = f.readline()

with open('../data/aligned_embeddings_pl.pkl', 'wb') as f:
    pickle.dump(aligned_embeddings_es, f)
