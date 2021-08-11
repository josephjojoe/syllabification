# Imports dependencies
import re
from itertools import chain


# Flattens word list
with open("mhyph.txt", "r", errors="ignore") as f:
    li = list(chain.from_iterable([i.split() for i in f.readlines()]))

    
with open("preprocessed.txt", "w+") as f:
    for word in sorted(set(li)):

        # List holding undemarcated word (e.g. 'syllable') and corresponding binary ('00101000').
        writer = []
        writer.append(re.sub("¥", "", word))
        
        # Converts to binary, where '1' denotes a syllable's start.
        word = list(re.sub("¥", "1", re.sub("[^¥]", "0", word)))
        # Removes extra zero before each '1'.
        one_index = [i for i, j in enumerate(word) if j == "1"]
        for i in range(0, len(one_index)):
            del word[one_index[i] - (i + 1)]
        writer.append("".join(word))

        f.write(writer[0] + "," + writer[1] + "\n")
