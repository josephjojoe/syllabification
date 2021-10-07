# Imports dependencies
import re
from itertools import chain

# The words are stored in mhyph.txt in the form 'git¥hub', where the yen character ('¥') denotes a syllable break:
# g i t ¥ h u b

# This script deletes duplicate words and creates a binary string showing where the syllable breaks are:
# g i t - h u b
# 0 0 1   0 0 0

# An example line in preprocessed.txt looks like 'github,001000'. This binary format is useful as we can use the 'binary_accuracy' 
# metric when training and evaluating our model.

# Flattens word list
with open("mhyph.txt", "r", errors="ignore") as f:
    li = list(chain.from_iterable([i.split() for i in f.readlines()]))

    
with open("preprocessed.txt", "w+") as f:
    for word in sorted(set(li)):

        # List holding undemarcated word (e.g. 'syllable') and corresponding binary ('00101000').
        writer = []
        writer.append(re.sub("¥", "", word))

        if not re.sub("¥", "", word).isalpha():
            continue # Jumps to next word
        
        # Converts to binary, where '1' denotes a syllable's start.
        word = list(re.sub("¥", "1", re.sub("[^¥]", "0", word)))
        # Removes extra zero before each '1'.
        one_index = [i for i, j in enumerate(word) if j == "1"]
        for i in range(0, len(one_index)):
            del word[one_index[i] - (i + 1)]
        writer.append("".join(word))

        f.write(writer[0] + "," + writer[1] + "\n")
