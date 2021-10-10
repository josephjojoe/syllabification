# syllabification
GRU-based neural network with Inception modules and an optional Linear Chain CRF that splits words into syllables.

### Model architecture
Tokenized data is passed into an Embedding layer and then into two 'stems' - the first stem contains a stack of 3 bidirectional GRU layers (2 x 256 = 512 units each) and the second stem uses a 1D implementation (since it is being used for sequence data, not images) of the Inception v2 module architecture. The stem outputs are concatenated and passed through two TimeDistributed layers then GlobalMaxPool1D is applied - finally there is a Dense layer with 15 units, outputting a binary string that is a prediction of the syllable breaks in the input data.

Tanh is the activation function used in the Inception module layers and Relu has been applied to the TimeDistributed layers - L2 regularisation has been introduced throughout the GRU and Inception stems to combat overfitting along with dropout of 0.1 in the GRU layers, although work is ongoing on modifying the hyperparameters and experimenting with novel architectutures that may lessen the need for this.

### Data format
Data is stored in a text file (`/dataset/preprocessed.txt`) with each line in the form `word,binary`:
```
python,010000
```
This is a compact representation of the syllable breaks in the word that allows the problem of syllabification to be framed as a multi-label classification task.
```
p y - t h o n
0 1   0 0 0 0
```

### Statistics
• Peak validation binary accuracy of 98.24% on the Moby Hyphenator II dataset.
