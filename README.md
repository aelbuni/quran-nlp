# Documentation

https://medium.com/@aelbuni/modeling-topics-in-quran-using-machine-learning-nlp-b88ca23fb44d

# quran-nlp
quran-nlp is an opensource project that aims to provide a boilerplate to help researchers to mine for knowledge from the Quran

# Usage example

```
from quran_nlp.quran_nlp import QuranContextToWords
import matplotlib.pyplot as plt 

# Instantiate a new object
contextToWords = QuranContextToWords()

# Start the learning process and cache the model for use
contextToWords.process_quran_book()

# Start using the model and play with visualization
# You can generate multiple plots through matlibplot as follows

plt.figure(1)
plt.subplot(211)
word = 'الجنة'
contextToWords.print_similar_word_cloud(word, 20)

plt.subplot(212)
word = 'الله'
contextToWords.print_similar_word_cloud(word, 20)
```
## Result Plot
Plotting two figures, the first for `Heaven`, and the second is `Allah`

![Quran NLP](sample.png "Quran ")
