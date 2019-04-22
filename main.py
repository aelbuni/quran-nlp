
import pandas as pd
import nltk
import arabic_reshaper
import matplotlib.pyplot as plt 
import re
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from nltk.stem.isri import ISRIStemmer
from bidi.algorithm import get_display
from wordcloud import WordCloud

# Download Arabic stop words
nltk.download('stopwords')

# Extract Arabic stop words
arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))

# Initialize Arabic stemmer
st = ISRIStemmer()

# Load Quran from csv into a dataframe
df = pd.read_csv('data/arabic-original.csv', sep='|', header='infer');

# Remove harakat from the verses to simplify the corpus
df['verse'] = df['verse'].map(lambda x: re.sub('[ًٌٍَُِّۙ~ْۖۗ]', '', x))
                
# Tokinize words from verses and vectorize them
df['verse'] = df['verse'].str.split()

# Remove Arabic stop words
df['verse'] = df['verse'].map(lambda x: [w for w in x if w not in arb_stopwords])

# Exclude these words from the stemmer
stem_not = ['الله', 'لله', 'إلهكم', 'اله', 'لله', 'إلهكم', 'إله', 'بالله', 'ولله']

# [On/Off] Stemming the words to reduce dimensionality except stem_not list
# df['verse'] = df['verse'].map(lambda x: [w if w in stem_not else st.stem(w) for w in x])

# You can filter for one surah too if you want!
verses = df['verse'].values.tolist()

# train model
model = Word2Vec(verses, min_count=15, window=7, workers=8, alpha=0.22)
# summarize the loaded model

# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)

# Pass list of words as an argument
for i, word in enumerate(words):
    reshaped_text = arabic_reshaper.reshape(word)
    artext = get_display(reshaped_text)
    plt.annotate(artext, xy=(result[i, 0], result[i, 1]))
    #plt.show()

def print_word_cloud_ar(artext_list):
    """Takes a list of Arabic words to print cloud."""
    full_string = ' '.join(artext_list)
    reshaped_text = arabic_reshaper.reshape(full_string)
    artext = get_display(reshaped_text)
    
    # Build the Arabic word cloud
    # use KacstOne font for linux systems because the other fonts cause errors
    wordc = WordCloud(font_path='KacstOne',background_color='white',width=2000,height=1000).generate(artext)
    
    # Draw the word cloud
    plt.imshow(wordc) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.show()
    
    
def print_similar_word_cloud(one_word, topn):
    """Takes an Arabic word and print similar word cloud for top number of words {$topn}."""
    temp_list=model.wv.most_similar(positive=[one_word], negative=[], topn=topn)
    similar_words=[i[0] for i in temp_list]
    print_word_cloud_ar(similar_words)
# an arbtary example touse wordcloud 
print_similar_word_cloud("الحمد",50)
