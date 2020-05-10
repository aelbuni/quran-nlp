'''
A class that enapsulate a machine learning model to build and train
a model on a digitized holy book of Quran. Train the model using word2vec
model (NGram).
'''

import pandas as pd
import nltk
import arabic_reshaper
import matplotlib.pyplot as plt 
from bidi.algorithm import get_display
from wordcloud import WordCloud
import re, sys
from gensim.models import Word2Vec

from typing import List, Dict

class QuranContextToWords:
    
    _word2vec_model = None
    _quran_data = None
    
    def __init__(self):
        ''' Load the quran book '''
        
        try:
            # Load Quran from csv into a dataframe
            self._quran_data = pd.read_csv('quran_nlp/data/arabic-original.csv', sep='|', header='infer');
        except:
            print('Failed to load the quran book with err')
    
    def process_quran_book(self) -> Word2Vec:
        ''' Pass Quran Verses on data preparation stages '''
        
        if self._quran_data is None:
            sys.exit('There is no data loaded! Please reinitiate object with the right source file')
        
        # Download Arabic stop words
        nltk.download('stopwords')
        
        # Extract Arabic stop words
        arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
        
        # Remove harakat from the verses to simplify the corpus
        self._quran_data['verse'] = self._quran_data['verse'].map(lambda x: re.sub('[ًٌٍَُِّۙ~ْۖۗ]', '', x))
                        
        # Tokinize words from verses and vectorize them
        self._quran_data['verse'] = self._quran_data['verse'].str.split()
        
        # Remove Arabic stop words
        self._quran_data['verse'] = self._quran_data['verse'].map(lambda x: [w for w in x if w not in arb_stopwords])
        
        # You can filter for one surah too if you want!
        verses = self._quran_data['verse'].values.tolist()

        self._build_model(verses)
        
        
    def _build_model(self, processed_verses: List[str], min_count=20, window=7, workers=8, alpha=0.22):
        ''' A private wrapper for the genism word2vec class '''
        self._word2vec_model = Word2Vec(processed_verses, min_count=15, window=7, workers=8, alpha=0.22)
        
    
    def _plot_word_cloud(self, word_list: List[str], word_frequency: Dict[str, float]):
        ''' Plot a WordCloud for top words that occured around the context word '''
        full_string = ' '.join(word_list)
        reshaped_text = arabic_reshaper.reshape(full_string)
        translated_text = get_display(reshaped_text)
        
        # Build the Arabic word cloud
        wordc = WordCloud(font_path='tahoma',background_color='white',width=2000,height=1000).generate(translated_text)
        wordc.fit_words(word_frequency)
        
        # Draw the word cloud
        plt.imshow(wordc)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        
        plt.show()
    
    
    def print_similar_word_cloud(self, one_word: str, topn: int):
        """Takes an Arabic word and print similar word cloud for top number of words {$topn}."""
        
        temp_tuple = self._word2vec_model.most_similar(positive=[one_word], negative=[], topn=topn)
        similar_words=[i[0] for i in temp_tuple]
        
        # Extract word weight to project it in the WordCloud plot
        word_frequency = {}
        for word_tuple in temp_tuple:
            reshaped_word = arabic_reshaper.reshape(word_tuple[0])
            key = get_display(reshaped_word)
            word_frequency[key] = word_tuple[1]
        
        self._plot_word_cloud(similar_words, word_frequency)
        