import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from scipy.sparse import hstack
from nltk.corpus import stopwords
import umap.umap_ as umap
from textblob import TextBlob
import string


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class Pipeline:
    """
    A class used to represent an ETL pipeline for transaction data.

    ...

    Attributes
    ----------
    None

    Methods
    -------      
    preprocess(data):
        Transforms the data by cleaning and preparing it for modeling.
        
    encode(data, filename):
        Loads the transformed data into a new CSV file.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.vocab = {}
        self.umap_model = umap.UMAP(n_components=50, n_neighbors=15, metric='cosine')
        self.w2v_model = None
        
    def preprocess(self, corpus: pd.Series):
        corpus = corpus.fillna('')  
        corpus = corpus.astype(str)
        
        corpus = corpus.apply(lambda x: [word for word in word_tokenize(x.lower()) if word not in stop_words and word.isalpha()])
        
        if not self.vocab:  
            all_tokens = [token for sublist in corpus for token in sublist]
            unique_tokens = sorted(set(all_tokens))
            self.vocab = {word: idx for idx, word in enumerate(unique_tokens)}
        else:  
            new_tokens = {token for sublist in corpus for token in sublist if token not in self.vocab}
            if new_tokens:
                start_idx = len(self.vocab)
                self.vocab.update({token: idx for idx, token in enumerate(new_tokens, start=start_idx)})
        
        corpus = corpus.apply(lambda tokens: [self.vocab[token] for token in tokens if token in self.vocab])
        
        return corpus
    
    def transform(self, data):
        """
        Transforms the data by cleaning and preparing it for modeling.

        Parameters
        ----------
        data : DataFrame
            The raw data to transform.

        Returns
        -------
        transformed_data : DataFrame
            The data after transformation.
        """ 
        # Text length as a feature
        data['text'] = data['text'].fillna('')
        data['review_title'] = data['review_title'].fillna('')
        data['review_length'] = data['text'].apply(len)
        data['word_count'] = data['text'].apply(lambda x: len(x.split()))
        data['unique_word_count'] = data['text'].apply(lambda x: len(set(x.split())))
        data['punctuation_count'] = data['text'].apply(lambda x: sum(1 for char in x if char in string.punctuation))
        
        data['polarity'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        data['subjectivity'] = data['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        encoded_text = self.encode(data['text'], method='tfidf')
        encoded_title = self.encode(data['review_title'], method='tfidf')
        
        reduced_text = self.umap_model.fit_transform(encoded_text.toarray())
        reduced_title = self.umap_model.fit_transform(encoded_title.toarray())
        
        data['verified_purchase'] = data['verified_purchase'].astype(int)
        
        scaler = StandardScaler()
        data['helpful_vote'] = scaler.fit_transform(data[['helpful_vote']])

        # Drop original columns not needed for modeling
        data.drop(['Unnamed: 0', 'bought_together', 'images_x', 'asin', 'parent_asin', 'timestamp', 'movie_title', 'subtitle', 'rating_number', 'features', 'description',
                   'images_y', 'videos', 'store', 'author', 'details', 'categories', 'main_category', 'text', 'review_title', 'user_id', 'average_rating', 'price'], axis=1, inplace=True, errors='ignore')
        
        encoded_text_df = pd.DataFrame(reduced_text, index=data.index)
        encoded_title_df = pd.DataFrame(reduced_title, index=data.index)
        transformed_data = pd.concat([data, encoded_text_df, encoded_title_df], axis=1)

        return transformed_data
        
        
    
    def encode(self, corpus: pd.Series, method='bow'):
        if method not in {'bow', 'tfidf', 'word2vec'}:
            raise ValueError(f"Unsupported encoding method: {method}")
        
        corpus = corpus.fillna('')  
        corpus = corpus.astype(str)
        corpus = corpus.apply(lambda x: [word for word in word_tokenize(x.lower()) if word not in stop_words and word.isalpha()])
        
        if method == 'word2vec':
            if self.w2v_model:
                model = self.w2v_model;
            else:
                model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=2, workers=4, epochs=10)
                self.w2c_model = model

            def document_vector(doc):
                doc = [word for word in doc if word in model.wv.index_to_key]
                return np.mean(model.wv[doc], axis=0) if doc else np.zeros((model.vector_size,))

            encoded_data = np.vstack([document_vector(doc) for doc in corpus])
            return encoded_data
        else:
            if isinstance(corpus.iloc[0], list):  
                str_corpus = corpus.apply(lambda x: ' '.join(map(str, x)))
            else:
                str_corpus = corpus


            if self.vectorizer and hasattr(self.vectorizer, 'method') and self.vectorizer.method == method:
                encoded_data = self.vectorizer.transform(str_corpus)    
            else:
                if method == 'bow':
                    self.vectorizer = CountVectorizer()
                elif method == 'tfidf':
                    self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                
                encoded_data = self.vectorizer.fit_transform(str_corpus)
                self.vectorizer.method = method  
                
        return encoded_data