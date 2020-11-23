import numpy as np
import pandas

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 

def create_model(total_words, max_sq_len, verbose=True):
    model = tf.keras.models.Sequential([
        Embedding(total_words, 150, input_length=max_sq_len-1),
        #Bidirectional(LSTM(256, return_sequences = True)),
        #Dropout(0.2),
        LSTM(256, return_sequences=True),
        #Dropout(0.2),
        LSTM(256),
        Dense(total_words/8, activation='selu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(total_words, activation='softmax')
      ])
    model.compile(optimizer=Adam(learning_rate=0.0008),
                loss='categorical_crossentropy',#tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', 'categorical_crossentropy'])#[tf.metrics.SparseCategoricalAccuracy()])
    if verbose:
        print(model.summary())
    return model

"""
old model = tf.keras.models.Sequential([
        Embedding(total_words, 100, input_length=max_sequence_len-1),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(total_words, activation='softmax')
      ])
"""

"""
old model = tf.keras.models.Sequential([
        Embedding(total_words, 300, input_length=max_sq_len-1),
        Bidirectional(LSTM(256, return_sequences = True)),
        Dropout(0.2),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dense(total_words/8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        """

def tokenize(file_name = "clean_tweets.csv", num_words = None):
    tweets = pandas.read_csv(file_name)
    punction_filter = '?!"“$”&\'()*+,-./:;<=>[\\]^_`{|}~'
    tokenizer = Tokenizer(num_words=num_words, filters=punction_filter, lower=True,
        split=' ', char_level=False)
    tokenizer.fit_on_texts(tweets.tweet) #tokenizer.fit_on_texts(trump_dict.keys())
    total_words = len(tokenizer.word_index) + 1
    print("total words: ", total_words, "\n")
    return tokenizer, total_words, tweets

def generate_seq(tokenizer, model, total_words, seed_text = "", seq_len = 10):
    next_words = seq_len - len(seed_text.split(' '))
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_len-1, padding='pre')
        probs = model.predict(token_list, verbose=0)
        predicted = np.random.choice(total_words, 1, p=probs.flatten())
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text