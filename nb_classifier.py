import numpy as np
import pandas as pd

from naive_bayes import NBmodel
from utils import *

#prepping data
train_messages = pd.DataFrame({'flag' : 0, 'tweet':pd.read_csv("clean_mixed_tweets.csv").tweet})
trump = pd.DataFrame({'flag' : 1, 'tweet':
                      pd.read_csv("clean_mixed_tweets_trump.csv").tweet.sample(frac=0.6, replace=False)})
train_messages = train_messages.append(trump, ignore_index=True).sample(frac=1., replace=False).reset_index(drop=True)

#validation partition ~ 10%
n_train = round(train_messages.shape[0]*0.9)

tweets, labels = train_messages.tweet[:n_train], train_messages.flag[:n_train]
test_tweets, test_labels = train_messages.tweet[n_train:], train_messages.flag[n_train:]

dictionary = create_dictionary(tweets, 5)
print('Size of dictionary: ', len(dictionary))

train_matrix = transform_text(tweets, dictionary)
test_matrix = transform_text(test_tweets, dictionary)

nb_model = NBmodel()
nb_model.fit(train_matrix, labels)
nb_predictions = nb_model.predict(test_matrix)

nb_accuracy = np.mean(nb_predictions == test_labels)
print('Naive Bayes had an accuracy of {} on the testing set'.format(nb_accuracy))

top_n = 10
top_N_words = nb_model.get_topN(dictionary, top_n)
print('The top', top_n, 'indicative words for Naive Bayes are: ', top_N_words)

tp = sum(nb_predictions) - fp
fn = abs(sum(np.minimum(nb_predictions - test_labels, 0)))
fp = sum(np.maximum(nb_predictions - test_labels, 0))
tn = len(nb_predictions) - tp - fn - fp
print('precision: ', tp/(tp+fp), ', recall:' , tp/(tp+fn))