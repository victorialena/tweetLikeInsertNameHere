import numpy as np
import pandas

import matplotlib.pyplot as plt

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

df = pd.read_csv("trump_15k.csv")
scores = []
compound = []
for tweet in df.tweet.sample(frac=0.6, replace=False):
    score = sid.polarity_scores(tweet)
    scores.append(score)
    compound.append(score['compound'])
    
    
n_bis= 20
y, x = np.histogram(compound, bins)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.bar(x.tolist(),y.tolist(), width = 0.11)
plt.savefig('sentiment.png', dpi=200, bbox_inches = "tight")
plt.show()