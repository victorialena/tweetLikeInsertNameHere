import numpy as np
import matplotlib.pyplot as plt
import string

punction_filter = '?!"“$”&\'()*+,-./:;<=>[\\]^_`{|}~'
alt_punction = '"“$”&\'()*+,-/:;<=>[\\]^_`{|}~'
eos_punction = '!?.'

def get_words(message, punc = punction_filter, lower = False):
    """
    Split a message into a list of normalized words. We remove the punction, 
    except for #, @ and % - as these are common traits in tweets -, and we keep 
    the original capitalization.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    if lower:
        return message.translate(str.maketrans('','',punc)).lower().split(' ')
    return message.translate(str.maketrans('','',punc)).split(' ')

def filter_junk(message):
    words = [x.strip() for x in get_words(message, alt_punction) if not x.startswith('http') and x != 'amp' and x !=' ']
    msg = ' '.join(words)
    mapping =  {'!': '.', '?': '.', 
                "U.S": "USA", "E.U": "EU", 'D.C': 'DC', 'P.M': 'PM', 'A.M': 'AM', 
                '. #':' #', '. @':' @'}

    for key, value in mapping.items():
        msg = msg.replace(key, value)
        
    return " ".join(msg.split())

def create_dictionary(messages, n = 1, prefix = ''):
    """
    Create a dictionary of word to indices using the provided
    training messages. Rare words are often not useful for modeling. 
    We can filter for words that occur less than n times.

    Args:
        messages: A list of strings containing SMS messages
        prefix: '@' to find all usertags for example

    Returns:
        A python dict mapping words to integers.
    """

    duden = {} # Duden is a german dictionary, the equivalent to the Oxford English Dictionary
    for msg in messages:
        for word in list(np.unique(get_words(msg, lower=True))):
            if not word.startswith(prefix):
                continue
            if word not in duden.keys():
                duden[word] = 1
            else:
                duden[word] += 1
    
    for key, count in sorted(duden.items()):
        if count < n:
            del duden[key]
    for (i, key) in enumerate(sorted(duden.keys())):
        duden[key] = i
    
    return duden

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    matrix = np.zeros((len(messages), len(word_dictionary)))
    for (i, msg) in enumerate(messages):
        for w in get_words(msg):
            if w in word_dictionary.keys():
                matrix[i][word_dictionary[w]] += 1
    return matrix

def pl0t(df, log=False, grid=False, filename = ''):
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(211)
    ax.set_ylabel('Loss', fontsize=12)
    ax.plot(df.epoch, df.loss, 'r', label = 'training')
    ax.plot(df.epoch, df.val_loss, 'r--', label = 'validation')
    ax.legend(fontsize=11)

    ax2= fig.add_subplot(212)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.plot(df.epoch, df.acc, 'b', label = 'training')
    ax2.plot(df.epoch, df.val_acc, 'b--', label = 'validation')
    ax2.legend(fontsize=11)

    ax2.set_xlabel('Epoch', fontsize=12)
    if log:
        ax2.set_xscale('log')
        ax.set_xscale('log')
    if grid:
        ax.grid(True)
        ax2.grid(True)

    ax.set_title('Training History', fontsize=13)
    
    if filename:
        fig.savefig(filename+'.png')

    return fig