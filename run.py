import os
from model import *

tokenizer, total_words, tweets = tokenize("clean_trump_15k.csv", 10000)

# create input sequences using list of tokens
input_sequences = []
for tweet in tweets.tweet:
    token_list = tokenizer.texts_to_sequences([tweet])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[max(0, i+1-10):i+1]
        input_sequences.append(n_gram_sequence)
print(len(input_sequences))
np.random.shuffle(input_sequences)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
min_sequence_len = min([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
print("sequence length max: ", max_sequence_len, ", min: ", min_sequence_len, "\n")

# create predictors and label
predictors, labels = input_sequences[:,:-1],input_sequences[:,-1]
labels = ku.to_categorical(labels, num_classes=total_words)

# Create a basic model instance
model = create_model(total_words, max_sequence_len)

folder = "training_8/"
checkpoint_path = folder+"cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(predictors, labels, batch_size=128, epochs=100, verbose=1, 
                    validation_split=0.05,           
                    #validation_data=(val_predictors, val_labels), 
                    callbacks=[cp_callback])

hist_df = pandas.DataFrame(history.history) 
hist_csv_file = folder+'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

f = open(folder+'sample_sequences.txt', 'w+')
for i in range(10):
    f.write(generate_seq(tokenizer, model))
    f.write("\n")
    
f.close()
