
import numpy as np

import glob
import os
import os.path as path
import pandas as pd
import tensorflow as tf

from urllib.request import urlretrieve
from tqdm import tqdm
import zipfile

dataset_folder_path = 'openaddr'
dataset_filename = 'openaddr-collected-us_northeast.zip'
dataset_name = 'Openaddress Dataset'


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


if not path.isfile(dataset_filename):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset_name) as pbar:
        urlretrieve(
            'https://s3.amazonaws.com/data.openaddresses.io/openaddr-collected-us_northeast.zip',
            dataset_filename,
            pbar.hook)

if not path.isdir(dataset_folder_path):
    with zipfile.ZipFile(dataset_filename) as zip_ref:
        zip_ref.extractall(dataset_folder_path)

id_to_address = {}
address_to_id = {}
i = 0
for state in os.listdir('./openaddr/us'):

    for filename in glob.glob('./openaddr/us/{}/*.csv'.format(state)):
        csv = pd.read_csv(filename)
        if i == 0:
            print("Example data:\n {}".format(csv.iloc[2]))

        stack = np.stack((csv['CITY'],), axis=-1)
        for j in stack:
            addr = " ".join([str(k).lower()
                             for k in j if not isinstance(k, type(np.nan))])
            if addr not in address_to_id and addr != '' and addr != ' ':
                id_to_address[i] = addr
                address_to_id[addr] = i
                i += 1

del csv

print([i for i in address_to_id.keys()][:30])

print("Total addresses: {}".format(len(address_to_id)))

vocab_to_id = {}
int_to_vocab = {}
idx = 0
for address in address_to_id.keys():
    for j in range(len(address)):
        if address[j] not in vocab_to_id:
            vocab_to_id[address[j]] = idx
            int_to_vocab[idx] = address[j]
            idx += 1
    for j in range(len(address) - 2):
        if address[j:j + 3] not in vocab_to_id:
            vocab_to_id[address[j:j + 3]] = idx
            int_to_vocab[idx] = address[j:j + 3]
            idx += 1

vocab_size = len(vocab_to_id)
print("Total unique letters: {}".format(vocab_size))

features = []
labels = []
count = 0

for address, idx in address_to_id.items():
    X = np.zeros(vocab_size)
    Y = idx
    for j in range(len(address)):
        X[vocab_to_id[address[j]]] += 1

    # normalize input sizes
    X = ((X - X.min()) / (X.max() - X.min())) + 0.001

    features.append(X)

    labels.append(Y)

print("Example feature vector and label:")
print(features[500])
print(labels[500])


# Size of the encoding layer (the hidden layer)
encoding_dim = 500
alpha = 0.1

# Input and target placeholders
inp_shape = vocab_size

inputs_ = tf.placeholder(tf.float32, (None, inp_shape))
targets_ = tf.placeholder(tf.float32, (None, inp_shape))

# dense layer
encoded = tf.layers.dense(inputs_, encoding_dim)
relu4 = tf.maximum(alpha * encoded, encoded)
print(relu4.shape)

# Output layer logits
logits = tf.layers.dense(relu4, inp_shape, activation=None)
print(logits.shape)
# Sigmoid output from logits
#decoded = tf.nn.sigmoid(logits, name='outputs')

# loss = tf.log(tf.reduce_sum(tf.abs(targets_ - logits)) + 1)
loss = tf.losses.mean_squared_error(targets_, logits) + tf.log(1 + tf.reduce_sum(logits))
# Mean of the loss
cost = tf.reduce_mean(loss)

# Adam optimizer
opt = tf.train.AdamOptimizer(0.0002).minimize(cost)

# Create the session
sess = tf.Session()

epochs = 20
batch_size = 128
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range((len(features) // batch_size) - 1):
        batch = features[ii * batch_size:(ii + 1) * batch_size]
        targ = features[ii * batch_size:(ii + 1) * batch_size]

        # add random noise
        for jj in batch:
            jj[np.random.randint(vocab_size)] += 0.005
            jj[np.random.randint(vocab_size)] -= 0.005

        # targ = np.array(labels[ii * batch_size:(ii + 1) * batch_size])

        feed = {inputs_: batch, targets_: targ}
        batch_cost, _ = sess.run([cost, opt], feed_dict=feed)
        if ii % (len(features) / 10) == 0:
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Training loss: {:.4f}".format(batch_cost))

originals = [
    'milford', 'stonington', 'middletown', 'berlin', 'east hampton', 'watertown',
    'hartford', 'vernon', 'wethersfield', 'windsor locks', 'west hartford', 'new britain',
    'east haven', 'new haven', 'woodbridge', 'norwalk', 'newington', 'meriden', 'cromwell',
    'darien', 'east granby', 'hamden', 'oxford', 'lebanon', 'torrington', 'east hartford',
    'glastonbury', 'kent', 'madison', 'roxbury'
]

test_words = [
    'milford', 'stonington', 'middletown', 'berlin', 'east hampton', 'watertown',
    'hartford', 'vernon', 'wethersfield', 'windsor locks', 'west hartford', 'new britain',
    'east haven', 'new haven', 'woodbridge', 'nortwalk', 'newinghton', 'meriiden', 'cromzwell',
    'darien', 'easat granby', 'chamden', 'oxfortd', 'lebanont', 'torrington', 'eeast hartford',
    'glastonbury', 'kient', 'madison', 'roxbury'
]
in_words = []
for word in test_words:
    X = np.zeros(vocab_size)
    for j in range(len(word)):
        if word[j] in vocab_to_id:
            X[vocab_to_id[word[j]]] += 1
    for j in range(len(word) - 2):
        if word[j:j + 3] in vocab_to_id:
            X[vocab_to_id[word[j:j + 3]]] += 1

    # normalize input sizes
    X = ((X - X.min()) / (X.max() - X.min())) + 0.001
    in_words.append(X)

in_words = np.array(in_words)

reconstructed, compressed = sess.run([logits, encoded], feed_dict={inputs_: in_words})

out_words = []
best_match = (0, 0)
for out, inp, org in zip(reconstructed, test_words, originals):
    for X, idx in zip(features, labels):
        similarity = np.dot(X, out)
        if similarity > best_match[0]:
            best_match = (similarity, idx)

    out_words.append((id_to_address[idx], inp, org))

for i in out_words:
    print(i)