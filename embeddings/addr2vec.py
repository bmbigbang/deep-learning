import numpy as np
import time

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
##############################################
data = []
valid = []
test = []
field_names = ['NUMBER', 'STREET', 'UNIT', 'CITY', 'DISTRICT', 'REGION', 'POSTCODE']
for state in os.listdir('./openaddr/us'):

    for filename in glob.glob('./openaddr/us/{}/*.csv'.format(state)):
        csv = pd.read_csv(filename, dtype={
            'LON': np.float64, 'LAT': np.float64, 'NUMBER': np.str, 'STREET': np.str,
            'UNIT': np.str, 'CITY': np.str, 'DISTRICT': np.str, 'REGION': np.str,
            'POSTCODE': np.str, 'ID': np.object, 'HASH': np.object})
        csv.fillna('', inplace=True)
        # print("Example data:\n {}".format(csv.iloc[2]))

        stack = np.stack((csv[x] for x in field_names), axis=-1)

        for i in stack:
            for field_name, k in zip(field_names, i):
                roll = np.random.random()
                if roll > 0.9:
                    for l in k.split():
                        valid.append('{}\t{}\n'.format(l, field_name))
                elif roll > 0.8:
                    for l in k.split():
                        test.append('{}\t{}\n'.format(l, field_name))
                else:
                    for l in k.split():
                        data.append('{}\t{}\n'.format(l, field_name))

with open('valid.txt', 'w') as f:
    f.writelines(valid)

with open('test.txt', 'w') as f:
    f.writelines(test)

with open('train.txt', 'w') as f:
    f.writelines(data)

import sys
sys.path.append('/home/ubuntu/anago')

import anago
from anago.reader import load_data_and_labels

x_train, y_train = load_data_and_labels('train.txt')
x_valid, y_valid = load_data_and_labels('valid.txt')
x_test, y_test = load_data_and_labels('test.txt')

model = anago.Sequence()
model.train(x_train, y_train, x_valid, y_valid)

##################################################

words = []
vocab_to_id = {}
id_to_vocab = {}
field_names = ['STREET', 'CITY', 'DISTRICT', 'REGION']
idx = 0
for state in os.listdir('./openaddr/us'):

    for filename in glob.glob('./openaddr/us/{}/*.csv'.format(state)):
        csv = pd.read_csv(filename, dtype={
            'LON': np.float64, 'LAT': np.float64, 'NUMBER': np.str, 'STREET': np.str,
            'UNIT': np.str, 'CITY': np.str, 'DISTRICT': np.str, 'REGION': np.str,
            'POSTCODE': np.str, 'ID': np.object, 'HASH': np.object})
        csv.fillna('', inplace=True)
        # print("Example data:\n {}".format(csv.iloc[2]))

        stack = np.stack((csv[x] for x in field_names), axis=-1)

        for i in stack:
            for k in i:
                for l in k.split():
                    if l not in vocab_to_id:
                        vocab_to_id[str(l)] = idx
                        id_to_vocab[idx] = str(l)
                        idx += 1
                    words.append(l)

words = words[:len(words)//2]

int_words = [vocab_to_id[word] for word in words]

from collections import Counter
import random

c = Counter(int_words)
total_count = len(int_words)
train_words = [i for i in int_words if (1 - (np.sqrt(1e-5 / (c[i] / total_count)))) > random.random()]
print(train_words[:10])


def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''

    s = np.random.normal(0, window_size / 1.5, 1)
    R = int(abs(s[0]))
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx + 1:stop + 1])

    return list(target_words)


def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''

    n_batches = len(words) // batch_size

    # only full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y


train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
    labels = tf.placeholder(tf.int32, [None, None], name='labels')

n_vocab = len(id_to_vocab)
n_embedding = 100 # Number of embedding features
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)

# Number of negative labels to sample
n_sampled = 100
with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(n_vocab))

    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                      labels, embed,
                                      n_sampled, n_vocab)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with train_graph.as_default():
    ## From Thushan Ganegedara's implementation
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size // 2))

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

# !mkdir checkpoints

epochs = 10
batch_size = 1000
window_size = 10

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches:

            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            if iteration % 1000 == 0:
                ## From Thushan Ganegedara's implementation
                # note that this is expensive (~20% slowdown if computed every 500 steps)
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = id_to_vocab[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = id_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1
    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    embed_mat = sess.run(embedding)

import pickle
pickle.dump(embed_mat, open('embed_mat.p', 'wb'))
pickle.dump(id_to_vocab, open('id_to_vocab.p', 'wb'))
#
# for address, idx in address_to_id.items():
#     X = np.zeros(vocab_size)
#     Y = idx
#     for j in range(len(address)):
#         X[vocab_to_id[address[j]]] += 1
#
#     # normalize input sizes
#     X = ((X - X.min()) / (X.max() - X.min())) + 0.001
#
#     features.append(X)
#
#     labels.append(Y)
#
# print("Example feature vector and label:")
# print(features[500])
# print(labels[500])
#
#
# # Size of the encoding layer (the hidden layer)
# encoding_dim = 500
# alpha = 0.1
#
# # Input and target placeholders
# inp_shape = vocab_size
#
# inputs_ = tf.placeholder(tf.float32, (None, inp_shape))
# targets_ = tf.placeholder(tf.float32, (None, inp_shape))
#
# # dense layer
# encoded = tf.layers.dense(inputs_, encoding_dim)
# relu4 = tf.maximum(alpha * encoded, encoded)
# print(relu4.shape)
#
# # Output layer logits
# logits = tf.layers.dense(relu4, inp_shape, activation=None)
# print(logits.shape)
# # Sigmoid output from logits
# #decoded = tf.nn.sigmoid(logits, name='outputs')
#
# # loss = tf.log(tf.reduce_sum(tf.abs(targets_ - logits)) + 1)
# loss = tf.losses.mean_squared_error(targets_, logits) + tf.log(1 + tf.reduce_sum(logits))
# # Mean of the loss
# cost = tf.reduce_mean(loss)
#
# # Adam optimizer
# opt = tf.train.AdamOptimizer(0.0002).minimize(cost)
#
# # Create the session
# sess = tf.Session()
#
# epochs = 20
# batch_size = 128
# sess.run(tf.global_variables_initializer())
# for e in range(epochs):
#     for ii in range((len(features) // batch_size) - 1):
#         batch = features[ii * batch_size:(ii + 1) * batch_size]
#         targ = features[ii * batch_size:(ii + 1) * batch_size]
#
#         # add random noise
#         for jj in batch:
#             jj[np.random.randint(vocab_size)] += 0.005
#             jj[np.random.randint(vocab_size)] -= 0.005
#
#         # targ = np.array(labels[ii * batch_size:(ii + 1) * batch_size])
#
#         feed = {inputs_: batch, targets_: targ}
#         batch_cost, _ = sess.run([cost, opt], feed_dict=feed)
#         if ii % (len(features) / 10) == 0:
#             print("Epoch: {}/{}...".format(e + 1, epochs),
#                   "Training loss: {:.4f}".format(batch_cost))
#
# originals = [
#     'milford', 'stonington', 'middletown', 'berlin', 'east hampton', 'watertown',
#     'hartford', 'vernon', 'wethersfield', 'windsor locks', 'west hartford', 'new britain',
#     'east haven', 'new haven', 'woodbridge', 'norwalk', 'newington', 'meriden', 'cromwell',
#     'darien', 'east granby', 'hamden', 'oxford', 'lebanon', 'torrington', 'east hartford',
#     'glastonbury', 'kent', 'madison', 'roxbury'
# ]
#
# test_words = [
#     'milford', 'stonington', 'middletown', 'berlin', 'east hampton', 'watertown',
#     'hartford', 'vernon', 'wethersfield', 'windsor locks', 'west hartford', 'new britain',
#     'east haven', 'new haven', 'woodbridge', 'nortwalk', 'newinghton', 'meriiden', 'cromzwell',
#     'darien', 'easat granby', 'chamden', 'oxfortd', 'lebanont', 'torrington', 'eeast hartford',
#     'glastonbury', 'kient', 'madison', 'roxbury'
# ]
# in_words = []
# for word in test_words:
#     X = np.zeros(vocab_size)
#     for j in range(len(word)):
#         if word[j] in vocab_to_id:
#             X[vocab_to_id[word[j]]] += 1
#     for j in range(len(word) - 2):
#         if word[j:j + 3] in vocab_to_id:
#             X[vocab_to_id[word[j:j + 3]]] += 1
#
#     # normalize input sizes
#     X = ((X - X.min()) / (X.max() - X.min())) + 0.001
#     in_words.append(X)
#
# in_words = np.array(in_words)
#
# reconstructed, compressed = sess.run([logits, encoded], feed_dict={inputs_: in_words})
#
# out_words = []
# best_match = (0, 0)
# for out, inp, org in zip(reconstructed, test_words, originals):
#     for X, idx in zip(features, labels):
#         similarity = np.dot(X, out)
#         if similarity > best_match[0]:
#             best_match = (similarity, idx)
#
#     out_words.append((id_to_address[idx], inp, org))
#
# for i in out_words:
#     print(i)