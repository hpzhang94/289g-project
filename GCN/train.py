from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()


from sklearn import metrics
from utils import *
from models import GCN, MLP
import random
import os

dataset = 'R8'

# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    FLAGS.dataset)
features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
features = preprocess_features(features)
support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy,
                        model.pred, model.labels], feed_dict=feed_dict_val)
        
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


# Init variables
cost_val = []
sess.run(tf.global_variables_initializer())

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,
                     model.layers[0].embedding], feed_dict=feed_dict)

    # Validation
    cost, acc, pred, labels, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    print(f'Epoch: {epoch} Train Accuracy: {outs[2]} Val Accuracy: {acc}')

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Trainable Parameters: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
print("Optimization Finished!")

saver = tf.train.Saver()
saver.save(sess, 'models/model')

# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)

print(f'Accuracy: {test_acc}')

test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])
        
test_size = len(test_pred)
doc_len = open('./data/R8.len.txt').readlines()[-test_size:]
doc_len = [int(doc) for doc in doc_len]
splits = lambda x: 0 if x < 30 else 1 if x < 50 else 2 if x < 70 else 3
sums = {0: 0, 1: 0, 2: 0, 3: 0}
accs = {0: 0, 1: 0, 2: 0, 3: 0}
print("sums", sums)
print("accs", accs)
for doc in doc_len:
    sums[splits(doc)] += 1

for i, (a, b) in enumerate(zip(test_pred, test_labels)):
    if a == b:
        accs[splits(doc_len[i])] += 1
for i in range(4):
    print(accs[i] / sums[i])

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(test_labels, test_pred, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(
    test_labels, test_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(
    test_labels, test_pred, average='micro'))

# doc and word embeddings
word_embeddings = outs[3][train_size: adj.shape[0] - test_size]
train_doc_embeddings = outs[3][:train_size]  # include val docs
test_doc_embeddings = outs[3][adj.shape[0] - test_size:]


f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
words = f.readlines()
f.close()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
f = open('data/' + dataset + '_word_vectors.txt', 'w')
f.write(word_embeddings_str)
f.close()

doc_vectors = []
doc_id = 0
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
f = open('data/' + dataset + '_doc_vectors.txt', 'w')
f.write(doc_embeddings_str)
f.close()
