import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from termcolor import cprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.combine import SMOTEENN

from colnames import *
from load_dataset import load_and_preprocess, extract_features, split_categorical_and_interval

def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

create_batches = lambda arr, batch_size: np.split(arr, np.arange(0, arr.shape[0], step=batch_size))[1:]
np_to_onehot = lambda y: np.eye(np.max(y) + 1)[y]

class Model:

    def __init__(self, training_epochs, train_X, train_y, test_X, test_y,
                 columns_categ, columns_numeric, vocab_size, variable_types, use_ft_embedding, use_onehot, n_classes=4):

        # Input Data
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        # Input Data Details
        # self.vocab = vocab
        # self.vocab_size = len(vocab)
        self.vocab_size = vocab_size
        self.variable_types = variable_types
        self.columns_categ = columns_categ
        self.columns_numeric = columns_numeric
        self.num_columns_categ = len(columns_categ)
        self.num_columns_numeric = len(columns_numeric)

        self.use_onehot = use_onehot
        
        if self.use_onehot:
            self.num_columns_categ = 0
            self.num_columns_numeric = train_X.shape[1]

        self.use_ft_embedding = use_ft_embedding

        if self.use_ft_embedding:
            self.num_columns_numeric += 300

        print('vocab size:', self.vocab_size)

        # Training Parameters
        self.learning_rate = 0.0001 #0.0001
        self.training_epochs = training_epochs
        self.batch_size = 75
        self.display_step = 1

        # Model Parameters
        self.embedding_size = 30  # 40, 20
        self.n_hidden_1 = 3500  # 1st layer number of neurons, 300, try 2048, #2500
        self.n_hidden_2 = 3500  # 2nd layer number of neurons, #1500
        self.n_classes = n_classes

        cprint('{} categorical columns (embedding size {}), {} numerical columns'.format(self.num_columns_categ, self.num_columns_categ * self.embedding_size, self.num_columns_numeric), 'white', 'on_grey')
        pprint(zip(features_to_use, variable_types))

        self.X0 = tf.placeholder(tf.int32, [None, self.num_columns_categ])
        self.X1 = tf.placeholder(tf.float32, [None, self.num_columns_numeric])
        self.Y = tf.placeholder(tf.int32, [None, self.n_classes])
        
        [tf.add_to_collection('vars', var) for var in [self.X0, self.X1, self.Y]]

        self.logits = self.multilayer_perceptron(self.X0, self.X1, prelu)

        # Initializing the variables
        self.saver = tf.train.Saver()  # defaults to saving all variables
        # self.writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
        print('Initialised...')

    def multilayer_perceptron(self, x0, x1, activation):
        if self.use_onehot:
            input_layer = tf.cast(x0, tf.float32)
            # print(input_layer.eval())
        
        else:
            # Otherwise, Use Random Uniform Embedding LUT
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
                embedding_matrix = tf.nn.embedding_lookup(W, x0)

            input_layer = tf.reshape(embedding_matrix, [-1, self.embedding_size * self.num_columns_categ])

        # Concatenate embeddings of categorical variables with numeric variables
        concat_1 = tf.concat([input_layer, x1], 1)
        # concat_1 = tf.Print(concat_1, [concat_1], summarize=1000)

        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        # Hidden fully connected layer with n_hidden_1 neurons
        dense_1 = tf.layers.dense(concat_1, self.n_hidden_1, activation)
        
        # if training:, could be placeholder
        # dropout = tf.layers.dropout(inputs=dense_1, rate=keep_prob, training=(mode == tf.estimator.ModeKeys.TRAIN))

        if self.n_hidden_2 is not None:
            # Hidden fully connected layer with n_hidden_2 neurons
            dense_2 = tf.layers.dense(dense_1, self.n_hidden_2, activation)
            out_layer = tf.layers.dense(dense_2, self.n_classes)
        else:
            out_layer = tf.layers.dense(dense_1, self.n_classes)
        
        return out_layer

    def optimise(self):
        # Define loss and optimizer
        # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=multilayer_perceptron(X, 0.4), labels=Y))
        # Sparse softmax supports index labels ()
        loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        # lossL1 = tf.add_n([tf.nn.l1_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.001
        # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.0005

        loss_op = loss_cross_entropy + lossL2

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        # create a summary for our cost and accuracy
        # tf.summary.scalar("loss_cross_entropy", loss_cross_entropy)
        # tf.summary.scalar("loss_l2", lossL2)
        # tf.summary.scalar("loss_total", loss_op)
        
        return train_op, loss_op

    def train(self, sess, train_op, loss_op):
        # Assuming data has been preprocessed
        # TODO: convert to onehot prior to preprocess
        scores = []
        train_writer = tf.summary.FileWriter('tensorboard_log/test_acc', sess.graph)

        # Training cycle
        X_batches = create_batches(self.train_X, self.batch_size)
        y_batches = create_batches(np_to_onehot(self.train_y), self.batch_size)
        num_batches = int(float(len(train_y)) / self.batch_size)

        for epoch in range(self.training_epochs):
            avg_cost = 0.0
            # Loop over all batches
            for batch_num in range(num_batches):
                batch_x = X_batches[batch_num]
                batch_y = y_batches[batch_num]

                batch_x_categ, batch_x_numeric = split_categorical_and_interval(batch_x, variable_types)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={self.X0: batch_x_categ, self.X1: batch_x_numeric, self.Y: batch_y})
                # Compute average loss
                avg_cost += c / num_batches
            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

            if epoch % (self.display_step * 4) == 0:
                score, summary_op = self.test(sess, report=True)
                scores.append(score)
                train_writer.add_summary(summary_op, epoch)

                # Create a checkpoint in every iteration
                self.saver.save(sess, save_path="checkpoints/ass3", global_step=epoch)

        print("Optimisation Finished!")
        return max(scores)

    def test(self, sess, report=False):

        test_X_categ, test_X_numeric = split_categorical_and_interval(self.test_X, self.variable_types)

        # Assuming data has been preprocessed
        # TODO: convert to onehot prior to preprocess
        pred = tf.nn.softmax(self.logits)  # Apply softmax to logits
        pred_classes = tf.argmax(pred, 1, name='pred_labels')

        # Test model
        correct_prediction = tf.equal(pred_classes, tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        y_p = tf.argmax(pred, 1)
        tf.summary.scalar("test_accuracy", accuracy)
        merge = tf.summary.merge_all()
        # print(self.test_y)
        # np_to_onehot(test_y)
        (accuracy, y_pred, summary) = sess.run([accuracy, y_p, merge], feed_dict={self.X0: test_X_categ, self.X1: test_X_numeric, self.Y: np_to_onehot(self.test_y)})

        print('acc: {}, F1: {}'.format(accuracy, f1_score(self.test_y, y_pred, average='micro')))
        
        if report:
            target_names = ['class 0', 'class 1', 'class 2', 'class 3']
            print(classification_report(self.test_y, y_pred, target_names=target_names))
            # confusion = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=tf.argmax(pred, 1), num_classes=num_classes)

        return f1_score(self.test_y, y_pred, average='micro'), summary

    def inference(self, sess, inference_X):
        cprint('Predicting...', 'green', 'on_yellow')
        inference_X_categ, inference_X_numeric = split_categorical_and_interval(inference_X, self.variable_types)
        pred = tf.nn.softmax(self.logits)  # Apply softmax to logits
        pred_classes = tf.argmax(pred, 1, name='pred_labels')

        y_pred = sess.run([pred_classes], feed_dict={self.X0: inference_X_categ, self.X1: inference_X_numeric})
        return y_pred

if __name__ == '__main__':
    cprint('STARTING', 'white', 'on_green')

    nrows = 500
    # nrows = None

    df_train = load_and_preprocess('TrainingSet(3).csv', nrows=nrows)
    df_inference = load_and_preprocess('TestingSet(2).csv', nrows=nrows)

    # Inference is the final prediction (without ground-truth)
    X, y, inference_row_ids, inference_X, vocab_size, variable_types, features_to_use = extract_features(df_train, df_inference, colnames_categ, colnames_interval, fuzzy_matching=True)

    # FIXME: ugly...
    if variable_types[-1] == 'ft_embedding':
        del variable_types[-1]
        use_ft_embedding = True
    else:
        use_ft_embedding = False

    use_onehot = not('categorical_nominal' in variable_types)

    cprint('use_ft_embedding: {}, use_onehot: {}'.format(use_ft_embedding, use_onehot), 'white', 'on_cyan')

    # Test is ONLY for evaluation metrics (contains ground-truth)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.15)

    # sm = SMOTEENN()
    # train_X, train_y = sm.fit_sample()
    # from imblearn.under_sampling import RandomUnderSampler
    # from imblearn.over_sampling import RandomOverSampler
    # from collections import Counter

    # ros = RandomOverSampler(ratio='minority')
    # train_X, train_y = ros.fit_sample(train_X.copy(), train_y.copy())
    # train_X, train_y = ros.fit_sample(train_X.copy(), train_y.copy())
    # print(Counter(train_y))
    # exit()

    tf.reset_default_graph()
    with tf.Graph().as_default():
        m = Model(350, train_X, train_y, test_X, test_y, colnames_categ, colnames_interval, vocab_size, variable_types, use_ft_embedding, use_onehot)
        train_op, loss_op = m.optimise()
        
        init = tf.global_variables_initializer()
    
        with tf.Session() as sess:

            sess.run(init)

            m.train(sess, train_op, loss_op)
            m.test(sess)

            df = pd.DataFrame()
            y_pred = m.inference(sess, inference_X)

            print(len(y_pred[0]))
            print(len(inference_row_ids))
            df['row ID'] = inference_row_ids
            df['case_status'] = y_pred[0]
            df.to_csv('result_random.csv', index=False)
