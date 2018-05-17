import numpy as np
import tensorflow as tf
from pprint import pprint
from termcolor import cprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from colnames import *
from load_dataset import load_dataset, split_categorical_and_interval

def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def train_and_test(training_epochs, selected_feature_names_categ, selected_feature_names_interval):
    # Parameters
    learning_rate = 0.0001
    training_epochs = 80
    batch_size = 200
    display_step = 1

    X, y, vocab_size, variable_types, features_to_use = load_dataset('TrainingSet(3).csv', selected_feature_names_categ, selected_feature_names_interval, fuzzy_matching=True)
    # test_X, vocab_size, variable_types, features_to_use = load_dataset('TrainingSet(3).csv')

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.15)
    test_X_categ, test_X_numeric = split_categorical_and_interval(test_X, variable_types)

    np_to_onehot = lambda y: np.eye(np.max(y) + 1)[y]
    create_batches = lambda arr, batch_size: np.split(arr, np.arange(0, arr.shape[0], step=batch_size))[1:]

    # TODO sort categorical variables by vocab contribution, reduce it this way!
    print('vocab size:', vocab_size)

    # Network Parameters
    embedding_size = 50
    n_hidden_1 = 5000  # 1st layer number of neurons, 300, try 2048
    n_hidden_2 = 5000  # 2nd layer number of neurons
    # n_hidden_2 = None  # 2nd layer number of neurons
    n_classes = len(np.unique(train_y))  # number total classes

    num_columns_categ = test_X_categ.shape[1]
    num_columns_numeric = test_X_numeric.shape[1]
    print(test_X_numeric)
    print(test_X_numeric.shape)

    cprint('{} categorical columns (embedding size {}), {} numerical columns'.format(num_columns_categ, num_columns_categ * embedding_size, num_columns_numeric), 'white', 'on_grey')
    pprint(zip(features_to_use, variable_types))

    # tf Graph input
    X0 = tf.placeholder(tf.int32, [None, num_columns_categ])
    X1 = tf.placeholder(tf.float32, [None, num_columns_numeric])
    Y = tf.placeholder(tf.int32, [None, n_classes])

    # Create model
    def multilayer_perceptron(x0,x1,activation):

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            embedding_matrix = tf.nn.embedding_lookup(W, x0)

        embedding_layer = tf.reshape(embedding_matrix, [-1, embedding_size * num_columns_categ])

        # numeric_input_reshaped = tf.reshape(x1, [-1, num_columns_numeric])

        # Concatenate embeddings of categorical variables with numeric variables
        concat_1 = tf.concat([embedding_layer, x1], 1)

        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        # Hidden fully connected layer with n_hidden_1 neurons
        dense_1 = tf.layers.dense(concat_1, n_hidden_1, activation)
        
        # if training:, could be placeholder
        # dropout = tf.layers.dropout(inputs=dense_1, rate=keep_prob, training=(mode == tf.estimator.ModeKeys.TRAIN))

        if n_hidden_2 is not None:
            # Hidden fully connected layer with n_hidden_2 neurons
            dense_2 = tf.layers.dense(dense_1, n_hidden_2, activation)
            out_layer = tf.layers.dense(dense_2, n_classes)
        else:
            out_layer = tf.layers.dense(dense_1, n_classes)
        
        return out_layer

    logits = multilayer_perceptron(X0, X1, prelu)

    def test(sess, report=False):
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        y_p = tf.argmax(pred, 1)
        (accuracy, y_pred) = sess.run([accuracy, y_p], feed_dict={X0: test_X_categ, X1: test_X_numeric, Y: np_to_onehot(test_y)})
        print('acc: {}, F1: {}'.format(accuracy, f1_score(test_y, y_pred, average='micro')))
        
        # auc, update_op = tf.metrics.auc(np_to_onehot(test_y), pred)
        # print("tf auc: {}".format(sess.run([auc, update_op])))

        # print(test_y)
        # print(y_pred)

        if report:
            target_names = ['class 0', 'certified', 'class 2', 'class 3']
            print(classification_report(test_y, y_pred, target_names=target_names))

    # Define loss and optimizer
    # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=multilayer_perceptron(X, 0.4), labels=Y))
    # Sparse softmax supports index labels ()
    loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.001

    loss_op = loss_cross_entropy + lossL2

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        X_batches, y_batches = create_batches(train_X, batch_size), create_batches(np_to_onehot(train_y), batch_size)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            num_batches = int(float(len(train_y)) / batch_size)
            # Loop over all batches
            for batch_num in range(num_batches):
                batch_x = X_batches[batch_num]
                batch_y = y_batches[batch_num]

                batch_x_categ, batch_x_numeric = split_categorical_and_interval(batch_x, variable_types)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X0: batch_x_categ, X1: batch_x_numeric, Y: batch_y})
                # Compute average loss
                avg_cost += c / num_batches
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

            if epoch % (display_step*5) == 0:
                test(sess)

        print("Optimisation Finished!")
        # need to save the model 

        # # confusion = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=tf.argmax(pred, 1), num_classes=num_classes)
        
        # # Calculate accuracy
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        # print("Accuracy:", accuracy.eval({X: test_X, Y: np_to_onehot(test_y)}))
        return test(sess, report=True)

if __name__ == '__main__':
    score = train_and_test(50, selected_feature_names_categ, selected_feature_names_interval)