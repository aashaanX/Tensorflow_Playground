import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def read_dataset():
    df = pd.read_csv("C:\\python_playground\\tensorflow\\tensorflow_playground\\Tensorflow_Playground\\sonar.all-data.csv")
    #print(len(df.columns))
    X = df[df.columns[0:60]].values
    y = df[df.columns[60]]

    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    print(X.shape)
    return (X, Y)

#define encoder function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

#Read the data set
X,Y = read_dataset()

#Shuffle the data set
X,Y = shuffle(X, Y, random_state=1)

# Convert the data set in to train and test
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

# Inspect the shpae of training and test
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

#Define the important parameters and varialble to work with tensors
learning_rates = 0.3
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = 2
model_path = "C:\\python_playground\\tensorflow\\tensorflow_playground\\Tensorflow_Playground"

# define the number of hidden layers nad number of neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


#Define the model
def multi_layer_perceptron(x, weights, biases):

    #Hidden layer with RELU activationsd
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    #Hidden layer with sigmoid activation
    layer_2 =  tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Hidden layer with sigmoid activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)

    # output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out'] + biases['out'])
    return out_layer

# define the weight and biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4,n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# Initialize all the Variables

init = tf.global_variables_initializer()

saver = tf.train.Saver()

#Call your model defined

y = multi_layer_perceptron(x, weights, biases)

# Define the cost function and optimizer
print(y_,y)
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_Step = tf.train.GradientDescentOptimizer(learning_rates).minimize(cost_function)

sess = tf.Session()

sess.run(init)

# Calculate the cost and accuaracy for each epoch

mse_history = []
accuaracy_history = []

for epoch in range(training_epochs):
    sess.run(train_Step, feed_dict={x: train_x, y_:train_y})
    cost = sess.run(cost_function,feed_dict={x: train_x, y_:train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuaracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuaracy = (sess.run(accuaracy, feed_dict={x: train_x, y_:train_y}))
    accuaracy_history.append(accuaracy)

    print('epoch : ', epoch, ' - ', 'cost: ', cost, "-MSE : ", mse_, "TrainingAccuaracy : ", accuaracy)

save_path = saver.save(sess, model_path)
print("Model saved in the file {}".format(save_path))

#Plot mse and accuraacy graph

plt.plot(mse_history, 'r')
plt.show()
plt.plot(accuaracy_history)
plt.show()


# print final accuaracy

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuaracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuaracy: ", (sess.run(accuaracy, feed_dict={x: test_x, y_:test_y})))

#print final mean square error

pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" %  sess.run(mse))
