# LSTM for international airline passengers problem with regression framing
from cv2 import data
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, LSTMCell
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix


def read_datasetX(npy_path, label_path, file_list):

    dataX = []
    dataY = []
    with open(file_list, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):

            file_name = lines[i].split('/')[1].split('.')[0]
            # print(file_name)
            npy_file_name = '{}/{}.npy'.format(npy_path, file_name)
            labe_file_name = '{}/{}.txt'.format(label_path, file_name)

            with open(labe_file_name, 'r') as lf:
                line = lf.readline()
                label_data = [float(x) for x in line.split(' ')[1:]]
                if '15' in line.split(' '):
                    label_data.append(1)
                else:
                    print(file_name)
                    print('bottle not detected')
            img_data = numpy.load(npy_file_name, allow_pickle=True)
            # print(img_data.shape)
            # img_data = numpy.reshape(img_data, (1797,))

            dataX.append(img_data)
            dataY.append(label_data)
        dataX = numpy.array(dataX)
        print(dataX.shape)

    return dataX, numpy.array(dataY)


def LSTM_single(name,  _X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
    # Reshape to prepare input to hidden activation
    # (num_steps*batch_size, num_input)
    _X = tf.reshape(_X, [num_steps * batch_size, num_input])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    # n_steps * (batch_size, num_input)
    _X = tf.split(0, num_steps, _X)
    # print("_X: ", _X)

    cell = LSTMCell(num_input, num_input)
    state = _istate
    for step in range(num_steps):
        outputs, state = tf.keras.layers.RNN(cell, [_X[step]], state)
        tf.compat.v1.get_variable_scope().reuse_variables()
    return outputs


def build_networks():
    if disp_console:
        print("Building ROLO graph...")

    # Build rolo layers
    lstm_module = LSTM_single(
        'lstm_test', x, istate, weights, biases)
    ious = tf.Variable(tf.zeros([batch_size]), name="ious")
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    #saver.restore(sess, rolo_weights_file)
    if disp_console:
        print("Loading complete!" + '\n')


def training(disp_console, x_path, y_path):
    total_loss = 0
    if disp_console:
        print("TRAINING ROLO...")
    # Use rolo_input for LSTM training
    pred = LSTM_single('lstm_train', x,
                       istate, weights, biases)
    if disp_console:
        print("pred: ", pred)
    pred_location = pred[0][:, 4097:4101]
    if disp_console:
        print("pred_location: ", pred_location)
    if disp_console:
        print("y: ", y)

    correct_prediction = tf.square(pred_location - y)
    if disp_console:
        print("correct_prediction: ", correct_prediction)
    accuracy = tf.reduce_mean(correct_prediction) * 100
    if disp_console:
        print("accuracy: ", accuracy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        accuracy)  # Adam Optimizer

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:

        if (restore_weights == True):
            sess.run(init)
            saver.restore(sess, rolo_weights_file)
            print "Loading complete!" + '\n'
        else:
            sess.run(init)

        step = 0

        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            # Load training data & ground truth
            # [num_of_examples, num_input] (depth == 1)
            batch_xs = rolo_utils.load_yolo_output(
                x_path, batch_size, num_steps, step)
            print('len(batch_xs)= ', len(batch_xs))
            # for item in range(len(batch_xs)):

            batch_ys = rolo_utils.load_rolo_gt(
                y_path, batch_size, num_steps, step)
            batch_ys = utils.locations_from_0_to_1(
                w_img, h_img, batch_ys)

            # Reshape data to get 3 seq of 5002 elements
            batch_xs = np.reshape(
                batch_xs, [batch_size, num_steps, num_input])
            batch_ys = np.reshape(batch_ys, [batch_size, 4])
            if disp_console:
                print("Batch_ys: ", batch_ys)

            pred_location = sess.run(pred_location, feed_dict={
                                     x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*num_input))})
            if disp_console:
                print("ROLO Pred: ", pred_location)
            if disp_console:
                print("ROLO Pred in pixel: ", pred_location[0][0]*w_img, pred_location[0]
                      [1]*h_img, pred_location[0][2]*w_img, pred_location[0][3]*h_img)

            # Save pred_location to file
            utils.save_rolo_output(
                output_path, pred_location, step, num_steps, batch_size)

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros(
                (batch_size, 2*num_input))})
            if step % display_step == 0:
                # Calculate batch loss
                loss = sess.run(accuracy, feed_dict={
                                x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*num_input))})
                if disp_console:
                    # + "{:.5f}".format(accuracy)
                    print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss)
                total_loss += loss
            step += 1
            if disp_console:
                print(step)
            # show 3 kinds of locations, compare!
        print "Optimization Finished!"
        avg_loss = total_loss/step
        print "Avg loss: " + str(avg_loss)
        save_path = saver.save(sess, rolo_weights_file)
        print("Model saved in file: %s" % save_path)
    return avg_loss


lstm_depth = 3
num_steps = 6  # number of frames as an input sequence
num_feat = 4096
num_predict = 6  # final output of LSTM 6 loc parameters
num_gt = 4
num_input = num_feat + num_predict  # data input: 4096+6= 5002

learning_rate = 0.00001
training_iters = 210
batch_size = 1  # 128
display_step = 1

weights = {
    'out': tf.Variable(tf.random_normal([num_input, num_predict]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_predict]))
}

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset

npy_path = 'LSTM_npy'
label_path = 'lstm_train_label'
datasetX, datasetY = read_datasetX(
    npy_path, label_path, file_list='lstm_bottle_cut.txt')

datasetX = datasetX.astype('float32')
datasetY = datasetY.astype('float32')

print(datasetX.shape)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
datasetX = scaler.fit_transform(datasetX)

# split into train and test sets
train_size = int(len(datasetX) * 0.9)
test_size = len(datasetX) - train_size
# train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+1
trainX, trainY = datasetX[0:train_size, :], datasetY[0:train_size, :]
testX, testY = datasetX[train_size:len(
    datasetX), :], datasetY[train_size:len(datasetX), :]
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
