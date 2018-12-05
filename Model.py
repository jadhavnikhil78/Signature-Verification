import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def LogisticRegression(Training_X, Training_Y, Testing_X, Testing_Y, display = False):
    # Begin building the model framework
    # Declare the variables that need to be learned and initialization
    # There are 4 features here, A's dimension is (4, 1)
    A = tf.Variable(tf.random_normal(shape=[901, 2]))
    b = tf.Variable(tf.random_normal(shape=[2]))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)



    # Define placeholders
    data = tf.placeholder(dtype=tf.float32, shape=[None, 901])
    target = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    # Declare the model you need to learn
    mod = tf.matmul(data, A) + b

    # Declare loss function
    # Use the sigmoid cross-entropy loss function,
    # first doing a sigmoid on the model result and then using the cross-entropy loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target))


    # Define the learning rate， batch_size etc.
    learning_rate = 0.001
    batch_size = 30
    iter_num = 1500

    # Define the optimizer
    opt = tf.train.GradientDescentOptimizer(learning_rate)

    # Define the goal
    goal = opt.minimize(loss)

    # Define the accuracy
    # The default threshold is 0.5, rounded off directly
    prediction = tf.round(tf.sigmoid(mod))
    # Bool into float32 type
    correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
    # Average
    accuracy = tf.reduce_mean(correct)
    # End of the definition of the model framework

    # Start training model
    # Define the variable that stores the result
    loss_trace = []
    train_acc = []
    test_acc = []

    # training model
    for epoch in range(iter_num):
        # Generate random batch index
        batch_index = np.random.choice(len(Training_X), size=batch_size)
        batch_train_X = Training_X[batch_index]
        batch_train_y = np.matrix(Training_Y[batch_index])
        sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_y})
        temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
        # convert into a matrix, and the shape of the placeholder to correspond
        temp_train_acc = sess.run(accuracy, feed_dict={data: Training_X, target: np.matrix(Training_Y)})
        temp_test_acc = sess.run(accuracy, feed_dict={data: Testing_X, target: np.matrix(Testing_Y)})
        # recode the result
        loss_trace.append(temp_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        # output
        if (epoch + 1) % 300 == 0:
            print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
                                                                          temp_train_acc, temp_test_acc))

    if display:
        # Visualization of the results
        # loss function
        plt.plot(loss_trace)
        plt.title('Cross Entropy Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
