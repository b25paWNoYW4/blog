---
title: Deep Learning in 7 lines of code 个人翻译
categories: DeepLearning
date: 2017-10-16 15:15:19
---

[原文链接：https://chatbotslife.com/deep-learning-in-7-lines-of-code-7879a8ef8cfb](https://chatbotslife.com/deep-learning-in-7-lines-of-code-7879a8ef8cfb)

#   7行代码入门深度学习

 机器学习的本质是识别数据中的模式。这可以归结为三个要素：数据，软件和数学。7行代码可以做很多的事。  

 ![Steve McQueen and Yul Brynner in “The Magnificent Seven” (1960)](https://cdn-images-1.medium.com/max/1600/1*ICbj3kCYcukiYM6dC-wT5w.jpeg)

 将一个深度学习问题转化为数行代码的方法是使用抽象层，也就是“框架”。今天我们将使用 `tensorflow`和 `tflearn`。  

 抽象化是软件的本质：此时此刻你用来浏览这篇文章的应用程序正是某种操作系统上方的一个抽象应用，它知道如何读取文件、显示图像等。这是一个在较低级别功能上方的抽象层。最终，是操作比特的CPU级别——“裸机”。

<!-- more -->

##   程序框架是抽象层

 我们通过使用 tflearn实现。tflearn是一个基于 tensorflow之上的高层API，tensorflow是基于python的框架。和往常一样，我们将使用iPython notebook完成我们的工作。  

##   让我们开始吧

 在“[神经网络是如何工作的](https://chatbotslife.com/how-neural-networks-work-ff4c7ad371f7)”一文中我们已经在Python中（不使用框架）构建过了一个神经网络，用“toy data”例子，展示了机器学习是如何从数据中“学习”得到模式的。“toy data”很简单，我们就用它来直观的了解其中的模式。  

 本例中的“零抽象”代码在[这里](https://github.com/ugik/notebooks/blob/master/Simple_Neural_Network.ipynb).模型中的每一步数学运算都有详细的注释。  

 ![2-hidden layer ANN](https://cdn-images-1.medium.com/max/1600/1*CcQPggEbLgej32mVF2lalg.png)

 用两个隐藏层和梯度下降算法来扩展我们之前构建的文本分析网络模型，模型只有80行代码，并且没有使用框架。这便是一个“深度学习”的例子，“深度”得名于网络中使用的隐藏层。  

 将模型定义得直观易明会很吃力，大部分的代码都是用于模型训练：

        # convert output of sigmoid function to its derivative
        def sigmoid_output_to_derivative(output):
            return output*(1-output) 

        def think(sentence, show_details=False):
            x = bow(sentence.lower(), words, show_details)
            if show_details:
                print ("sentence:", sentence, "\n bow:", x)
            # input layer is our bag of words
            l0 = x
            # matrix multiplication of input and hidden layer
            l1 = sigmoid(np.dot(l0, synapse_0))
            # output layer
            l2 = sigmoid(np.dot(l1, synapse_1))
            return l2

        def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

            print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
            print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
            np.random.seed(1)

            last_mean_error = 1
            # randomly initialize our weights with mean 0
            synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
            synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

            prev_synapse_0_weight_update = np.zeros_like(synapse_0)
            prev_synapse_1_weight_update = np.zeros_like(synapse_1)

            synapse_0_direction_count = np.zeros_like(synapse_0)
            synapse_1_direction_count = np.zeros_like(synapse_1)
                
            for j in iter(range(epochs+1)):

                # Feed forward through layers 0, 1, and 2
                layer_0 = X
                layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                        
                if(dropout):
                    layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

                layer_2 = sigmoid(np.dot(layer_1, synapse_1))

                # how much did we miss the target value?
                layer_2_error = y - layer_2

                if (j% 10000) == 0 and j > 5000:
                    # if this 10k iteration's error is greater than the last iteration, break out
                    if np.mean(np.abs(layer_2_error)) < last_mean_error:
                        print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                        last_mean_error = np.mean(np.abs(layer_2_error))
                    else:
                        print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                        break
                        
                # in what direction is the target value?
                # were we really sure? if so, don't change too much.
                layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

                # how much did each l1 value contribute to the l2 error (according to the weights)?
                layer_1_error = layer_2_delta.dot(synapse_1.T)

                # in what direction is the target l1?
                # were we really sure? if so, don't change too much.
                layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
                
                synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
                synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
                
                if(j > 0):
                    synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
                    synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
                
                synapse_1 += alpha * synapse_1_weight_update
                synapse_0 += alpha * synapse_0_weight_update
                
                prev_synapse_0_weight_update = synapse_0_weight_update
                prev_synapse_1_weight_update = synapse_1_weight_update

            now = datetime.datetime.now()

            # persist synapses
            synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
                       'datetime': now.strftime("%Y-%m-%d %H:%M"),
                       'words': words,
                       'classes': classes
                      }
            synapse_file = "synapses.json"

            with open(synapse_file, 'w') as outfile:
                json.dump(synapse, outfile, indent=4, sort_keys=True)
        print ("saved synapses to:", synapse_file)

 这样就完成了，接着使用框架将其抽象化。

##   使用Tensorflow抽象化

 在“揭秘Tensorflow”一文中，我们构建了和本例相同的神经网络；现在，让我们再次展示机器学习是如何从数据中“学”得模式的。

 使用tensorflow框架将简化我们的代码（底层的数学结果仍然相同），例如，计算处理梯度下降和损失函数的代码被减少到了2行。

            # formula for cost (error)
            cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
            # optimize for cost using GradientDescent
            optimizer = tf.train.GradientDescentOptimizer(1).minimize(cost)

 模型的定义也会随之简化，常用的数学函数（如sigmoid函数）封装被在了框架中。

        # our predictive model's definition
        def neural_network_model(data):

            # hidden layer 1: (data * W) + b
            l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
            l1 = tf.sigmoid(l1)

            # hidden layer 2: (hidden_layer_1 * W) + b
            l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
            l2 = tf.sigmoid(l2)

            # output: (hidden_layer_2 * W) + b
            output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']

            return output
 
 你可以想象一个复杂的神经网络“流动”，就像AlexNet，使用tensorflow来简化模型定义、“流”特性进行工作。

 ![https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](https://cdn-images-1.medium.com/max/1600/1*WXMS_K-zeVgqIttDjKaTlQ.png)

##   再次抽象

 仍旧觉得代码长、复杂？使用tflearn再次对代码进行抽象：

        # Build neural network
        net = tflearn.input_data(shape=[None, 5])
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        # Start training (apply gradient descent algorithm)
        model.fit(train_x, train_y, n_epoch=500, batch_size=16, show_metric=True)

 **牛B了**——5行代码用来定义神经网络的结构（输出层+2层隐藏层+输出层+回归），2行代码用于训练。

 notebook源码在[这里](https://github.com/ugik/notebooks/blob/master/tflearn%20toy%20ANN.ipynb)

 仔细看下代码，你会发现使用的数据和学习目标和[之前的例子](https://chatbotslife.com/tensorflow-demystified-80987184faf7)是一样的。

##   框架安装

 确保你已经安装了tensorflow 1.0.x ，tflearn框架不支持版本1.0之前的tensorflow。

        import tensorflow as tf
        tf.__version__

        '1.0.1'

 你可以使用pip（在linux上）：

        python -m pip install — upgrade tensorflow tflearn

##   数据

 下一步，准备好我们的数据，数据和之前的[tensorflow例子](https://chatbotslife.com/tensorflow-demystified-80987184faf7)里的"toy data"一样。训练数据在那个例子中已经详细解释过了——应该是很容易明白的。请注意，我们不需要准备测试数据了，tflearn 框架能帮我们做这份工作。

        import numpy as np
        import tflearn
        import random

        def create_feature_sets_and_labels():

            # known patterns (5 features) output of [1] of positions [0,4]==1
            features = []
            features.append([[0, 0, 0, 0, 0], [0,1]])
            features.append([[0, 0, 0, 0, 1], [0,1]])
            features.append([[0, 0, 0, 1, 1], [0,1]])
            features.append([[0, 0, 1, 1, 1], [0,1]])
            features.append([[0, 1, 1, 1, 1], [0,1]])
            features.append([[1, 1, 1, 1, 0], [0,1]])
            features.append([[1, 1, 1, 0, 0], [0,1]])
            features.append([[1, 1, 0, 0, 0], [0,1]])
            features.append([[1, 0, 0, 0, 0], [0,1]])
            features.append([[1, 0, 0, 1, 0], [0,1]])
            features.append([[1, 0, 1, 1, 0], [0,1]])
            features.append([[1, 1, 0, 1, 0], [0,1]])
            features.append([[0, 1, 0, 1, 1], [0,1]])
            features.append([[0, 0, 1, 0, 1], [0,1]])
            features.append([[1, 0, 1, 1, 1], [1,0]])
            features.append([[1, 1, 0, 1, 1], [1,0]])
            features.append([[1, 0, 1, 0, 1], [1,0]])
            features.append([[1, 0, 0, 0, 1], [1,0]])
            features.append([[1, 1, 0, 0, 1], [1,0]])
            features.append([[1, 1, 1, 0, 1], [1,0]])
            features.append([[1, 1, 1, 1, 1], [1,0]])
            features.append([[1, 0, 0, 1, 1], [1,0]])

            # shuffle our features and turn into np.array
            random.shuffle(features)
            features = np.array(features)

            # create train and test lists
            train_x = list(features[:,0])
            train_y = list(features[:,1])

            return train_x, train_y

##   不可思议的7行代码

 深度学习部分的代码：

        # Build neural network
        net = tflearn.input_data(shape=[None, 5])
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        # Start training (apply gradient descent algorithm)
        model.fit(train_x, train_y, n_epoch=500, batch_size=16, show_metric=True)

 前5行代码使用了tflearn的函数定义了神经网络中的“网络”：从tflearn.input_data到tflearn.fully_connected到tflearn.regression。这里“流”的过程和前面的tensorflow例子中的一样：输入数据有5个特征，在每个隐藏层中有32个节点，网络输出有2个类。

 接下来，实例化这个深度神经网络：将"net"传入[tflearn.DNN](http://tflearn.org/models/dnn/)函数，附带上tensorboard参数以启用日志。

 最后，将模型与训练数据连接。注意训练指标中的"sweet interface"，设置"n_epochs"以观察准确率的变化情况。

 ![模型训练时的实时数据](https://cdn-images-1.medium.com/max/1600/1*5UIqnedBzsYTXJ81wEU-vg.gif)

        Training Step: 1999  | total loss: 0.01591 | time: 0.003s
        | Adam | epoch: 1000 | loss: 0.01591 - acc: 0.9997 -- iter: 16/22
        Training Step: 2000  | total loss: 0.01561 | time: 0.006s
        | Adam | epoch: 1000 | loss: 0.01561 - acc: 0.9997 -- iter: 22/22
        --

##   预测

 现在可以使用模型来预测输出了。请确保将训练数据中任何与用来测试的"patterns"（注释掉测试用的模式），否则模型将作弊。

        print(model.predict([[0, 0, 0, 1, 1]]))
        print(model.predict([[1, 0, 1, 0, 1]]))

        [[0.004509848542511463, 0.9954901337623596]]
        [[0.9810173511505127, 0.018982617184519768]]

 模型正确的识别出了 [1, _, _, _, 1] 的模式，并输出了 [1, 0]。

 为了方便在notebook中进行迭代操作，可以通过在我们的代码前加上两行代码以重置模型的"graph"：

        # reset underlying graph data
        import tensorflow as tf
        tf.reset_default_graph()
        # Build neural network
        ...

 通过抽象化，我们能专注于准备数据并使用模型进行预测。

##   Tensorboard

 tflearn框架自动将数据传至tensorboard：tensorflow自带的一个可视化工具。因为我们在**tflearn.DNN**配置了日志文件，所以可以通过下发代码快速查看日志。

        $ tensorboard — logdir=tflearn_logs

        Starting TensorBoard b’41' on port 6006
        (You can navigate to http://127.0.1.1:6006)

 我们能看到“流”的图形化：

 ![tensorboard Graphs view](https://cdn-images-1.medium.com/max/1600/1*Hxyae_jiPNunhyGAIPWDlw.png)

 模型准确率和损失函数：

 ![tensorboard Scalars view](https://cdn-images-1.medium.com/max/1600/1*uprKwyFqZJZFez6zFEy0iQ.png)

 可以很明显的看到，训练时不需要太多的epochs就能达到一个稳定的准确度。

##   其它例子

 这是一个LSTM RNN（Long-Short-Term-Memory Recurrent Neural-Net）的tflearn设置， often used to learn sequences of data with memory.注意网络和tflearn.lstm的设置，不过大体上基本概念是一样的。

        # Network building
        net = tflearn.input_data([None, 100])
        net = tflearn.embedding(net, input_dim=10000, output_dim=128)
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='categorical_crossentropy')

        # Training
        model = tflearn.DNN(net)
        model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
        batch_size=32)

 这里有一个常用于图像识别的卷积神经网络的tflearn设置。Notice again all we’re doing is providing the mathematical sequence for the network’s mathematical equations, then feeding it data.

        # Building convolutional network
        network = input_data(shape=[None, 28, 28, 1], name='input')
        network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        network = fully_connected(network, 128, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, 256, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.01,
                             loss='categorical_crossentropy', name='target')

        # Training
        model = tflearn.DNN(network)
        model.fit({'input': X}, {'target': Y}, n_epoch=20,
                   validation_set=({'input': testX}, {'target': testY}),
        snapshot_step=100, show_metric=True, run_id='convnet_mnist')

 我们从不使用框架的代码开始，了解了深度学习的具体过程。没有“黑盒子”。当对底层代码有深入了解后，使用框架来简化我们的工作。

 深度学习框架通过封装必要的基础功能来简化您的工作。As the frameworks evolve and improve, we inherit those improvements automatically, consequentially we go from ‘black box’ to ‘black boxes within a black box’.

 >“So let it be written, So let it be done”

 ![Yul Brenner “The King and I” (1954)](https://cdn-images-1.medium.com/max/1600/1*R37Lnu2faOc9XE83h41gww.jpeg)
 
