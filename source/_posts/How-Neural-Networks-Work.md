---
title: How Neural Networks Work
categories: DeepLearning
date: 2017-10-13 17:15:06
---

 原文链接：[https://chatbotslife.com/how-neural-networks-work-ff4c7ad371f7](https://chatbotslife.com/how-neural-networks-work-ff4c7ad371f7)  
 
 作者：[gk_](https://chatbotslife.com/@gk_)

#   神经网络是如何工作的

 人工神经网络 Artificial neural networks (ANN)早已成为一个热门话题，聊天机器人也经常使用人工网络来进行文本识别。但老实说，除非你是一个神经科学家，使用大脑来类比神经网络不会让你懂得更多。软件类似于动物大脑中的突触和神经元概念最近火了起来，但软件中的神经网络已经存在了数十年。

 让我们用一个简单的比喻： 一个弦乐器。

 ![](https://cdn-images-1.medium.com/max/800/1*jdBlZ2jfNuyA-Wf65p5AFw.jpeg)

<!-- more -->

##   就像吉他调和弦

 人工神经网络就像一个数据集经过“调整”（tuned）过后的字符集合。就像吉他调和弦。``（不懂乐理就不翻了，下面是原文）`

 （`Imagine a guitar and the process of tuning its strings to achieve a specific chord. As each string is tightened, it becomes more “in tune” with a specific note, the weight of this tightening causes other strings to require adjustment. Iterating through the strings, each time reducing errors in pitch, and eventually you arrive at a reasonably tuned instrument.`）

 想象一下，吉他的每根弦（突触）都连接着调音钉（神经元），调弦就像一个迭代的过程（训练数据）。每次调弦时，还要有微调（反向传播）以调整到合适的音调。最终乐器调弦完毕，用于演奏（预测），乐器将正确谐调（可接受的低错误率）。（不懂乐理，下面是原文）  

 `Imagine the weight of each string (synapse) connecting a series of tuning pegs (neurons) and an iterative process to achieve proper tuning (training data). In each iteration there is additional fine tuning (back-propagation) to adjust to the desired pitch. Eventually the instrument is tuned and when played (used for prediction) it will harmonize properly (have acceptably low error rates).`

 ![多层人工神经网络](https://cdn-images-1.medium.com/max/800/1*CcQPggEbLgej32mVF2lalg.png)

 让我们在程序中制造一个“玩具”人工神经网络来进行学习。例子的源代码在[这里](https://github.com/ugik/notebooks/blob/master/Simple_Neural_Network.ipynb),我们将使用Pyhton和iPythonBook。和往常一样，我们将使用“无黑匣子”的方法，以方便我们来理解工作原理。  

 我们将提供一个非常小的训练数据集，三个二进制数的集合和一个输出：

        input [ 0, 0, 1 ] output: 0
        input [ 1, 1, 1 ] output: 1
        input [ 1, 0, 1 ] output: 1
        input [ 0, 1, 0 ] output: 0

 注意，（上面的）输出（output）看起来似乎与第一个值相关，这就是模式。 例如，我们不知道[1，1，0]的输出是什么，但训练数据的模式强烈地表明输出是1。

 >“人类智慧的核心是对模式的敏感性” - 道格拉斯·霍夫斯塔特（Douglas Hofstadter）

 我们将使用Numpy库（做一些数学计算）将训练数据转化为数组列表。

        import numpy
        # The training set. We have 4 examples, each consisting of 3 input values and 1 output value.

        training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0]])

        training_set_outputs = array([[0, 1, 1, 0]]).T

 然后在一个类里面定义我们的神经网络函数，通过init方法初始化。

        class NeuralNetwork():
            def __init__(self):
                # Seed the random number generator, so it generates the same numbers
                # every time the program runs.
                random.seed(1)

                # We model a single neuron, with 3 input connections and 1 output connection.
                # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
                # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

 突触的权重开始为随机值，注意网络的维度：3x1，与我们的训练数据一致（如[0，1，0]）。

        Random starting synaptic weights: 
        [[-0.16595599]
         [ 0.44064899]
         [-0.99977125]]

 接着定义激励函数。

            # The Sigmoid function, which describes an S shaped curve.
            # We pass the weighted sum of the inputs through this function to
            # normalise them between 0 and 1.
            def __sigmoid(self, x):
                return 1 / (1 + exp(-x))

            # S函数的导数（计算梯度下降）.
            # This is the gradient of the Sigmoid curve.
            # It indicates how confident we are about the existing weight.
            def __sigmoid_derivative(self, x):
                return x * (1 - x)

 我们需要一种将结果规范化至某个范围内的方法，这里是0到1之间。

 ![Sigmoid函数](https://cdn-images-1.medium.com/max/800/1*8SJcWjxz8j7YtY6K-DWxKw.png)

 梯度（Sigmoid函数的导数）告诉我们曲线上任意一点的斜率，这有助于了解“离”（错误）“调和”有多“远”。  
 想象有这样一个吉他调谐器，它能详细告诉我们有多“不和弦”。

 ![](https://cdn-images-1.medium.com/max/600/1*sGo81thwUpgLrSOdr5SvtA.jpeg)

 现在我们准备训练（调弦）我们的网络。

        # We train the neural network through a process of trial and error.
            # Adjusting the synaptic weights each time.
            def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
                for iteration in iter(range(number_of_training_iterations)):
                    # Pass the training set through our neural network (a single neuron).
                    output = self.think(training_set_inputs)

                    # Calculate the error (The difference between the desired output
                    # and the predicted output).
                    error = training_set_outputs - output

                    # Multiply the error by the input and again by the gradient of the Sigmoid curve.
                    # This means less confident weights are adjusted more.
                    # This means inputs, which are zero, do not cause changes to the weights.
                    adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

                    # Adjust the weights.
                    self.synaptic_weights += adjustment
                    if (iteration % 1000 == 0):
                        print ("error after %s iterations: %s" % (iteration, str(numpy.mean(numpy.abs(error))))) 

 上面代码中（只有6行）的是矩阵乘法，中学有学。  
 我们可以看到每次1000次迭代后，错误率（音准程度）稳步下降。 在总共10,000次迭代之后，每次调整突触权重时，我们都能获得足够低的错误率。  

        error after 1000 iterations: 0.0353771814512
        error after 2000 iterations: 0.024323319584
        error after 3000 iterations: 0.0196075022358
        error after 4000 iterations: 0.016850233908
        error after 5000 iterations: 0.014991814044
        error after 6000 iterations: 0.0136320935305
        error after 7000 iterations: 0.01258242301
        error after 8000 iterations: 0.0117408289409
        error after 9000 iterations: 0.0110467781322 

 注意，网络过于紧密（过拟合）将不会有任何作用，就像一个太过敏感的乐器，会在演奏时走调。 合适的迭代次数通常与训练数据的维度有关。

 >“我们考虑得太多而感知得太少.” 查理·卓别林(Charlie Chaplin)

 下一部分是我的最爱：一行用来“思考”的代码。

 **sigmoid( dot ( inputs, synaptic_weights))**
    
        # The neural network 'thinks'
            def think(self, inputs):
                # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

 输入和权重的点积（矩阵乘法）的归一化值。 由此得到结果：神经网络能“思考”了（更精确地说是能“预测”了）。

 **等等，这是不是...**

 那这是不是“人工智能”呢？

 我们可以看得到0和1的数据流。这是真的。  
 现在只需要编辑代码的main部分，然后运行。

        if __name__ == "__main__":

            #Intialise a single neuron neural network.
            neural_network = NeuralNetwork()

            print ("Random starting synaptic weights: ")
            print (neural_network.synaptic_weights)

            # Train the neural network using a training set.
            # Do it 10,000 times and make small adjustments each time.
            neural_network.train(training_set_inputs, training_set_outputs, 10000)

            print ("New synaptic weights after training: ")
            print (neural_network.synaptic_weights)

            # Test the neural network with a new pattern
            test = [1, 0, 0]
            print ("Considering new situation %s -> ?: " % test )
            print (neural_network.think(array(test)))

 输出：

            Random starting synaptic weights: 
            [[-0.16595599]
             [ 0.44064899]
             [-0.99977125]]
            error after 0 iterations: 0.578374046722
            error after 1000 iterations: 0.0353771814512
            error after 2000 iterations: 0.024323319584
            error after 3000 iterations: 0.0196075022358
            error after 4000 iterations: 0.016850233908
            error after 5000 iterations: 0.014991814044
            error after 6000 iterations: 0.0136320935305
            error after 7000 iterations: 0.01258242301
            error after 8000 iterations: 0.0117408289409
            error after 9000 iterations: 0.0110467781322
            New synaptic weights after training: 
            [[ 12.79547496]
             [ -4.2162058 ]
             [ -4.21608782]]
            Considering new situation [1, 0, 0] -> ?: 
            [ 0.99999723]

##   发生了什么

 我们从4个训练数据开始：  

 input [ 0, 0, 1 ] output: 0

 input [ 1, 1, 1 ] output: 1

 input [ 1, 0, 1 ] output: 1

 input [ 0, 1, 0 ] output: 0

 然后然神经网络预测[1,0,0]的输出，它给出了0.99————有效输出1，正确。

 再让我们给网络一个输出不确定的input：

        Considering new situation [0, 0, 0] -> ?: 
        [ 0.5]

 在这里，它“认为”输出是0.5，虽然不是0，但已经接近了。我们需要更多的训练数据以及更多的迭代（训练）次数。

 厉害的是，是软件自行计算得到有用的模式，而不是人工为模式编写“规则”。我们可以轻松的编写以下的代码：

        # a new pattern applied to code
        test = [0, 0, 0]
        if test[0] == 1:
            print ([1])
        else:
            print ([0])

 人工编写规则的前提是我们可以提前理解了模式的“规则”。但在实际中这通常是不现实的。

##   深度学习

 在具有大量数据的真实分类器中，网络的维度更为显著。想象一下，一个500*7500的乘法矩阵，需要超过10万次以上的迭代次数，以及多层的神经元需要“调谐”。这就是所谓的“深度学习”。深度与网络的层数有关。
 
 试一下这个例子中的代码。注意底部的输出显示了每个矩阵与每次迭代后的结果。

 ![a 12-string harp guitar](https://cdn-images-1.medium.com/max/800/1*1CEU_YPnHAws8NpDYY_ebw.jpeg)
