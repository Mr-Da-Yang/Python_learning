import numpy
import scipy.special
class neurlNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate
        self.wih =(numpy.random.rand(self.hnodes,self.inodes)-0.5)
        self.who =(numpy.random.rand(self.onodes,self.hnodes)-0.5)
        self.activation_fuction = lambda x: scipy.special.expit(x)
        pass



    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T       #计算输出
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_fuction(hidden_inputs)

        final_inputs = numpy.dot(self.hnodes, hidden_outputs)
        final_outputs = self.activation_fuction(final_inputs)

        output_errors = targets-final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr*numpy.dot((output_errors* final_outputs*(1-final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_outputs * hidden_outputs * (1 - hidden_outputs)),numpy.transpose(inputs))
        pass


    def query(self,inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_fuction(hidden_inputs)
        final_inputs = numpy.dot(self.hnodes,hidden_outputs)
        final_outputs = self.activation_fuction(final_inputs)
        return final_outputs



n=neurlNetwork(3,3,3,0.3)
n.query([1,0.5,-1.5])