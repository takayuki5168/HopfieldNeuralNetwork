import math
import random

class HopfieldNeuralNetwork:
    def __init__(self):
        self.true_neurons_num = 1
        self.true_neurons = [[self.plus_or_minus() for i in range(25)] for k in range(self.true_neurons_num)]

        self.neuron = [self.plus_or_minus() for i in range(25)]
        self.neuron_weight = [[0 for i in range(25)] for j in range(25)]

        for i in range(len(self.neuron_weight)):
            for j in range(len(self.neuron_weight[i])):
                for m in range(self.true_neurons_num):
                    if i == j:
                        continue
                    self.neuron_weight[i][j] += self.true_neurons[m][i] * self.true_neurons[m][j]
                self.neuron_weight[i][j] /= 1.0 * self.true_neurons_num
        print(self.neuron_weight)

    def plus_or_minus(self):
        if random.random() > 0.5:
            return 1
        else:
            return -1
        
    def add_noise(self, neuron, rate):
        for i in range(len(neuron)):
            if random.random() < rate:
                neuron[i] = neuron[i] * -1
        return neuron
    
    def print_neuron(self, neuron):
        for i in range(int(math.sqrt(len(neuron)))):
            for j in range(int(math.sqrt(len(neuron)))):
                if neuron[i * int(math.sqrt(len(neuron))) + j] > 0:
                    print("● ", end="")
                else:
                    print("○ ", end="")
            print("")
        

if __name__ == '__main__':
    hnn = HopfieldNeuralNetwork()
    hnn.print_neuron(hnn.neuron)
