import math
import random
import copy

class HopfieldNeuralNetwork:
    def __init__(self):
        self.debug_print = False
        self.square_size = 5
        
        self.true_neurons_num = 3
        self.init_true_neuron()

        self.neuron_weight = [[0 for i in range(self.square_size**2)] for j in range(self.square_size**2)]
        self.init_neuron_weight()

    def init_true_neuron(self):
        self.true_neurons = [[HopfieldNeuralNetwork.plus_or_minus() for i in range(self.square_size**2)] for k in range(self.true_neurons_num)]
        if self.debug_print:
            print("[True Neuron]")    
            self.print_true_neuron()
            
    def init_neuron_weight(self):
        for i in range(len(self.neuron_weight)):
            for j in range(len(self.neuron_weight[i])):
                for m in range(self.true_neurons_num):
                    if i == j:
                        continue
                    self.neuron_weight[i][j] += self.true_neurons[m][i] * self.true_neurons[m][j]
                self.neuron_weight[i][j] /= 1.0 * self.true_neurons_num
                
    def add_noise(self, neuron_, rate):
        neuron = copy.deepcopy(neuron_)
        for i in range(len(neuron)):
            if random.random() < rate:
                neuron[i] = HopfieldNeuralNetwork.plus_or_minus()
        return neuron
    
    def print_neuron(self, neuron):
        for i in range(self.square_size):
            for j in range(self.square_size):
                if neuron[i * self.square_size + j] > 0:
                    print("● ", end="")
                else:
                    print("○ ", end="")
            print("")
        print("")            
            
    def print_true_neuron(self):
        for i in range(self.true_neurons_num):
            self.print_neuron(self.true_neurons[i])

    def update_neuron(self, neuron, sync=False):
        if sync:
            new_neuron = copy.deepcopy(neuron)
            for i in range(len(new_neuron)):
                val = 0
                for j in range(len(neuron)):
                    val += self.neuron_weight[i][j] * neuron[j]
                    new_neuron[i] = self.activate_function(val)
            neuron = new_neuron
        else:
            for i in range(len(neuron)):
                ii = int(random.random() * len(neuron) * 4 % len(neuron))
                val = 0
                for j in range(len(neuron)):
                    val += self.neuron_weight[ii][j] * neuron[j]
                neuron[ii] = self.activate_function(val)
        return neuron

    def optimize_neuron(self, neuron_):
        neuron = copy.deepcopy(neuron_)
        #if self.debug_print:
        #    print("[Optimize] Neuron Before Trained")
        #    self.print_neuron(neuron)

        cnt = 0
        while(True):
            pre_neuron = copy.deepcopy(neuron)            
            neuron = self.update_neuron(neuron, sync=False)
            cnt += 1
            if self.judge_if_match(pre_neuron, neuron) > 0:
                print("[Optimize] finished with {} times".format(cnt))
                break
            
        if self.debug_print:
            print("[Optimize] Neuron After Trained")
            self.print_neuron(neuron)
        return neuron

    def calc_similarity(self, neuron1, neuron2):
        sim = 0
        for i in range(len(neuron1)):
            if neuron1[i] == neuron2[i]:
                sim += 1.
        sim /= len(neuron1)
        
        print("[Similarity] {}%".format(sim * 100))
        return sim

    def judge_if_match(self, neuron1, neuron2):
        for i in range(len(neuron1)):
            if neuron1[i] != neuron2[i]:
                return -1
        return 1
    
    def activate_function(self, val):
        return HopfieldNeuralNetwork.sgn(val)

    @staticmethod
    def plus_or_minus():
        if random.random() > 0.5:
            return 1
        else:
            return -1
        
    @staticmethod
    def sgn(val):
        if val > 0:
            return 1
        else:
            return -1

if __name__ == '__main__':
    hnn = HopfieldNeuralNetwork()
    for i in range(hnn.true_neurons_num):
        neuron = hnn.add_noise(hnn.true_neurons[i], 0.5)
        neuron = hnn.optimize_neuron(neuron)
        hnn.calc_similarity(neuron, hnn.true_neurons[i])
