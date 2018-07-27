# TODO delete copy.deepcopy() if it is not needed
import math
import random
import copy

class HopfieldNeuralNetwork:
    def __init__(self, square_size=5, true_neurons_num=1, debug_print=False):
        self.debug_print = debug_print
        self.square_size = square_size
        
        self.true_neurons_num = true_neurons_num
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
            neuron = self.update_neuron(neuron, sync=False)
            cnt += 1
            if self.check_if_finished(neuron) > 0:
                if self.debug_print:
                    print("[Optimize] finished with {} times".format(cnt))
                break
            
        if self.debug_print:
            print("[Optimize] Neuron After Trained")
            self.print_neuron(neuron)
        return neuron, cnt

    def check_if_finished(self, neuron):
        new_neuron = copy.deepcopy(neuron)
        for i in range(len(new_neuron)):
            val = 0
            for j in range(len(neuron)):
                val += self.neuron_weight[i][j] * neuron[j]
                new_neuron[i] = self.activate_function(val)
                
        for i in range(len(new_neuron)):
            if new_neuron[i] != neuron[i]:
                return -1
        return 1

    def calc_similarity(self, neuron):
        sim = [0 for i in range(self.true_neurons_num)]
        for i in range(self.true_neurons_num):
            for j in range(len(neuron)):
                if neuron[j] == self.true_neurons[i][j]:
                    sim[i] += 1.
            sim[i] /= len(neuron)
        
        if self.debug_print:        
            print("[Similarity] {}%".format(max(sim) * 100))
        return max(sim)

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
    for i in range(10):
        tnn = i + 1
        hnn = HopfieldNeuralNetwork(square_size=5, true_neurons_num=tnn, debug_print=False)
    
        for j in range(0, 100+1, 5):
            sim = 0   
            sim_cnt = 0
            cnt = 0
            for k in range(100):
                neuron = hnn.add_noise(hnn.true_neurons[cnt % tnn], j * 0.01)
                neuron, po = hnn.optimize_neuron(neuron)
                sim_tmp = hnn.calc_similarity(neuron)
                sim += sim_tmp
                if int(sim_tmp) == 1:
                    sim_cnt += 1
                cnt += 1
            print("{} {} {:2.4} {:2.4}".format(tnn, j, sim * 100.0 / cnt, sim_cnt * 100.0 / cnt))
        
    '''
    tnn = 4
    hnn = HopfieldNeuralNetwork(square_size=5, true_neurons_num=tnn, debug_print=False)
    
    for i in range(tnn):
        neuron = hnn.add_noise(hnn.true_neurons[i], 0.8)
        neuron = hnn.optimize_neuron(neuron)
        sim_tmp = []
        sim_tmp = hnn.calc_similarity(neuron)
        print("{:2.4}%".format(sim_tmp * 100.0))
    '''
