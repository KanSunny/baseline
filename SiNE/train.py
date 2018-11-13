import torch
import numpy as np
from get_data import Graph
from model import SiNE
from model import fit_model
from model import link_prediction
from torch.autograd import Variable

class TT(object):
    def __init__(self):
        self.graph = Graph.read_from_file('/home/kansunny/Documents/postgraduate/hash_community/hsne/code/dataset/epinions.txt',delimiter='	')
        self.nodes_num = len(self.graph)
        print("1. Dataset loading finished！")
        print("Number of users : %d" % self.nodes_num)
        print("Number of positive edges : %d" % len(self.graph.get_positive_edges()))
        print("Number of negative edges : %d" % len(self.graph.get_negative_edges()))
        self.training_data = self.graph.get_training_triplets()
        self.testing_data = self.graph.get_testing_triplets()
        #print()
        #print(self.testing_data)
    
    def train(self):
        sine = SiNE(self.nodes_num, 20, 20)
        fit_model(sine, self.training_data, 1, 0.5, 1, 100, 0.0001)
        print("3. Training finished！")
        torch.save(sine.state_dict(), './parameters')
        
    def test(self):
        sine1 = SiNE(self.nodes_num, 20, 20)
        sine1.load_state_dict(torch.load('./parameters'))
        print(len(self.testing_data))
        number = 0
        correct = 0
        for test in self.testing_data:
            temp = []
            temp.append(test)
            temp = np.array(temp)
            loss = link_prediction(sine1, temp, 0, 0, 1)
            if loss.data[0] == 0:
                correct += 1
            number += 1
            print("number %d is %f " % (number, correct/number))
        print(correct/number)

tt = TT()
#tt.train()
#tt.test()
