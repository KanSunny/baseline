import torch
import numpy as np
from get_data import Graph
from model import SiNE
from model import fit_model
from torch.autograd import Variable

class TT(object):
    def __init__(self):
        self.graph = Graph.read_from_file('/home/gjn/hsne/dataset/epinions.txt',delimiter='	')
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
        self.sine = SiNE(self.nodes_num, 40, 20)
        fit_model(self.sine, self.training_data, 1, 0.5, len(self.training_data), 500, 0.0001)
        print("3. Training finished！")
        torch.save(self.sine.state_dict(), './epinions_parameters')
        
    def test_facc(self):# facc=0.949152
        sine1 = SiNE(self.nodes_num, 40, 20)
        sine1.load_state_dict(torch.load('./epinions_parameters',map_location=lambda storage, loc: storage))
        number = 0
        correct = 0
        for test in self.testing_data:
            test = torch.from_numpy(test)
            xi = test[0]
            xj = test[1]
            xk = test[2]
            if sine1.get_edge_feature(xi, xj).data[0,0] < 0:
                correct += 1
            if sine1.get_edge_feature(xi, xk).data[0,0] > 0:
                correct += 1
            number += 2
            #print(sine1.get_edge_feature(xi, xj).data[0,0])
            #print(sine1.get_edge_feature(xi, xk).data[0,0])
            print("the facc: number %d is %f" % (number, correct/number))
            
    def test(self, operation='hadamard'):
        sine1 = SiNE(self.nodes_num, 40, 20)
        sine1.load_state_dict(torch.load('./epinions_parameters',map_location=lambda storage, loc: storage))
        number = 0
        correct = 0
        for test in self.testing_data:
            test = torch.from_numpy(test)
            xi = test[0]
            xj = test[1]
            xk = test[2]
            print(np.sum(sine1.get_distance(xi, xj, operation)))
            print(np.sum(sine1.get_distance(xi, xk, operation)))
	    #print("the facc: number %d is %f" % (number, correct/number))
        
tt = TT()
#tt.train()
tt.test_facc()
#tt.test('l2')
