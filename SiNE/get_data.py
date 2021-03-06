import networkx as nx
import numpy as np
import random

'''
node <-> id
node is defined by dataset (contain node information)
id is a number to store
'''
class Vocabulary(object):
    def __init__(self, graph):
        self._id2node = {}# private variable '_'
        self._node2id = {}
        self._curr_id = 0
        for node in graph.nodes():
            if node not in self._node2id:
                self._curr_id += 1
                self._node2id[node] = self._curr_id
                self._id2node[self._curr_id] = node
    
    def id2node(self, id):
        return self._id2node[id]
    
    def node2id(self, node):
        return self._node2id[node]
    
    def augment(self, graph):
        for node in graph.nodes():
            if node not in self._node2id:
                self._curr_id += 1
                self._node2id[node] = self._curr_id
                self._id2node[self._curr_id] = node
    
    def __len__(self):
        return self._curr_id

    
class Graph(object):
    def __init__(self, positive_graph, negative_graph):
        self.positive_graph = positive_graph
        self.negative_graph = negative_graph
        #2. correspond node and id
        self.vocab = Vocabulary(positive_graph)
        self.vocab.augment(negative_graph)
        self._get_triplets()
        
    def get_positive_edges(self):
        return self.positive_graph.edges()
        
    def get_negative_edges(self):
        return self.negative_graph.edges()
    
    def __len__(self):
        return len(self.vocab)
    
    #3. according to graph, create triplet dataset(shown by id)
    def _get_triplets(self, p0=True, ids=True):
        self.train_triplets = []
        self.test_triplets = []
        #print(self.negative_graph.edges())
        #print(self.positive_graph.edges())
        for xi in self.positive_graph.nodes():# xi positive point
            for xj in self.positive_graph[xi]:# xj is query point
                #print(xj)
                if xj in self.negative_graph:
                    for xk in self.negative_graph[xj]:# xk is negative point
                        a, b, c = xi, xj, xk
                        if ids:
                            a = self.vocab.node2id(xi)
                            b = self.vocab.node2id(xj)
                            c = self.vocab.node2id(xk)
                            if random.random() >= 0.8:
                                self.test_triplets.append([b, a, c])
                            else:
                                self.train_triplets.append([b, a, c])
                elif p0:
                    a, b = xi, xj
                    c = 0
                    if ids:
                        a = self.vocab.node2id(xi)
                        b = self.vocab.node2id(xj)
                    if random.random() >= 0.8:
                        self.test_triplets.append([b, a, c])
                    else:
                        self.train_triplets.append([b, a, c])
        self.train_triplets = np.array(self.train_triplets)
        self.test_triplets = np.array(self.test_triplets)
        #return triplets# [query, positive, negative]
    
    def get_training_triplets(self):
        return self.train_triplets
    
    def get_testing_triplets(self):
        return self.test_triplets
    
    @staticmethod# use this function with init Graph
    def read_from_file(filepath, delimiter=' ', directed=False):
        #1. get data and combine graph
        # the relationship is directed but the distance is un-directed ?
        positive_graph = nx.DiGraph() if directed else nx.Graph()# init directed graph
        negative_graph = nx.DiGraph() if directed else nx.Graph()
        file = open(filepath)
        for line in file:
            line = line.strip()# remove ' '
            #print(line)
            line = line.split(delimiter)
            if len(line) != 3:# remove illegal
                continue
            u, v, w = line
            w = float(w)
            if w > 0:
                positive_graph.add_edge(u, v, weight=w)
            if w < 0:
                negative_graph.add_edge(u, v, weight=w)
        file.close()
        graph = Graph(positive_graph, negative_graph)
        return graph

#graph = Graph.read_from_file('/home/kansunny/Documents/postgraduate/hash_community/hsne/code/dataset/epinions.txt',delimiter='	')
#train = graph.get_testing_triplets()
#test = graph.get_testing_triplets()
#print(train)
#print(test)
