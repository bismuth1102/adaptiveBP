import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import os
import time
import math

from collections import deque


class Graph:
    """ a Graph object represents a graph

    Attributes:
        graph: a "Networkx" graph
        Node: inherit Node object
        Edge: inherit Edge object
        nodes: a dictionary of all nodes in the graph
        edges: a dictionary of all edges in the graph
        bfs_label: a dictionary of node labels created by breadth-first search
        bfs_res: a list of breadth-first search result
        dfs_label: a dictionary of node labels created by depth-first search
        dfs_res: a list of depth-first search result
        degree: a dictionary of the depth for each node in the tree graph
        E: a list of Euler-tour traverses result
        D: a list of depth for each node in E
        H: a list of the first occurrences index for each node in E
        M_matrix: a numpy array of range minimum/maximum query (RMQ)

    """

    def __init__(self):
        self.graph = None
        self.Node = None
        self.Edge = None
        self.nodes = None
        self.edges = None

        self.bfs_label = None
        self.bfs_res = None

        self.dfs_label = None
        self.dfs_res = None

        self.degree = None

        self.E = None
        self.D = None
        self.H = None
        self.M_matrix = None

    def create_random_graph(self, node_num):
        """ create a random balanced tree graph given a number of nodes

        Args:
            node_num: a string
                        - 's': a small size tree, branching factor of the tree
                        is 2, and height of the tree is 3
                        - 'm': a medium size tree, branching factor of the tree
                        is 4, and height of the tree is 4
                        - 'l': a large size tree, branching factor of the tree
                        is 8, and height of the tree is 5
                     or an int
                        - height of the tree is node_nm / 2 + 1
        """
        if node_num is 's':
            G = nx.balanced_tree(2, 3)
        elif node_num is 'm':
            G = nx.balanced_tree(3, 4)
        elif node_num is 'l':
            G = nx.balanced_tree(4, 5)
        elif node_num <= 0:
            warnings.warn("node_num must be larger than 0, default to small graph")
            G = nx.balanced_tree(2, 3)
        else:
            degree = int(node_num / 2 + 1)
            G = nx.balanced_tree(node_num, degree)
        self.graph = G
        self.Node = Node(self.graph)
        self.Edge = Edge(self.graph)
        self.nodes = self.Node.n_dict
        self.edges = self.Edge.e_dict

    def create_graph(self, edges, nodes=None):
        """ create a tree graph depending on a list of edges, and a
            list of nodes. If the list of nodes is not given, create a
            tree graph depending on the list of edges

        Args:
            edges: a list of edges and each edge is a tuple: (nodeA, nodeB)
            nodes: a list of nodes. If the list of nodes is not given, create
            a tree graph depending on the list of edges
        """
        G = nx.Graph()
        if nodes is None:
            if type(edges) is tuple:
                edges = [edges]
            else:
                try:
                    G.add_edges_from(edges)
                except TypeError:
                    print("Edges must be tuple or list")
        else:
            G.add_nodes_from(nodes)
            if type(edges) is tuple:
                edges = [edges]
            else:
                try:
                    G.add_edges_from(edges)
                except TypeError:
                    print("Edges must be tuple or list")
        self.graph = G
        self.Node = Node(self.graph)
        self.Edge = Edge(self.graph)
        self.nodes = self.Node.n_dict
        self.edges = self.Edge.e_dict

    def show_graph(self, with_labels=False):
        """ Show the graph figure. If with_labels is False, the figure will
            not show the label for each node.

        Args:
            with_labels: default is False. If true, show the label for each
            node in the graph
        """
        if with_labels is False:
            nx.draw(self.graph)
        else:
            nx.draw(self.graph, with_labels=True)
        plt.show()

    def add_node_attribute(self, node='all', attribute=None, attribute_value=None):
        """ add node attribute in the node dictionary. The node can be specific or
            add an attribute to every node

        Args:
            node: a list of nodes which you want to add the new attribute, default is 'all'.
            attribute: the key in the node dictionary, default is None
            attribute_value: the value in the node dictionary, default is None
        """
        if node is 'all':
            node = list(self.graph.nodes)
        else:
            try:
                for i in node:
                    self.nodes.add_attribute(i, attribute, attribute_value)
            except TypeError:
                self.nodes.add_attribute(node, attribute, attribute_value)

    def add_edge_attribute(self, edge='all', attribute=None, attribute_value=None):
        """ add edge attribute in the edge dictionary. The edge can be specific or
            add an attribute to every edge

        Args:
            edge: a list of edges which you want to add the new attribute, default is 'all'.
            attribute: the key in the edge dictionary, default is None
            attribute_value: the value in the edge dictionary, default is None
        """
        if edge is 'all':
            edge = list(self.graph.edges)
        else:
            try:
                for i in edge:
                    self.edges.add_attribute(i, attribute, attribute_value)
            except TypeError:
                self.edges.add_attribute(edge, attribute, attribute_value)

    def bfs(self, start=None, degree=False):
        """ Return the order in which nodes are visited in a breadth-first
            traversal, starting with the given node, if the start node is
            ""None"", randomly choose one node as the root node

        Args:
            start: the breadth-first search will start at this node
            degree: whether return the node degree, default is False
        Returns:
            res: a list of the breadth-first search result
            label: a dictionary of the node searching index
            degree: a dictionary of the node depth from the root
        """
        if start is None:
            start = np.random.choice(self.graph.nodes)

        if degree is False:
            query = deque()
            query.append(start)
            seen = set()  # nodes we have already visited
            res = []
            label = {}
            label_to_node = {}
            label_num = 1
            while len(query) > 0:  # while more to visit
                n = query.popleft()
                if n not in seen:
                    res.append(n)
                    seen.add(n)
                    label[n] = label_num
                    label_to_node[label_num] = n
                    label_num = label_num + 1
                for neighbor in self.graph.neighbors(n):
                    if neighbor not in seen:
                        query.append(neighbor)
            self.bfs_label = label
            self.bfs_label_to_node = label_to_node
            self.bfs_res = res
            return res, label

        else:
            query = deque()
            degree = {}
            level = 0
            nl_index = 1
            query.append(start)
            seen = set()  # nodes we have already visited
            res = []
            label = {}
            label_to_node = {}
            label_num = 1
            while len(query) > 0:  # while more to visit
                n = query.popleft()
                if n not in seen:
                    res.append(n)
                    seen.add(n)
                    nl_index = nl_index - 1
                    label[n] = label_num
                    label_to_node[label_num] = n
                    label_num = label_num + 1
                    degree[label[n]] = level
                for neighbor in self.graph.neighbors(n):
                    if neighbor not in seen:
                        query.append(neighbor)
                if nl_index is 0:
                    nl_index = len(query)
                    level = level + 1
            self.bfs_label = label
            self.bfs_label_to_node = label_to_node
            self.degree = degree
            self.bfs_res = res
            return res, label, degree

    def dfs(self, start=None):
        """ Return the order in which nodes are visited in a depth-first
            traversal, starting with the given node, if the start node is
            ""None"", randomly choose one node as the root node

        Args:
            start: the depth-first search will start at this node
        Returns:
            res: a list of the depth-first search result
            label: a dictionary of the node searching index
        """
        if start is None:
            start = np.random.choice(self.graph.nodes)
        query = deque()
        query.append(start)
        seen = set()  # nodes we have already visited
        res = []
        label = {}
        label_to_node = {}
        label_num = 1
        while len(query) > 0:  # while more to visit
            n = query.popleft()
            if n not in seen:
                res.append(n)
                seen.add(n)
                neighbor_list = []
                label[n] = label_num
                label_to_node[label_num] = n
                label_num = label_num + 1
            for neighbor in self.graph.neighbors(n):
                if neighbor not in seen:
                    neighbor_list.append(neighbor)
            if len(neighbor_list) > 0:
                neighbor_list.reverse()
                query.extendleft(neighbor_list)
        self.dfs_label = label
        self.dfs_label_to_node = label_to_node
        self.dfs_res = res
        return res, label

    def shortest_path(self, source, target):
        """ find the shortest path from the source node to the
            target node

        Args:
            source: the beginning node of the search
            target: the final node of the search

        Returns:
            path: a list of the shortest path
        """
        path = nx.shortest_path(self.graph, source, target)
        return path

    def Euler_tour(self, start=None):
        """ find the Euler tour path, which using dfs method with the bfs_label

        Args:
            start: determine the root node in the graph

        Returns:
            E: a list of Euler tour path
            D: a list of node depth with the same order in E
        """
        if start is None:
            start = np.random.choice(self.graph.nodes)
        _, label, degree = self.bfs(start, degree=True)

        query = deque()
        query.append(start)
        seen = set()  # nodes we have already visited
        res = []
        parent_node = {}
        index = 0
        
        while len(query) > 0:  # while more to visit
            n = query[0]
            if n not in seen:
                res.append(n)
                seen.add(n)
                neighbor_list = []
                
            for neighbor in self.graph.neighbors(n):
                if neighbor not in seen:
                    neighbor_list.append(neighbor)
            if len(neighbor_list) > 0:
                neighbor_list.reverse()
                query.extendleft(neighbor_list)
                for i in neighbor_list:
                    parent_node[i] = n
            else:
                query.popleft()
                if n is start:
                    continue
                else:
                    res.append(parent_node[n])
        E = []
        D = []
        H = []
        first_occur = {}
        res_seen = set()

        for r in res:
            if r not in res_seen:
                res_seen.add(r)
                first_occur[label[r]]= index
                index = index + 1
            else:
                index = index + 1
            E.append(label[r])
            
        for i in E:
            D.append(degree[i])
        
        for j in range(1, len(first_occur)+1):
            H.append(first_occur[j])
            
        del first_occur
        
        self.E = E
        self.D = D
        self.H = H
        return E, D, H

#     def first_occurrence(self):
#         """ find every node first occurrence index. Must perform "Eular_tour" to
#             get E list first.

#         Returns:
#             H: a list of the first occurrences index for each node in E
#         """
#         H = []
#         Q = []
#         for i in range(1, max(self.E) + 1):
#             H.append(self.E.index(i))
            
#         for i in range(1, len(self.nodes) + 1):
#             Q.append(self.first_occur[i])
#         self.H = H
#         print('E:', self.E)
#         print('H:', H)
#         print('Q:', Q)
#         return H

    def RMQ(self):
        """ Range Minimum/Maximum Query algorithm. Must perform Euler_tour fist to
            obtain E. RMQ is used to find minimum/maximum value in any sub-array in
            constant time when we computed M_matrix
        Returns:
            M_matrix: RMQ minimum/maximum index matrix
        """
        N = len(self.E)
        L = math.ceil(np.log2(N)) + 1

        # init M matrix NxL as -1
        M_matrix = -np.ones([N, L], dtype=int)

        # init every M_matrix[i, 0] as D[i]
        for i in range(N):
            M_matrix[i, 0] = self.D[i]
        for j in range(1, L):
            for i in range(N):
                if (i + pow(2, j - 1)) >= N - 1:
                    r = self.D[N - 1]
                else:
                    r = M_matrix[i + pow(2, j - 1), j - 1]
                M_matrix[i, j] = min(M_matrix[i, j - 1], r)

        self.M_matrix = M_matrix
        return M_matrix

    def LCA(self, w1, w2):
        """ Lowest Common Ancestor algorithm. Must perform RMQ first to obtain
            M_matrix. LCA is used to find the lowest common parent node in the
            tree graph given arbitrary two nodes
        Args:
            w1: the previous modified node
            w2: the current modified node
        Returns:
            lca: the lowest common parent node given w1 and w2
        """
        w1_index = self.bfs_label[w1]
        w2_index = self.bfs_label[w2]

        H1_index = self.H[w1_index - 1]
        H2_index = self.H[w2_index - 1]
        if int(H1_index) < int(H2_index):
            i = H1_index
            j = H2_index
        else:
            i = H2_index
            j = H1_index

        k = int(np.log2(j - i + 1))
        s = j - pow(2, k) + 1
        min_M = min(self.M_matrix[i, k], self.M_matrix[s, k])
        index = i + self.D[i:j + 1].index(min_M)
        lca_index = self.E[index]
        lca = self.bfs_label_to_node[lca_index]
        return lca

    def LCA_shortest_path(self, source, target, lca=None):
        """ find the shortest path in lca, which is faster than shortest_path
            method

        Args:
            source: the previous node
            target: the current node
            lca: the lowest common parent node, which between the source node and
            the target node. If the lca is None, perform LCA method first
        Returns:
            path: the shortest path with lca
        """
        if lca is None:
            lca = self.LCA(source, target)
        s_l = nx.shortest_path(self.graph, source, lca)
        l_t = nx.shortest_path(self.graph, lca, target)
        path = s_l + l_t[1:]
        return path

    def save_graph(self, name=None):
        """ save model into current path and the suffix is ``.object``.
        If the file name is ``None``, automatically generate file as
        ``seconds time.object``. It is easier to open the latest model file.
            Args:
                model : the model ``object`` which you want to save
                name  : ``string``, the file name optional (default = ``None``)
                If the file name is ``None``, generate file as ``seconds time.object``
        """
        if name is None:
            for f_type in ['Node', 'Edge']:
                ticks = str(int(time.time()))
                file_name = f_type + '_' + ticks + '.object'
                model = eval('self.' + f_type)
                with open(file_name, 'wb') as model_file:
                    pickle.dump(model, model_file)
                print('Saved file {}'.format(file_name))
                model_file.close()
        else:
            for f_type in ['Node', 'Edge']:
                file_name = f_type + '_' + name + '.object'
                model = eval('self.' + f_type)
                with open(file_name, 'wb') as model_file:
                    pickle.dump(model, model_file)
                print('Saved file {}'.format(file_name))
                model_file.close()

    def load_graph(self, file_name=None):
        """ read model. If the file name is ``None``, automatically load latest file.
            Args:
                file_name : ``string``, the model file name optional (default = ``None``)
                            If the file name is ``None``, find latest model file and load
            return:
                model : ``object``
        """
        recent = 0
        if file_name is None:
            files = os.listdir()
            for file in files:
                if file.endswith('.object'):
                    temp = file.split('_')[1]
                    date = int(temp.split('.')[0])
                    if date > recent:
                        recent = date
            with open('Node_' + str(recent) + '.object', 'rb') as model_file:
                opened_model = pickle.load(model_file)
                model_file.close()
            return opened_model
        else:
            with open(file_name, 'rb') as model_file:
                opened_model = pickle.load(model_file)
                model_file.close()
            return opened_model


class Node:
    """ a Node object represents all nodes on the graph.

    Attributes:
        graph: the graph which which created by "Networkx".
        nodes: get all nodes in the graph.
        n_dict: a dictionary of all nodes which contains attributes for each node.

    """

    def __init__(self, G):
        """ create the attributes

        Args:
            G: a "Networkx" graph
        """
        # "Networkx" graph
        self.graph = G

        # all nodes in the graph
        self.nodes = G.nodes

        # a dictionary to store node attributes
        self.n_dict = {}

        self.init_dict()
        self.add_neighbors()

    def init_dict(self):
        for i in self.nodes:
            self.n_dict[i] = {}
        return self.n_dict

    def add_attribute(self, node, attribute, attribute_value):
        self.n_dict[node].update({attribute: attribute_value})

    def del_attribute(self, node, attribute):
        del self.n_dict[node][attribute]

    def neighbors(self, node):
        return list(self.graph.neighbors(node))

    def add_neighbors(self):
        for i in self.nodes:
            neighbor = self.neighbors(i)
            self.add_attribute(i, attribute='neighbors', attribute_value=neighbor)

    def get_nodes(self):
        return self.n_dict


class Edge:
    """ an Edge object represents all edges on the graph

    Attributes:
        edges: get all nodes in the graph.
        e_dict: a dictionary of all edges which contains attributes for each edge.
    """

    def __init__(self, G):
        """ create the attributes

        Args:
            G: a "Networkx" graph
        """
        self.edges = G.edges
        self.e_dict = {}
        self.init_dict()

    def init_dict(self):
        for i in self.edges:
            self.e_dict[i] = {}
        return self.e_dict

    def add_attribute(self, edge, attribute, attribute_value):
        self.e_dict[edge].update({attribute: attribute_value})

    def del_attribute(self, edge, attribute):
        del self.e_dict[edge][attribute]

    def get_edges(self):
        return self.e_dict


class BPGenerator:
    """ generate a random tree graph which suitable for implementing
        adaptive-belief propagation.
    Attributes:
        graph: create a Graph object
        node_num: the number of nodes you want to create
        eps: prevent underflow in creating potential matrix
    """
    def __init__(self, node_num='s', eps=1e-5, potential_type=1):
        self.graph = Graph()
        self.node_num = node_num
        self.eps = eps
        self.potential_type = potential_type
        self.create_graph()
        self.potential = self.potential_generator()
        self.node_prior()
        self.edge_potential()
        self.init_outgoing()


    def create_graph(self):
        self.graph.create_random_graph(self.node_num)

    def node_prior(self):
        for i in self.graph.Node.nodes:
            prior_value = np.random.random_sample()
            prior_matrix = np.array(np.log([prior_value, 1 - prior_value]))
            self.graph.Node.add_attribute(node=i, attribute='prior', attribute_value=prior_matrix)

    def edge_potential(self):
        for j in self.graph.Edge.edges:
            self.graph.Edge.add_attribute(edge=j, attribute='p_type', attribute_value=1)

    def potential_generator(self):
        matrix = np.asarray([[1 - self.eps, self.eps], [self.eps, 1 - self.eps]])
        potential = {}
        potential[self.potential_type] = matrix
        return potential

    def init_outgoing(self):
        outgoing_matrix = np.array([0., 0.])
        for k, v in self.graph.nodes.items():
            out_dict = {}
            for i in v['neighbors']:
                out_dict[i] = outgoing_matrix
            self.graph.Node.add_attribute(node=k, attribute='outgoing', attribute_value=out_dict)
        
    def nodes(self):
        return self.graph.nodes

    def edges(self):
        return self.graph.edges

    def save_graph(self, name=None):
        self.graph.save_graph(name)
        if name is None:
            ticks = str(int(time.time()))
            f_type = 'potential_'
            file_name = f_type + ticks + '.object'
            with open(file_name, 'wb') as model_file:
                pickle.dump(self.potential, model_file)
            print('Saved file {}'.format(file_name))
        else:
            file_name = name + '.object'
            with open(file_name, 'wb') as model_file:
                pickle.dump(self.potential, model_file)
            print('Saved file {}'.format(file_name))
            model_file.close()

    def load_graph(self, name=None):
        opened_model = self.graph.load_graph(name)
        return opened_model


# testing XGraph

if __name__ == '__main__':
    # Test BPGenerator
    print("Testing BPGenerator")
    bp =BPGenerator()
    print('-'*80)
    print('bp.graph.nodes:')
    print(bp.graph.nodes)
    print('-'*80)
    print('bp.graph.edges')
    print(bp.graph.edges)
    print(bp.potential)
    print(bp.graph.Edge.e_dict)
    print("\n\n")
    print("Testing Graph")
    G = Graph()
    G.create_graph([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('D', 'G'), ('D', 'H'), ('C', 'F'), ('F', 'I')])
    E, D = G.Euler_tour(start='A')
    print(G.first_occurrence())
    print(G.degree)
    G.show_graph(with_labels=True)
    print(G.degree)
    print(G.RMQ())
    print(G.LCA('H', 'G'))
    print(G.LCA_shortest_path('H', 'I'))
    print(G.E)
    print(G.D)
    print(G.H)
    print(G.bfs_label)
    print("Done!!!!")


