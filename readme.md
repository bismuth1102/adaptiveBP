# XGraph.py
All the objects are based on networkx (https://networkx.github.io/documentation/stable/)

# class Node
	Each node is an object
	Can add/delete any attributes
	All the nodes are saved in dict

# class Edge
	Each edge is and object
	Can add/delete any attributes
	All the edges are saved in dict

# class Graph
	Create any tree graphs
	Use bfs or dfs retrieval
	Find the shortest path between any two nodes
	Find euler tour
	Build minimum/maximum query algorithm matrix (RMQ) based on Euler tour
	save/read graph model

	# class BPGenerator
	Build a random tree graph which can be used to implement adaptive-belief propagation

# XBeliefProgation.py
	Implement standard BP and adaptive BP

E: a list of Euler-tour traverses result
	res转成label序列，E是label的排序
D: a list of depth for each node in E
	degree序列
H: a list of the first occurrences index for each node in E.
	bfs中每个label，在Euler_tour中是第几个出现的

