# XGraph.py
所有object 基于 networkx (https://networkx.github.io/documentation/stable/)

	# class Node
	每个node都是一个object
	可以增加/删减任意attribute
	所有的nodes都存于dict里

	# class Edge
	每个edge都是一个object
	可以增加/删减任意attribute
	所有的edgess都存于dict里

	# class Graph
	创建任意tree graph
	可以进行bfs dfs 检索
	查找任意点之间的shortest path
	查找Euler tour
	基于Euler tour 建立minimum/maximum query algorithm matrix（RMQ）
	存/读graph model

	# class BPGenerator
	创建一个随机的tree graph 可以用于实现adaptive-belief propagation

# XBeliefProgation.py
	实现standard BP 和 adaptive BP

# 示例代码在graph_generator.ipynb和BeliefPropagation.ipynb

E: a list of Euler-tour traverses result
	res转成label序列，E是label的排序
D: a list of depth for each node in E
	degree序列
H: a list of the first occurrences index for each node in E.
	bfs中每个label，在Euler_tour中是第几个出现的

