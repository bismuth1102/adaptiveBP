from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
from XGraph import *


class BeliefPropagation:

    def __init__(self, nodes, edges, potential, max_iters=1):
        self.nodes = nodes
        self.edges = edges
        self.potentials = potential
        self.max_iters = max_iters
        self.num_classes = 2
        self.graph = self.init_graph()
        self.init_adaptive_bp = False
        self.init_standard_bp = False
        self.modified_nodes = tuple()
        self.sorted_nodes_list = []

    def init_graph(self):
        G = Graph()
        G.create_graph(self.edges.edges)
        return G

    def sorted_nodes(self):
        """
            Sort the nodes in descending order of their degrees
        """
        items = [(k, len(n['neighbors'])) for k, n in self.nodes.n_dict.items()]
        items = sorted(items, key=lambda x: x[1], reverse=True)
        self.sorted_nodes_list = [name for name, _ in items]
        return [name for name, _ in items]

    def standard_bp(self, save_model=False, tol=1e-3, show_result=True):
        schedule = self.sorted_nodes()
        self.run_bp(schedule, tol, show_result)
        self.init_standard_bp = True
        if save_model is True:
            self.save_model()

    def init_ad_bp(self, tol=1e-3):
        if self.init_standard_bp is False:
            print("Must have standard BP model to perform adaptive BP")
            print("Do standard BP now")
            self.standard_bp(True, tol)
        print('Begining Euler_tour...')
        self.graph.Euler_tour(start=self.sorted_nodes_list[0])
        print('Euler_tour finished')
        print('Begining RMQ')
        self.graph.RMQ()
        self.init_adaptive_bp = True
        print('Init adaptive BP finished')

    def LCA(self, wl1, wl2):
        try:
            return self.graph.LCA(wl1, wl2)
        except KeyError:
            print('Cannot find the path between {} and {}'.format(wl1, wl2))

    def LCA_shortest_path(self, wl1, wl2):
        try:
            return self.graph.LCA_shortest_path(wl1, wl2)
        except KeyError:
            print('Cannot find the path between {} and {}'.format(wl1, wl2))
            return []

    def modify_node(self, node, value):
        self.nodes.n_dict[node]['prior'] = value
        if len(self.modified_nodes) is 0:
            self.modified_nodes = (node,)
        elif len(self.modified_nodes) is 1:
            self.modified_nodes = (self.modified_nodes[0], node)
        else:
            self.modified_nodes = (self.modified_nodes[1], node)

    def interest_node(self, v):
        path1 = self.LCA_shortest_path(self.modified_nodes[0], self.modified_nodes[1])
        if self.modified_nodes[1] == v:
            total_path = path1
        else:
            path2 = self.LCA_shortest_path(self.modified_nodes[1], v)
            total_path = path1[:-1] + path2

        if len(total_path) is 0:
            print('Cannot find any path between modified node and visible node')
            return self.get_belief(v)
        for n in total_path:
            self.recompute_outgoing(n)
        new_belief, _ = self.get_belief(v)
        return new_belief

    def run_bp(self, schedule, tol=1e-3, show_result=True):
        for it in range(self.max_iters):
            delta = 0
            for n in schedule:
                delta += self.recompute_outgoing(n)
            delta /= len(self.nodes.n_dict)
            if show_result is True:
                print('%d-th times' % (it + 1))
                print('difference in messages: %f' % delta)
                if abs(delta) < tol:
                    print('BP has converged.')
                    break
            else:
                if abs(delta) < tol:
                    break

    def get_belief(self, node):
        """ 
            return the belief of the node, along with the messages used to compute the belief
        """
        incoming = []

        # log 1 = 0
        belief = np.zeros(self.num_classes)

        # add log of phi
        belief += self.nodes.n_dict[node]['prior']

        # go through each neighbor of the node
        for node_id in self.nodes.n_dict[node]['neighbors']:
            # get the message sent from the neighbor n to the current node (self._name)

            # look up the neighboring node in all_nodes
            n = self.nodes.n_dict[node_id]

            # getting message from the neighboring node to this node
            # consider working in the log scale to prevent underflowing

            # sum log m_ij
            belief += n['outgoing'][node]

            # in the same order as self._neighbors
            incoming.append(n['outgoing'][node])

            # print (n.get_message_for(self._name))
        return belief, incoming

    def recompute_outgoing(self, node):
        """ 
            for each neighbor j, update the message sent to j
        """

        # return value
        diff = 0

        # the messages in incoming is in the same order of self._neighbors
        # total = log phi_i + sum_{j~i} log m_ji
        # incoming = [log m_ji]
        total, incoming = self.get_belief(node)

        # go through each neighbor of the node
        for j, n_id in enumerate(self.nodes.n_dict[node]['neighbors']):

            # retrieve the actual neighboring node
            n = self.nodes.n_dict[n_id]

            # log phi_i + \sum_{k~j} log m_ki
            log_m_i = total - incoming[j]

            try:
                edge_type = self.edges.e_dict[(n_id, node)]['p_type']
            except KeyError:
                edge_type = self.edges.e_dict[(node, n_id)]['p_type']

            # log H, where H is symmetric and there is no need to transpose it
            log_H = self.potentials[edge_type]
            # \sum_y log_H(x,y) + log phi_i + sum_{k~j} log m_ki(y)
            log_m_ij = logsumexp(log_H + np.tile(log_m_i.transpose(), (2, 1)), axis=1)

            # normalize the message
            log_Z = logsumexp(log_H + np.tile(log_m_i.transpose(), (2, 1)))
            log_m_ij -= log_Z

            # accumulate the difference
            diff += np.sum(np.abs(self.nodes.n_dict[node]['outgoing'][n_id] - log_m_ij))

            # set the message from i to j
            self.nodes.n_dict[node]['outgoing'][n_id] = log_m_ij
            # break

        return diff

    def classify(self):
        """ read out the id of the maximal entry of each belief vector """
        predictions = {}
        for k in self.nodes.n_dict.keys():
            belief, _ = self.get_belief(k)
            # from log scale to prob scale
            posterior = np.exp(belief)
            # normalizing
            predictions[k] = posterior[1] / np.sum(posterior)
            # max_idx = np.argmax(belief)
            # predictions.append((belief[max_idx], max_idx))
        return predictions

    def save_model(self, file_name='bp_model.object'):
        with open(file_name, 'wb') as model_file:
            pickle.dump(self, model_file)
        model_file.close()
        print("Model has been saved as {}".format(file_name))


def evaluate(y, pred_y):
    """Evaluate the prediction of account and review by SpEagle
    Args:
        y: dictionary with key = user_id/review_id and value = ground truth (1 means spam, 0 means non-spam)

        pred_y: dictionary with key = user_id/review_id and value = p(y=spam | x) produced by SpEagle.
                the keys in pred_y must be a subset of the keys in y
    """
    posteriors = []
    ground_truth = []

    for k, v in pred_y.items():
        posteriors.append(v)
        ground_truth.append(y[k])

    auc = roc_auc_score(ground_truth, posteriors)
    return auc


def init_unit_test(graph_list):
    for g in graph_list:
        n = eval(g + '.graph.Node')
        e = eval(g + '.graph.Edge')
        p = eval(g + '.potential')

        g_bp = BeliefPropagation(n, e, p, max_iters=10)
        g_bp.standard_bp()
        g_bp.init_ad_bp()
        g_bp.save_model(file_name=str(g) + '.object')


def do_unit_test(model_list):
    res = {}
    for m in model_list:
        result = {}
        with open(m, 'rb') as model_file:
            bp = pickle.load(model_file)
            model_file.close()
        all_nodes = bp.graph.Node.n_dict
        w_seq, v_seq, w_value = wv_sequence(all_nodes)
        
        # Original Belief
        ori_belief = []
        for v in range(1,len(v_seq)):
            belief, _ = bp.get_belief(v_seq[v])
            ori_belief.append(belief)

        # standard BP
        with open(m, 'rb') as model_file:
            sbp = pickle.load(model_file)
            model_file.close()
        standard_belief = []
        for w in range(1,len(w_seq)):
            sbp.nodes.n_dict[w_seq[w-1]]['prior'] = w_value[w-1]
            sbp.nodes.n_dict[w_seq[w]]['prior'] = w_value[w]
            sbp.standard_bp(show_result=False)
            belief, _ = sbp.get_belief(v_seq[w])
            standard_belief.append(belief)

        # adaptive BP
        with open(m, 'rb') as model_file:
            abp = pickle.load(model_file)
            model_file.close()
        adaptive_belief = []
        for w in range(1,len(w_seq)):
            abp.modify_node(w_seq[w-1], w_value[w-1])
            abp.modify_node(w_seq[w], w_value[w])
            belief = abp.interest_node(v_seq[w])
            adaptive_belief.append(belief)
        result['ori_belief'] = ori_belief
        result['standard_belief'] = standard_belief
        result['adaptive_belief'] = adaptive_belief
        result['o_s_diff'] = np.sum(np.abs(np.subtract(ori_belief, standard_belief)))
        result['o_a_diff'] = np.sum(np.abs(np.subtract(ori_belief, adaptive_belief)))
        result['s_a_diff'] = np.sum(np.abs(np.subtract(standard_belief, adaptive_belief)))
        res[str(m)] = result
    return res


def wv_sequence(all_nodes, node_per=0.02):
    w_seq = []
    v_seq = []
    node_num = 10
    node_list = list(all_nodes.keys())
    for i in range(node_num):
        choice_node = np.random.choice(node_list)
        node_list.remove(choice_node)
        w_seq.append(choice_node)

    node_list = list(all_nodes.keys())
    for i in range(node_num):
        choice_node = np.random.choice(node_list)
        node_list.remove(choice_node)
        v_seq.append(choice_node)
    v = np.random.random(len(w_seq))
    w_value = np.transpose(np.tile(v, (2, 1)))
    w_value[:, 1] = 1 - w_value[:, 1]
    w_value = np.log(w_value)
    return w_seq, v_seq, w_value


def adaptive_bp(w1, p1, w2, p2, v, model_name='G2.object', with_labels=True):
    with open(model_name, 'rb') as model_file:
        bp = pickle.load(model_file)
        model_file.close()
    with open(model_name, 'rb') as model_file:
        sbp = pickle.load(model_file)
        model_file.close()
    with open(model_name, 'rb') as model_file:
        abp = pickle.load(model_file)
        model_file.close()
    sbp.graph.show_graph(with_labels=with_labels)

    result = {}
    # Setting Prior
    pri1 = set_new_prior(p1)
    pri2 = set_new_prior(p2)

    # Standard Belief Propagation
    sbp.nodes.n_dict[w1]['prior'] = pri1
    sbp.nodes.n_dict[w2]['prior'] = pri2
    sbp.standard_bp(show_result=False)
    belief, _ = sbp.get_belief(v)
    result['Standard_BP'] = belief

    # Adaptive Belief Propagation
    abp.modify_node(w1, pri1)
    abp.modify_node(w2, pri2)
    res_a = abp.interest_node(v)
    result['Adaptive_BP'] = res_a

    ori_b, _ = bp.get_belief(v)
    # Original belief
    result['Original_belief'] = ori_b

    return result


def set_new_prior(prior):
    p = 1 - prior
    pr = np.array(np.log([prior, p]))
    return pr


if __name__ == '__main__':
    # Test Adaptive BP
    print("Testing adaptiveBP")
    G1 = BP_Generator('s')
    G2 = BP_Generator('m')
    G3 = BP_Generator('l')

    graph_list = ['G1', 'G2', 'G3']
    init_unit_test(graph_list)

    print()
    print("Do unit test")
    model_list = ['G1.object', 'G2.object', 'G3.object']
    res = do_unit_test(model_list)
    print("*" * 80)
    print("G1, G2, G3 are three different size tree graphs")
    print("Each graph will randomly change ten nodes' prior")
    print("ori_belief shows the ten nodes belief in the original graph")
    print("standard_belief shows the ten nodes belief in the standard belief propagation")
    print("adaptive_belief shows the ten nodes belief in the adaptive belief propagation")
    print("o_s_diff shows the belief difference between the original graph and standard belief propagation")
    print("o_a_diff shows the belief difference between the original graph and adaptive belief propagation")
    print("s_a_diff shows the belief difference between the standard and adaptive belief propagation")
    print("-" * 80)
    print("G1.object.o_s_diff", res['G1.object']['o_s_diff'])
    print("G1.object.o_a_diff", res['G1.object']['o_a_diff'])
    print("G1.object.s_a_diff", res['G1.object']['s_a_diff'])
    print()
    print("G2.object.o_s_diff", res['G2.object']['o_s_diff'])
    print("G2.object.o_a_diff", res['G2.object']['o_a_diff'])
    print("G2.object.s_a_diff", res['G2.object']['s_a_diff'])
    print()
    print("G3.object.o_s_diff", res['G3.object']['o_s_diff'])
    print("G3.object.o_a_diff", res['G3.object']['o_a_diff'])
    print("G3.object.s_a_diff", res['G3.object']['s_a_diff'])
    print("Done!!!!")