from datetime import datetime
import pickle
import numpy as np
import dgl
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from pathlib import Path


def parse_adjlist(adjlist, edge_metapath_indices, mode, exclude=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(" ")))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            neighbors = np.array(row_parsed[1:])
            ind = indices
            if exclude is not None:
                if mode == 0:
                    mask = [
                        False if [d1, p1] in exclude or [d2, p2] in exclude else True
                        for d1, p1, d2, p2 in indices[:, [0, 1, -1, -2]]
                    ]
                else:
                    mask = [
                        False if [p1, d1] in exclude or [p2, d2] in exclude else True
                        for d1, p1, d2, p2 in indices[:, [0, 1, -1, -2]]
                    ]
                neighbors = neighbors[mask]
                result_indices.append(ind[mask])
            else:
                result_indices.append(ind)
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch(
    adjlists_dp,
    edge_metapath_indices_list_dp,
    drug_protein_batch,
    device,
    useMasks=None,
):
    """
    This method is used for getting metapath-based subgraphs. This function gives us a list of subgraphs concerning big graph.
    Although we don't pass the big graph as input to this method,
    this function uses the `adjlists_dp` parameter to extract the metapath-based subgraph. Essentially this function builds subgraphs.
    There are two groups of subgraphs. One group of subgraphs contains graphs that the starting node of the used metapath is the drug.
    The other one is a group of subgraphs that the starting node of the used metapath is the target.
    In additional, this method give us the mapped indices after subgraphs creation.

    To create a graph, according to the type of metapath, it first determines whether the graph belongs to the category of drugs or targets.
    After specifying the type of metapath, the adjacency list of the graph is created. After the creation the graph, edges and nodes are added to the graph.
    And the constructed graph is a subgraph of the original graph. We store this subgraph in its list.
    Also, the new indices obtained from the constructed subgraph are stored in the corresponding list.

    Note: for reducing computing costs, we prefer the subgraphs to be homogenous.
    Feature selecting in the section of metapath-encoding is going to use the metapath indices variable.
    """
    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]

    for k in adjlists_dp.keys():
        if k in [
            "drug_protein_drug",
            "drug_se_drug",
            "drug_disease_drug",
            "drug_drug",
            "drug_sim",
        ]:
            mode = 0
        elif k in [
            "protein_disease_protein",
            "protein_drug_protein",
            "protein_protein",
            "protein_sim",
        ]:
            mode = 1
        alist = adjlists_dp[k]
        e_list = edge_metapath_indices_list_dp[k]
        if useMasks[k]:
            output = parse_adjlist(
                [alist[row[mode]] for row in drug_protein_batch],
                [e_list[row[mode]] for row in drug_protein_batch],
                mode,
                exclude=drug_protein_batch,
            )
        else:
            output = parse_adjlist(
                [alist[row[mode]] for row in drug_protein_batch],
                [e_list[row[mode]] for row in drug_protein_batch],
                mode,
            )
        edges, result_indices, num_nodes, mapping = output
        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_nodes)
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        g_lists[mode].append(g)
        result_indices_lists[mode].append(result_indices)
        idx_batch_mapped_lists[mode].append(
            np.array([mapping[row[mode]] for row in drug_protein_batch])
        )

    return g_lists, result_indices_lists, idx_batch_mapped_lists


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return self.indices[
            (self.iter_counter - 1)
            * self.batch_size : self.iter_counter
            * self.batch_size
        ]

    def num_iterations(self):
        return int(self.num_data // self.batch_size)

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0


class SaveEvaluation:
    def __init__(self, dirPath) -> None:
        self.dirPath = dirPath
        Path(dirPath).mkdir(parents=True, exist_ok=True)
        Path(dirPath + "auc/").mkdir(parents=True, exist_ok=True)
        Path(dirPath + "aupr/").mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.date = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    def save_roc_curve_info(self, y_true, y_prob, epoch):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        roc_name = (
            self.dirPath
            + "auc/"
            + self.date
            + "-"
            + str(self.counter)
            + "-"
            + str(epoch)
            + "-roc_curve.txt"
        )
        with open(roc_name, "wb") as file:
            pickle.dump(data, file)

    def update_counter(self):
        self.counter += 1

    def save_aupr_curve_info(self, y_true, y_prob, epoch):

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        data = {"precision": precision, "recall": recall, "thresholds": thresholds}
        aupr_name = (
            self.dirPath
            + "aupr/"
            + self.date
            + "-"
            + str(self.counter)
            + "-"
            + str(epoch)
            + "-aupr_curve.txt"
        )
        with open(aupr_name, "ab") as file:
            pickle.dump(data, file)

    def get_auc(self, x, y):
        return auc(x, y)
