import dendropy
import numpy as np
from dendropy import Tree
import networkx as nx
import scipy.stats as ss


def tree_to_newick(g: nx.DiGraph, root=None, weight=None, is_internal_call=False):
    """
    Copied from VICTree project
    """
    # make sure the graph is a tree
    assert nx.is_arborescence(g)
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    # sorting makes sure same trees have same newick
    for child in sorted(g[root]):
        node_str: str
        if len(g[child]) > 0:
            node_str = tree_to_newick(g, root=child, weight=weight, is_internal_call=True)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    newick = "(" + ','.join(subgs) + ")" + str(root)
    if not is_internal_call:
        newick += ';'
    return newick


def convert_networkx_to_dendropy(nx_tree, labels_mapping: dict = None, taxon_namespace = None, edge_length= None) -> dendropy.Tree:
    """
    Converts a NetworkX tree to a DendroPy tree through newick string.

    Args:
      nx_tree: The NetworkX tree to convert.
      labels_mapping: dict, mapping of taxa labels to new labels

    Returns:
      A DendroPy tree.
    """
    if labels_mapping is not None:
        nx_tree = nx.relabel_nodes(nx_tree, labels_mapping, copy=True)
    newick = tree_to_newick(nx_tree, weight=edge_length)
    dendropy_tree = Tree.get(data=newick, schema='newick', taxon_namespace=taxon_namespace)
    label_tree(dendropy_tree, method='group')
    dendropy_tree.is_rooted = True

    return dendropy_tree


def random_binary_tree(n: int, length_mean: float, seed=None):
    """
    Generate a random binary tree with n leaves using Dendropy.
    ref: https://dendropy.org/primer/treesims.html
    Args:
        n: Number of leaves.
        seed: Random seed.

    Returns:
        A Dendropy tree.
    """
    # set dendropy and scipy seed for reproducibility
    if seed is not None:
        dendropy.utility.GLOBAL_RNG.seed(seed)
        np.random.seed(seed)
    tns = dendropy.TaxonNamespace([dendropy.Taxon(str(i)) for i in range(n)], label='taxa')
    tree = dendropy.treesim.treesim.pure_kingman_tree(taxon_namespace=tns)
    tree.is_rooted = True
    label_tree(tree)
    # traverse the tree and assign lengths
    for edge in tree.preorder_edge_iter():
        # scale = 1 / lambda
        edge.length = ss.expon(scale=length_mean).rvs()
    return tree


def get_node2node_distance(tree: dendropy.Tree, node1_label: str, node2_label: str):
    tree.calc_node_root_distances()
    node1 = tree.find_node_with_label(node1_label)
    node2 = tree.find_node_with_label(node2_label)
    if node1.root_distance < node2.root_distance:
        node1, node2 = node2, node1
    return node1.root_distance - node2.root_distance


def label_tree(tree, method='int'):
    """
    Assigns int labels to tree nodes. Leaves are assigned with cell ids, internal nodes with decremental numbers
    different from cell ids.
    """
    if method == 'int':
        rev_node_idx = len(tree.nodes()) - 1
        for n in tree.nodes():
            if n.is_leaf():
                n.label = n.taxon.label
            else:
                n.label = str(rev_node_idx)
                rev_node_idx -= 1
    elif method == 'group':
        for n in tree.postorder_node_iter():
            if n.is_leaf():
                n.label = str(n.taxon.label)
            else:
                # group the taxa in the subtree in a sorted string
                taxa = []
                for c in n.child_node_iter():
                    for t in c.label.split('_'):
                        taxa.append(t)
                n.label = '_'.join(sorted(taxa, key=lambda x: int(x)))
    else:
        raise ValueError(f"Unknown method {method}")


