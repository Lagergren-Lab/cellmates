import random
from io import StringIO
from itertools import combinations

import dendropy as dpy
from dendropy.calculate import treecompare
import skbio
import numpy as np
import networkx as nx
import scipy.stats as ss
from Bio import Phylo
from sklearn.metrics import f1_score
from collections import defaultdict

from cellmates.utils import math_utils


def nxtree_to_newick(g: nx.DiGraph, root=None, weight=None, is_internal_call=False):
    # make sure the graph is a tree
    assert nx.is_arborescence(g)
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    # sorting makes sure same trees have same newick (if node names are comparable)
    child_list = sorted(g[root]) if all(isinstance(n, int) for n in g[root]) else list(g[root])
    for child in child_list:
        node_str: str
        if len(g[child]) > 0:
            node_str = nxtree_to_newick(g, root=child, weight=weight, is_internal_call=True)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    newick = "(" + ','.join(subgs) + ")" + str(root)
    if not is_internal_call:
        newick += ';'
    return newick


def convert_networkx_to_dendropy(nx_tree, labels_mapping: dict = None,
                                 taxon_namespace=None, edge_length=None, internal_nodes_label='group') -> dpy.Tree:
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
    newick = nxtree_to_newick(nx_tree, weight=edge_length)
    dendropy_tree = dpy.Tree.get(data=newick, schema='newick', taxon_namespace=taxon_namespace)
    label_tree(dendropy_tree, method=internal_nodes_label)
    dendropy_tree.is_rooted = True

    return dendropy_tree

def convert_dendropy_to_networkx(dendropy_tree: dpy.Tree, edge_attr='weight') -> nx.DiGraph:
    """
    Converts a DendroPy tree to a NetworkX tree through newick string.

    Args:
      dendropy_tree: The DendroPy tree to convert.

    Returns:
      A NetworkX tree.
    """
    newick = dendropy_tree.as_string(schema='newick')
    nx_tree = newick_to_nx(newick, edge_attr=edge_attr)
    return nx_tree


def random_binary_tree(n: int, length_mean: float = None, seed=None, full_length: float = None)-> dpy.Tree:
    """
    Generate a random binary tree with n leaves using Dendropy.
    ref: https://dendropy.org/primer/treesims.html
    Args:
        n: Number of leaves.
        seed: Random seed.
        length_mean: Mean branch length.
        full_length: If provided, scale the tree to have this total length, ignoring length_mean.

    Returns:
        A Dendropy tree.
    """
    # set dendropy and scipy seed for reproducibility
    if seed is not None:
        dpy.utility.GLOBAL_RNG.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    if full_length is not None and length_mean is not None:
        raise ValueError("Cannot specify both full_length and length_mean")
    # create taxon namespace
    tns = dpy.TaxonNamespace([dpy.Taxon(str(i)) for i in range(n)], label='taxa')
    # tree = dpy.treesim.treesim.pure_kingman_tree(taxon_namespace=tns)
    tree = dpy.treesim.treesim.uniform_pure_birth_tree(tns, birth_rate=1.0)
    tree.is_rooted = True
    label_tree(tree)
    # traverse the tree and assign _lengths
    if full_length is not None:
        tree = make_ultrametric(tree, target_height=full_length)
    else:
        for edge in tree.preorder_edge_iter():
            if full_length is not None:
                length_mean = edge.length # use current length as mean and add some noise
            # scale = 1 / lambda
            edge.length = ss.expon(scale=length_mean).rvs()
    return tree

def make_ultrametric(tree: dpy.Tree, target_height):
    # compute current leaf distances
    max_height = tree.max_distance_from_root()

    scale_factor = target_height / max_height

    # scale every edge
    for node in tree.postorder_node_iter():
        if node.edge.length is not None:
            node.edge.length *= scale_factor

    return tree

def get_node2node_distance(tree: dpy.Tree, node1_label: str, node2_label: str):
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
                n.label = str(n.taxon.label)
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

def newick_to_nx(nwk_str, edge_attr='weight', interior_node_names=None) -> nx.DiGraph:
    """
    Parameters
    ----------
    nwk_str: str, newick string
    edge_attr: str, edge attribute in which to store the weights

    Returns
    -------
    nx.DiGraph tree with nodes and weighted edges
    """
    tree = Phylo.read(StringIO(nwk_str), 'newick')
    und_tree_nx = Phylo.to_networkx(tree)
    # Phylo names add unwanted information in unstructured way
    # find node numbers and relabel nx tree
    if interior_node_names is not None:
        for cl in und_tree_nx.nodes:
            if cl.name is None:
                cl.name = interior_node_names.pop(0)
    names_string = list(str(cl.confidence) if cl.name is None else cl.name for cl in und_tree_nx.nodes)
    try:
        names = list(map(int, names_string))
    except ValueError:
        names = names_string
    mapping = dict(zip(und_tree_nx, names))
    relabeled_tree = nx.relabel_nodes(und_tree_nx, mapping)
    tree_nx = nx.DiGraph()
    tree_nx.add_weighted_edges_from(relabeled_tree.edges(data='weight'), weight=edge_attr)
    return tree_nx

def write_cells_to_tree(nx_tree, cell_names) -> nx.DiGraph:
    # relabel leaf nodes with cell ids from adata
    # add ancestor nodes names with breadth-first search
    assert nx.is_arborescence(nx_tree)
    root_node = list(filter(lambda p: p[1] == 0, nx_tree.in_degree()))[0][0]
    mapping = {root_node: "root"}
    count = 1
    for u, v in nx.bfs_edges(nx_tree, source=root_node):
        if nx_tree.out_degree(v) == 0:
            mapping[v] = cell_names[int(v)] # leaf node
        else:
            mapping[v] = "ancestor" + str(count) # internal node
            count += 1

    nx.relabel_nodes(nx_tree, mapping, copy=False)
    # save and plot inferred tree
    return nx_tree

def make_gt_tree_dist(ad, n_states, cell_names: list) -> tuple[dpy.Tree, np.ndarray]:
    # TODO: this function can be rewritten to re-use the code from function (D, Dp) = utils.testing.get_expected_changes(cnps, tree_nx, cell_pairs)
    #   together with utils.testing.get_expected_distances(D, Dp, n_states, cell_pairs)
    # traverse the tree, write lengths to branches and, for each pair, sum lengths between them
    n_sites = ad.n_vars
    nxtree = newick_to_nx(ad.uns['cell-tree-newick'])
    nxtree = nx.dfs_tree(nxtree, source='founder')  # make sure it's rooted
    # get copy number (ancestors) at each node and compute the length based on changes
    ancestor_idx = {a: i for i,a in enumerate(ad.uns['ancestral-names'])}  # names as in the tree (index for ancestral-cn)
    ancestor_cn = ad.uns['ancestral-cn'] # shape (n_ancestors, n_bins)
    for u, v in nxtree.edges():
        if not v.startswith('cell'):
            i,j = ancestor_idx[u], ancestor_idx[v]
            p = math_utils.compute_cn_changes(np.vstack([ancestor_cn[i], ancestor_cn[j]]), pairs=[(0, 1)])[0] / n_sites
            target_length = math_utils.l_from_p(p, n_states)
            nxtree[u][v]['length'] = target_length
        else:
            v_cn = ad[v].layers['state']
            u_cn = ancestor_cn[ancestor_idx[u]]
            p = math_utils.compute_cn_changes(np.vstack([u_cn, v_cn]), pairs=[(0, 1)])[0] / n_sites
            target_length = math_utils.l_from_p(p, n_states)
            nxtree[u][v]['length'] = target_length
        # print(f"Edge {u}->{v} p {p}, length {nxtree[u][v]['length']}")
    # relabel nodes to integers
    nxtree = relabel_name_to_int(nxtree, cell_names)
    dpy_tree = convert_networkx_to_dendropy(nxtree, edge_length='length', internal_nodes_label='int')
    # print("DPY tree with lengths:", dpy_tree.as_string(schema='newick'))
    dist_matrix = get_ctr_table_int(dpy_tree)
    return dpy_tree, dist_matrix

def relabel_name_to_int_mapping(nxtree: nx.DiGraph, cell_names: list, ancestors_mapping=None) -> tuple[nx.DiGraph, dict]:
    """
    Give integer labels to nodes in the tree. Cell names (leaves) are labeled from 0 to n-1 in the order of cell_names
    and ancestors are labeled from n to n+m-1 where m is the number of ancestors.
    Returns the relabeled tree and the mapping from original names to integer labels.
    Parameters
    ----------
    nxtree: nx.DiGraph, input tree
    cell_names: list, list of cell names (leaves)
    ancestors_mapping: dict, mapping of ancestor names to integer labels (optional)
    Returns
    -------
    relabeled_tree: nx.DiGraph, tree with integer labels
    full_mapping: dict, mapping from original names to integer labels
    """
    cells_mapping = {name: i for i, name in enumerate(cell_names)}
    if ancestors_mapping is None:
        ancestors = set(nxtree.nodes()) - set(cell_names)
        ancestors = sorted(list(ancestors))
        n_cells = len(cell_names)
        ancestors_mapping = {n: i + n_cells for i, n in enumerate(ancestors)}
    full_mapping = {**cells_mapping, **ancestors_mapping}
    # check that all cell names are in the tree and they are leaves
    for name in cell_names:
        assert name in nxtree.nodes(), f"Cell name {name} not in tree nodes"
        assert nxtree.out_degree(name) == 0, f"Cell name {name} is not a leaf node"
    return nx.relabel_nodes(nxtree, full_mapping, copy=True), full_mapping

def relabel_name_to_int(nxtree: nx.DiGraph, cell_names: list, ancestors_mapping=None) -> nx.DiGraph:
    """
    Give integer labels to nodes in the tree. Cell names (leaves) are labeled from 0 to n-1 in the order of cell_names
    and ancestors are labeled from n to n+m-1 where m is the number of ancestors.
    """
    relabeled_tree, _ = relabel_name_to_int_mapping(nxtree, cell_names, ancestors_mapping)
    return relabeled_tree

def get_root_distance(centroid):
    root_distance = 0
    while centroid.parent_node is not None:
        root_distance += centroid.edge_length
        centroid = centroid.parent_node
    return root_distance


def get_ctr_table_int(tree: dpy.Tree, full=False) -> np.ndarray:
    """
    Get the centroid table for a given tree where leaves are labeled with integers.
    The centroid table is a 3D numpy array of shape (n_cells, n_cells, 3) where n_cells is the number of leaves in the tree.
    For each pair of cells (r, s) with r < s, the entry ctr_table[r, s] is a vector of 3 values:
        - ctr_table[r, s, 0]: distance from the centroid of r and s to the root
        - ctr_table[r, s, 1]: distance from the centroid of r and s to r
        - ctr_table[r, s, 2]: distance from the centroid of r and s to s
    The entries for r >= s are set to -1.
    The tree must be rooted and all leaves must be labeled with unique integers from 0 to n_cells - 1.
    Parameters
    ----------
    tree: dpy.Tree, the input tree with edge _lengths
    full: bool, if True, returns the full symmetric matrix, otherwise only upper triangular part is filled

    Returns
    -------
    ctr_table: np.ndarray, the centroid table
    """
    # if leaves are not integers, raise error
    if not all(leaf.label.isdigit() for leaf in tree.leaf_nodes()):
        raise ValueError("Leaves must be labeled with integers to use get_ctr_table_int, use get_ctr_table instead")
    ctr_table, _ = get_ctr_table(tree, full=full)
    return ctr_table

def get_ctr_table(tree: dpy.Tree, full: bool = False) -> tuple[np.ndarray, list]:
    """
    Get the CTR table for a given tree where leaves are labeled with integers.
    The CTR table is a 2D numpy array of shape (n_cells, n_cells) where n_cells is the number of leaves in the tree.
    For each pair of cells (r, s), the entry ctr_table[r, s] is the distance between r and s in the tree.
    The tree must be rooted and all leaves must be labeled with unique integers from 0 to n_cells - 1.
    Parameters
    ----------
    tree: dpy.Tree, the input tree with edge _lengths

    Returns
    -------
    tuple[np.ndarray, list], the triplet distance table and the list of taxa labels as ordered in the table
    """
    taxa = tree.taxon_namespace
    n_cells = len(tree.leaf_nodes())
    assert n_cells == len(tree.taxon_namespace)
    ctr_table = - np.ones((n_cells, n_cells, 3))
    for r, s in combinations(range(n_cells), 2):
        assert r < s, "r must be less than s to ensure upper triangular matrix"
        # most recent common ancestor
        r_taxa = taxa[r].label
        s_taxa = taxa[s].label
        centroid = tree.mrca(taxon_labels=[r_taxa, s_taxa])
        ctr_table[r, s, 0] = get_root_distance(centroid)
        ctr_table[r, s, 1] = get_node2node_distance(tree, centroid.label, r_taxa)
        ctr_table[r, s, 2] = get_node2node_distance(tree, centroid.label, s_taxa)
        if full:
            ctr_table[s, r, :] = ctr_table[r, s, :]
    # fill diagonal with zeros
    for i in range(n_cells):
        ctr_table[i, i, :] = 0.0
    return ctr_table, [taxa[i].label for i in range(n_cells)]

def f1_score_clades(tree: dpy.Tree, clone_assignment: list) -> float:
    """
    Metrics defined in DICE: for each clone, find the clade that maximizes the F1 score
    Return the average F1 score across all clones.
    """
    # get all clades in the tree
    clades = []
    for n in tree.postorder_node_iter():
        if not n.is_leaf():
            clade = set()
            for leaf in n.leaf_iter():
                clade.add(int(leaf.label))
            clades.append(clade)

    # group cells by clone assignment
    clone_to_cells = defaultdict(set)
    for cell, clone in enumerate(clone_assignment):
        clone_to_cells[clone].add(cell)

    f1_scores = []
    for clone, cells in clone_to_cells.items():
        best_f1 = 0
        for clade in clades:
            y_true = [1 if i in cells else 0 for i in range(len(clone_assignment))]
            y_pred = [1 if i in clade else 0 for i in range(len(clone_assignment))]
            score = f1_score(y_true, y_pred)
            if score > best_f1:
                best_f1 = score
        f1_scores.append(best_f1)

    return np.mean(f1_scores).item()

def normalized_rf_distance(tree1: dpy.Tree, tree2: dpy.Tree) -> float:
    """
    Compute the normalized Robinson-Foulds distance between two (rooted) trees using DendroPy.
    The trees must have the same set of leaf labels.
    The normalized RF distance is the RF distance divided by the maximum possible RF distance.
    """
    rf = treecompare.symmetric_difference(tree1, tree2)
    n_leaves = len(tree1.leaf_nodes())
    max_rf = 2 * (n_leaves - 2)
    return rf / max_rf


def get_lowest_common_ancestor(tree_nx, node1, node2):
    """
    Get the index of the least common ancestor of two nodes in a directed tree.
    """
    return nx.lowest_common_ancestor(tree_nx, node1, node2)


def convert_skbio_to_networkx(tree_nj_skbio: skbio.TreeNode, interior_node_names=None)-> nx.DiGraph:
    """
    Convert a skbio TreeNode to a networkx DiGraph.
    Parameters
    ----------
    tree_nj_skbio

    Returns tree_nx: nx.DiGraph
    -------

    """
    newick = str(tree_nj_skbio)
    tree_nx = newick_to_nx(newick, edge_attr='weight', interior_node_names=interior_node_names)
    return tree_nx


def get_two_leaf_internal_nodes(tree_nx):
    """
    Get internal nodes that have two leaf children.
    """
    two_leaf_internals = []
    for n in tree_nx.nodes():
        if tree_nx.out_degree(n) == 2:
            children = list(tree_nx.successors(n))
            if all(tree_nx.out_degree(c) == 0 for c in children):
                two_leaf_internals.append(n)
    return two_leaf_internals


def reconstruct_internal_cnps(leaf_cnps, tree_nx, n_states, method):
    """
    Given CNPs of the leaves and a tree, reconstruct the CNPs of the internal nodes by
    minimum evolution.
    Assumes leaf_cnps are in the order of leaf node indices.
    """
    M, N = leaf_cnps.shape

    internal_cnps = generalized_sankoff_algorithm.reconstruct_cnps_with_block_mutations(tree_nx, leaf_cnps, max_mutations_per_edge=5)
    return internal_cnps


def skbio_neighbour_joining_from_pairwise_distances(pairwise_distances: np.ndarray | dict)-> skbio.TreeNode:
    """
    Takes an upper triangular pairwise distance matrix or a dict of pairwise distances and runs skbio neighbor joining
    to reconstruct a tree.
    Returns a skbio.TreeNode
    """
    if isinstance(pairwise_distances, dict):
        n_cells = max(max(v, w) for v, w in pairwise_distances.keys()) + 1
        pairwise_distances_matrix = np.zeros((n_cells, n_cells))
        for (v, w), dist in pairwise_distances.items():
            pairwise_distances_matrix[v, w] = dist
        pairwise_distances = pairwise_distances_matrix
    n_cells = pairwise_distances.shape[0]
    pairwise_distances = np.triu(pairwise_distances) + np.triu(pairwise_distances, k=1).T
    np.fill_diagonal(pairwise_distances, 0)
    skbio_dm = skbio.DistanceMatrix(pairwise_distances, ids=[str(i) for i in range(n_cells)])
    tree_nj_skbio = skbio.tree.nj(skbio_dm)
    tree_nj_skbio = tree_nj_skbio.root_at_midpoint()
    return tree_nj_skbio


def relabel_dendropy(tree: dpy.Tree, leaves_mapping: dict):
    """
    Relabels the leaves of a DendroPy tree according to a given mapping.
    ----------
    tree: dendropy.Tree
    leaves_mapping: dict, mapping from old labels to new labels
    """
    for n in tree.leaf_node_iter():
        if n.label is None:
            n.label = leaves_mapping[n.taxon.label]
            n.taxon.label = n.label
        else:
            n.label = leaves_mapping[n.label]