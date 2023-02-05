import itertools
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import loggamma

""" UTILITY METHODS AND DATA STRUCTURES """

def write_gph(dag, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(edge[0], edge[1]))

# The following is shamelessly stolen from https://github.com/denizetkar/priority-queue. I wasn't getting the
# functionality I wanted from Lib/queue.py, which doesn't allow users to update the priority of entries.
class PriorityQueue:
    def __init__(
        self,
        arr=None,
        has_higher_priority=lambda x, y: x[0] < y[0],
        id_of=lambda x: x[1],
    ):
        if arr is None:
            arr = []
        self._heap = arr
        self.has_higher_priority = has_higher_priority
        self.id_of = id_of
        self._elem_idxs = {}
        for i, elem in enumerate(self):
            self._add_elem_idx(elem=elem, idx=i)

        self._heapify()

    def _get_elem_idxs(self, elem=None, elem_id=None):
        if elem is not None:
            elem_id = self.id_of(elem)
        return self._elem_idxs.get(elem_id, None)

    def _add_elem_idx(self, idx, elem=None, elem_id=None):
        if elem is not None:
            elem_id = self.id_of(elem)
        assert self._get_elem_idxs(elem_id=elem_id) is None, "`elem` must not exist"
        self._elem_idxs[elem_id] = idx

    def _remove_elem_idx(self, elem=None, elem_id=None):
        if elem is not None:
            elem_id = self.id_of(elem)
        self._elem_idxs.pop(elem_id, None)

    def __len__(self):
        return len(self._heap)

    def __getitem__(self, i):
        return self._heap[i]

    @staticmethod
    def left_child_idx(i):
        return 2 * i + 1

    @staticmethod
    def right_child_idx(i):
        return 2 * i + 2

    @staticmethod
    def parent_idx(i):
        return (i - 1) // 2

    def _swap_elems(self, i, j):
        elem1, elem2 = self[i], self[j]
        self._heap[i], self._heap[j] = elem2, elem1
        self._remove_elem_idx(elem=elem1)
        self._add_elem_idx(elem=elem1, idx=j)
        self._remove_elem_idx(elem=elem2)
        self._add_elem_idx(elem=elem2, idx=i)

    def _bubble_up(self, i):
        while True:
            if i == 0:
                return
            parent_idx = self.parent_idx(i)
            if not self.has_higher_priority(self[i], self[parent_idx]):
                return
            self._swap_elems(parent_idx, i)
            i = parent_idx

    def _bubble_down(self, i):
        while True:
            left_child_idx = self.left_child_idx(i)
            right_child_idx = self.right_child_idx(i)
            if left_child_idx >= len(self):
                return
            prio_idx = left_child_idx
            if right_child_idx < len(self):
                if self.has_higher_priority(self[right_child_idx], self[prio_idx]):
                    prio_idx = right_child_idx

            if not self.has_higher_priority(self[prio_idx], self[i]):
                return

            self._swap_elems(i, prio_idx)
            i = prio_idx

    def _heapify(self):
        if len(self) == 0:
            return
        max_idx = self.parent_idx(len(self) - 1)
        for idx in range(max_idx, -1, -1):
            self._bubble_down(idx)

    def _append(self, elem):
        self._add_elem_idx(elem=elem, idx=len(self))
        self._heap.append(elem)

    def put(self, elem):
        self._append(elem)
        self._bubble_up(len(self) - 1)

    def _remove_last(self):
        last_idx = len(self) - 1
        elem = self._heap.pop(last_idx)
        self._remove_elem_idx(elem=elem)
        return elem

    def pop(self):
        if len(self) == 0:
            return None
        self._swap_elems(0, len(self) - 1)
        elem = self._remove_last()
        self._bubble_down(0)
        return elem

    def update_elem(self, elem_id, new_elem):
        idx = self._get_elem_idxs(elem_id=elem_id)
        if idx is None:
            return

        elem = self._heap[idx]
        self._heap[idx] = new_elem
        self._remove_elem_idx(elem_id=elem_id)
        self._add_elem_idx(elem=new_elem, idx=idx)
        if self.has_higher_priority(new_elem, elem):
            self._bubble_up(idx)
        elif self.has_higher_priority(elem, new_elem):
            self._bubble_down(idx)

    def remove(self, elem_id):
        idx = self._get_elem_idxs(elem_id=elem_id)
        if idx is None:
            return None

        last_idx = len(self) - 1
        last_elem = self[last_idx]
        self._swap_elems(idx, last_idx)
        elem = self._remove_last()
        if self.has_higher_priority(last_elem, elem):
            self._bubble_up(idx)
        elif self.has_higher_priority(elem, last_elem):
            self._bubble_down(idx)

        return elem

    def __repr__(self) -> str:
        return repr(self._heap)

def feature_values(df: pd.DataFrame, constant_support: list=None) -> dict:
    """
    Given a DataFrame, computes the support for each feature, i.e., the set of all possible values that each feature
    can take. Returns a dictionary whose keys are the feature names, and whose values are lists containing the supports.

    :param df: A DataFrame containing the data.
    :param constant_support: (Optional) A list that will be used for the support of all features. The default is None,
                             which means the supports will be inferred from the data.
    :return: A dict whose keys are the feature names and whose values are lists containing the supports.
    """
    values_dict = {}
    for col in df.columns:
        if constant_support is None:
            # I extract the index of the value_counts DataFrame, which gives me the unique values that appear in the
            # DataFrame for the selected feature.
            values_dict[col] = df[col].value_counts(sort=False).index.to_list()
        else:
            values_dict[col] = deepcopy(constant_support)
    return values_dict

def top_order_dict_from_array(top_order_list: list) -> dict:
    """
    Given a list containing features names in topological order, constructs a dict that encodes this order in a manner
    compatible with the optimal_graph method.

    :param top_order_array: A list containing the feature names in topological order.
    :return: A dict that maps a feature name to a frozenset containing the name of the feature that appears next in the
             user-specified topological order. The first feature is the value of the key "root", and the value of the
             last feature is an empty frozenset.
    """
    top_order_dict = {}
    top_order_dict["root"] = frozenset([top_order_list[0]])
    for predecessor, successor in zip(top_order_list[:-1], top_order_list[1:]):
        top_order_dict[predecessor] = frozenset([successor])
    top_order_dict[top_order_list[-1]] = frozenset()
    return top_order_dict

""" LEGOS AND LEGO SCORES """

@dataclass(frozen=True)
class Lego:
    parents: frozenset
    child: str

def lego_score(df: pd.DataFrame, num_vals: int, lego: Lego, saved_scores: dict=None) -> float:
    """
    Given a DataFrame, the number of values the child feature can take, and a Lego, this method computes the
    Lego's Score, assuming a uniform prior. If saved_scores is passed, then the method will first check to see
    if the Lego has already been scored, and otherwise computes and stores it.

    :param df: The DataFrame containing the data.
    :param num_vals: The cardinality of the support of the feature, i.e., the number of possible values the feature
                     can take. This is the value used for alpha_{ij0}.
    :param lego: The Lego being scored.
    :param saved_scores: (Optional) A dict containing the scores of Legos that have already been scored. The default
                         is None.
    :return: The score of the Lego.
    """
    # If saved_scores is passed, let's check to see if the Lego has already been scored.
    if not (saved_scores is None):
        if lego in saved_scores:
            return saved_scores[lego]

    # Pandas needs a list of columns, i.e., it does not support sets. Let's convert the set of parents to a list.
    parents_list = list(lego.parents)

    # This gives us all combinations of (parents, child) instantiations present in the data, as well as their counts.
    # In other words, this gives us the m_{ijk}.
    joint_counts = df[
        parents_list + [lego.child]
    ].value_counts(sort=False)

    # If the child has no parents, then m_{ij0} is just the number of data points.
    if len(lego.parents) == 0:
        total = loggamma(num_vals)
        total -= loggamma(num_vals + len(df))
        total += loggamma(1 + joint_counts).sum()

    # Otherwise, we need to compute the possible instantiations for the parents.
    else:
        parent_counts = df[parents_list].value_counts(sort=False)

        total = len(parent_counts) * loggamma(num_vals)
        total -= loggamma(num_vals + parent_counts).sum()
        total += loggamma(1 + joint_counts).sum()

    # If saved_scores was passed, we should add this Lego and its score to save on future computation.
    if not (saved_scores is None):
        saved_scores[lego] = total

    return total

""" PARENT SELECTION """

@dataclass(frozen=True)
class OutwardEdge:
    recipient_lego: Lego
    edge_label: str

@dataclass(frozen=True)
class ParentSearchNode:
    lego: Lego
    outward_edges: frozenset
    remaining_predecessors: frozenset

def parent_selection_dijkstra(
        df: pd.DataFrame,
        num_vals: int,
        predecessors: set,
        child: str,
        max_num_parents: int=None,
        saved_scores: dict=None,
        max_num_check: int=10
) -> Lego:
    """
    Performs Dijkstra's algorithm on a decision graph which considers what features to include next in the parent set,
    while keeping track of the remaining predecessor features that have not been considered.

    :param df: The DataFrame containing the data.
    :param num_vals: The number of values that the child feature can take.
    :param predecessors: A set of features that precede the child in the network's topological order. Parents will
                         be selected from this set.
    :param child: A child feature, whose parents we are trying to select.
    :param max_num_parents: (Optional) The maximum number of parents permitted. The default is None, indicating that
                            we place no cap on the number of parents for child.
    :param saved_scores: (Optional) A dict containing saved Lego scores. The default is None.
    :param max_num_check: (Optional) The maximum number of vertices we explore. The default is 10, since experimentation
                          suggests that the optimal parents are found within 10 exploration steps, or otherwise the
                          algorithm takes impractically long to run.
    :return: A Lego containing the (approximately) optimal parent set, as well as the input child.
    """
    # In order to hash ParentSearchNode objects, we need to use frozensets.
    predecessors = frozenset(predecessors)

    # Set default parameters for None inputs.
    if max_num_parents is None:
        max_num_parents = len(df)
    if saved_scores is None:
        saved_scores = {}

    # Define the root of the decision tree, which simply assigns an empty parent set to child.
    root_lego = Lego(frozenset(), child)
    root_outward_edges = []

    # We consider each predecessor in turn and create new vertices in the graph that either include or exclude that
    # predecessor, while also noting that we have already considered the predecessor (so deeper vertices along these
    # branches should not consider the predecessor again). The root points to each of these vertices.
    for predecessor in predecessors:
        root_outward_edges.append(OutwardEdge(
            recipient_lego=Lego(frozenset({predecessor}), child),
            edge_label=predecessor
        ))
        root_outward_edges.append(OutwardEdge(
            recipient_lego=Lego(frozenset(), child),
            edge_label=predecessor
        ))

    # Instantiate the root vertex
    root = ParentSearchNode(
        lego=root_lego,
        outward_edges=frozenset(root_outward_edges),
        remaining_predecessors=predecessors
    )

    # The "open set" in Dijkstra's algorithm refers to the priority queue of vertices that have yet to be explored.
    open_set = PriorityQueue()

    # We initialize the root with a distance of 0 from the root.
    open_set.put((0., root))

    # We initialize a dict that will map a vertex in the graph to its best-known distance from the root.
    distance_from_root = {}
    distance_from_root[root] = 0

    # We initialize a counter that counts how many vertices have been explored. If this ever exceeds max_num_check,
    # then we'll terminate early and return the vertex corresponding to the best Lego that we've explore thus far.
    i = 0
    best = root

    # Now we enter into the main logic for Dijkstra's algorithm
    while len(open_set) > 0 and i <= max_num_check:

        # Take the highest-priority vertex (note that the custom PriorityQueue object stores data as tuples of the
        # form (priority, data), and we just care about the data).
        _, current = open_set.pop()

        # Check if we've reached a goal vertex. If so, then we can happily return the current Lego.
        if (
            len(current.outward_edges) == 0 or
            len(current.remaining_predecessors) == len(predecessors) - max_num_parents
        ):
            return current.lego

        # Although we've not reached the goal vertex, we should check the current Lego's score to see if it beats the
        # current best-known.
        if -lego_score(df, num_vals, best.lego, saved_scores) > -lego_score(df, num_vals, current.lego, saved_scores):
            best = current

        # Now let's populate the graph with the current vertex's neighbors and the edge weights.
        for outward_edge in current.outward_edges:
            # The Lego that this edge points to is directly extractable.
            neighbor_lego = outward_edge.recipient_lego

            # If we go to this neighbor, then we'll have considered all of the predecessors that current has considered,
            # and we'll also consider including/excluding the predecessor that labels the edge pointing to neighbor.
            neighbor_remaining_predecessors = current.remaining_predecessors.difference({outward_edge.edge_label})

            # The neighbor's neighbors can be specified by just providing the neighbor's neighbors' Legos and the
            # predecessor that was considered to arrive there. We'll initialize an empty list, but populate it
            # iteratively.
            neighbor_outward_edges = []
            for predecessor in neighbor_remaining_predecessors:
                # We don't care to populate the graph with vertices whose parent set exceeds max_num_parents
                if len(neighbor_lego.parents) < max_num_parents:
                    recipient_lego_include = Lego(neighbor_lego.parents.union({predecessor}), child)
                    neighbor_outward_edges.append(OutwardEdge(
                        recipient_lego=recipient_lego_include,
                        edge_label=predecessor
                    ))

                # If the predecessor is not included, then we're sure not to exceed the max_num_parents, since we know
                # the parent set stored in current doesn't exceed max_num_parents (and hence the parent set
                # stored in neighbor has size at most max_num_parents).
                recipient_lego_exclude = Lego(neighbor_lego.parents, child)
                neighbor_outward_edges.append(OutwardEdge(
                    recipient_lego=recipient_lego_exclude,
                    edge_label=predecessor
                ))

            # We've finished computing the data needed for specifying the neighbor, so let's instantiate the vertex
            # and add it to the graph.
            neighbor_node = ParentSearchNode(
                lego=neighbor_lego,
                outward_edges=frozenset(neighbor_outward_edges),
                remaining_predecessors=neighbor_remaining_predecessors
            )

            # Let's find the edge weights from current to neighbor. Remember it's the difference of the lego scores.
            neighbor_score = -lego_score(df, num_vals, neighbor_lego, saved_scores)
            edge_weight = neighbor_score + lego_score(df, num_vals, current.lego, saved_scores)

            # How far is neighbor from root? Well, if we're getting to neighbor from current, it ought to be
            # the distance of current from root, plus the edge weight. This isn't necessarily the shortest way to
            # get to neighbor, though, so let's define an auxiliary variable and compare with what we know.
            new_distance_to_neighbor_from_root = distance_from_root[current] + edge_weight

            # If neighbor already belongs to the open set, then we've already encountered it. So we should check to see
            # if this is a faster route to neighbor, and update the priority accordingly.
            if not (open_set._elem_idxs.get(neighbor_node, None) is None):
                distance_from_root[neighbor_node] = min(
                    new_distance_to_neighbor_from_root, distance_from_root[neighbor_node]
                )
                open_set.update_elem(neighbor_node, (distance_from_root[neighbor_node], neighbor_node))

            # Otherwise, we haven't yet seen neighbor, so surely the distance we computed above is the best.
            else:
                distance_from_root[neighbor_node] = new_distance_to_neighbor_from_root
                open_set.put((distance_from_root[neighbor_node], neighbor_node))

        # Iterate the counter, as we have finished exploring this vertex
        i += 1

    # We've exceeded max_num_check, so let's just return the best Lego we've seen.
    return best.lego

def parent_selection_fixed_point(
        df: pd.DataFrame,
        num_vals: int,
        predecessors: set,
        child: str,
        rng: np.random.Generator,
        start_num_parents: int=7,
        saved_scores: dict=None,
        num_samples: int=10
) -> Lego:
    """
    A heuristic fixed-point method for optimal parent selection.

    :param df: The DataFrame containing the data.
    :param num_vals: The number of values that the child feature can take.
    :param predecessors: A set of features that precede the child in the network's topological order. Parents will
                         be selected from this set.
    :param child: A child feature, whose parents we are trying to select.
    :param rng: A numpy Generator object used for the initialization step.
    :param start_num_parents: (Optional) The (maximum) number of parents that will be selected for each parent set in
                              the random population. The default is 7.
    :param saved_scores: (Optional) A dict containing saved Lego scores. The default is None.
    :param num_samples: (Optional) The size of the population. The default is 10.
    :return: A Lego containing the (approximately) optimal parent set, as well as the input child.
    """
    # For efficiency's sake, let's initialize an empty dict to save scores if the user hasn't passed one in.
    if saved_scores is None:
        saved_scores = {}

    # Let's initialize variables to track the best-known lego and its score.
    best_score, best_lego = -np.inf, None

    # We'll produce num_samples random proposals for the parent set of child, and we'll iteratively refine them until
    # a fixed point is reached.
    for _ in range(num_samples):
        # The random parent set can have at most as many parents as there are predecessors, and also at most as many
        # parents as the user-specified start_num_parents. Note that we store the parents in a frozenset so that it
        # is hashable.
        parents = frozenset(
            rng.choice(list(predecessors), size=min(len(predecessors), start_num_parents), replace=False)
        )

        # We'll initialize a set of "previously seen" parent set proposals. Although the method is called a fixed-point
        # method, I didn't bother to prove that this approach always results in a fixed point. It could result in a
        # cycle for all I know. This is to safeguard against that possibility. (Of course, our search space is finite,
        # so we are certain to eventually retrace our steps.)
        previously_seen = set()
        while parents not in previously_seen:
            # Let's add the current proposal for parents to the previously_seen set.
            previously_seen |= {parents}

            # We'll recall how the current Lego performs, so that we can compare it against the newly proposed Legos.
            current_lego = Lego(parents, child)
            current_score = lego_score(df, num_vals, current_lego, saved_scores)

            # We'll copy the current parents, and begin the iteration process.
            new_parents = set(parents)
            for predecessor in predecessors:
                # If the predecessor we're currently considering belongs to the new_parents, then let's test
                # what happens if we remove it. If it's not in there, let's test what happens if we add it.
                proposal = new_parents.symmetric_difference({predecessor})
                proposal_lego = Lego(frozenset(proposal), child)
                proposal_score = lego_score(df, num_vals, proposal_lego, saved_scores)

                # Does our proposal outperform the current score? If so, let's update our current score.
                if proposal_score > current_score:
                    new_parents = proposal
                    current_score = proposal_score

            # Have we found anything better globally? If so, let's update the global trackers.
            if current_score > best_score:
                best_score, best_lego = current_score, Lego(frozenset(new_parents), child)

            # We're ready to iterate over predecessors again.
            parents = frozenset(new_parents)

    return best_lego

def parent_selection_combined(
        df: pd.DataFrame,
        num_vals: int,
        predecessors: set,
        child: str,
        rng: np.random.Generator,
        max_num_parents: int=7,
        max_num_check: int=10,
        start_num_parents: int=7,
        saved_scores: dict=None,
        num_samples: int=10
) -> Lego:
    """
    A method that combines Dijkstra's method and the fixed-point method for optimal parent selection.

    :param df: The DataFrame containing the data.
    :param num_vals: The number of values that the child feature can take.
    :param predecessors: A set of features that precede the child in the network's topological order. Parents will
                         be selected from this set.
    :param child: A child feature, whose parents we are trying to select.
    :param rng: A numpy Generator object used for the initialization step.
    :param max_num_parents: (Optional) The maximum number of parents permitted. The default is None, indicating that
                            we place no cap on the number of parents for child.
    :param max_num_check: (Optional) The maximum number of vertices we explore. The default is 10, since experimentation
                          suggests that the optimal parents are found within 10 exploration steps, or otherwise the
                          algorithm takes impractically long to run.
    :param start_num_parents: (Optional) The (maximum) number of parents that will be selected for each parent set in
                              the random population for the fixed-point method. The default is 7.
    :param saved_scores: (Optional) A dict containing saved Lego scores. The default is None.
    :param num_samples: (Optional) The size of the population in the fixed-point method. The default is 10. Note that
                        if 0 is passed, then the fixed-point method will be skipped.
    :return: A Lego containing the (approximately) optimal parent set, as well as the input child.
    """
    # For efficiency's sake, let's initialize an empty saved_scores dict in case the user hasn't provided one.
    if saved_scores is None:
        saved_scores = {}

    # Try running Dijkstra's algorithm.
    optimal_dijkstra = parent_selection_dijkstra(
        df=df,
        num_vals=num_vals,
        predecessors=predecessors,
        child=child,
        max_num_parents=max_num_parents,
        saved_scores=saved_scores,
        max_num_check=max_num_check
    )
    optimal_dijkstra_score = lego_score(df, num_vals, optimal_dijkstra, saved_scores)

    # If the user has provided num_samples = 0, then we don't need to run the fixed-point method.
    if num_samples == 0:
        return optimal_dijkstra

    # Let's try the fixed-point method.
    optimal_fixed_point = parent_selection_fixed_point(
        df=df,
        num_vals=num_vals,
        predecessors=predecessors,
        child=child,
        rng=rng,
        start_num_parents=start_num_parents,
        saved_scores=saved_scores,
        num_samples=num_samples
    )
    optimal_fixed_point_score = lego_score(df, num_vals, optimal_fixed_point, saved_scores)

    # Return the Lego that produces the better score.
    if optimal_dijkstra_score > optimal_fixed_point_score:
        return optimal_dijkstra

    return optimal_fixed_point

def construct_random_legos_with_optimal_parents(
        df: pd.DataFrame,
        feature_values_dict: dict,
        saved_scores: dict,
        rng: np.random.Generator,
        num_legos: int=1000,
        max_num_parents: int=7,
        max_num_check: int=10,
        start_num_parents: int=7,
        num_samples: int=10
) -> None:
    """
    Constructs a bunch of randomly generated Legos and stores their scores in saved_scores.

    :param df: The DataFrame containing the data.
    :param feature_values_dict: A dict mapping feature names to the values the feature can take.
    :param saved_scores: A dict containing the scores of Legos that have already been scored.
    :param rng: A numpy random number generator.
    :param num_legos: (Optional) The number of random Legos to generate (the default is 1000).
    :param max_num_check: (Optional) The maximum number of vertices we explore. The default is 10, since experimentation
                          suggests that the optimal parents are found within 10 exploration steps, or otherwise the
                          algorithm takes impractically long to run.
    :param start_num_parents: (Optional) The (maximum) number of parents that will be selected for each parent set in
                              the random population for the fixed-point method. The default is 7.
    :param num_samples: (Optional) The size of the population in the fixed-point method. The default is 10. Note that
                        if 0 is passed, then the fixed-point method will be skipped.
    :return: None.
    """
    # Extract the feature names
    features = list(df.columns)

    for _ in range(num_legos):
        # Pick a random number of features to use (we'll need at least one so that there is a child)
        num_features = rng.choice(np.arange(1, len(features) + 1))

        # Pick the features randomly
        feature_selection = rng.choice(features, size=num_features, replace=False)

        # (Arbitrarily) set the child to be the last entry, and the predecessors everything else. Note that the random
        # generation will yield a random permutation also, so this loses no generality.
        predecessors, child = frozenset(feature_selection[:-1]), feature_selection[-1]
        random_lego = parent_selection_combined(
            df=df,
            num_vals=len(feature_values_dict[child]),
            predecessors=predecessors,
            child=child,
            rng=rng,
            max_num_parents=max_num_parents,
            max_num_check=max_num_check,
            start_num_parents=start_num_parents,
            saved_scores=saved_scores,
            num_samples=num_samples
        )

        # Score the random lego
        lego_score(df, len(feature_values_dict[child]), random_lego, saved_scores)

""" TOPOLOGICAL ORDERING """

def parent_selection_all_predecessors(
        df: pd.DataFrame,
        feature_values_dict: dict,
        rng: np.random.Generator,
        max_num_parents: int=7,
        max_num_check: int=10,
        start_num_parents: int=7,
        saved_scores: int=None,
        num_samples: int=10
) -> dict:
    """
    For each feature in the dataset, computes the optimal parents given all other features as predecessors. This is
    useful for computing the heuristic for A*.

    :param df: The DataFrame containing the data.
    :param feature_values_dict: A dict which maps feature names to its support, i.e., an array of its possible values.
    :param rng: A numpy Generator object used for the initialization step.
    :param max_num_parents: (Optional) The maximum number of parents permitted. The default is None, indicating that
                            we place no cap on the number of parents for child.
    :param max_num_check: (Optional) The maximum number of vertices we explore. The default is 10, since experimentation
                          suggests that the optimal parents are found within 10 exploration steps, or otherwise the
                          algorithm takes impractically long to run.
    :param start_num_parents: (Optional) The (maximum) number of parents that will be selected for each parent set in
                              the random population for the fixed-point method. The default is 7.
    :param saved_scores: (Optional) A dict containing saved Lego scores. The default is None.
    :param num_samples: (Optional) The size of the population in the fixed-point method. The default is 10. Note that
                        if 0 is passed, then the fixed-point method will be skipped.
    :return: A dict which maps feature names to a Lego containing the feature as a child, and the (approximately)
             optimal parent set out of the remaining features.
    """
    # For efficiency's sake, let's define an empty saved_scores dict if the user has not provided one.
    if saved_scores is None:
        saved_scores = {}

    # We need the features for the data.
    feature_names = set(df.columns)

    # We're quite literally finding the optimal parents for any given feature. Of course, due to the acyclicity
    # constraint, we likely won't achieve this in our network. But it gives a heuristic function in A*.
    optimal_parents_dict = {}
    for child in feature_names:
        # Extract all features except for child.
        predecessors = feature_names.difference({child})

        # Compute the (approximately) optimal parents.
        optimal = parent_selection_combined(
            df=df,
            num_vals=len(feature_values_dict[child]),
            predecessors=predecessors,
            child=child,
            rng=rng,
            max_num_parents=max_num_parents,
            max_num_check=max_num_check,
            start_num_parents=start_num_parents,
            saved_scores=saved_scores,
            num_samples=num_samples
        )

        # Associate the optimal parents with this child.
        optimal_parents_dict[child] = optimal

    return optimal_parents_dict

@dataclass(frozen=True)
class SearchGraphNode:
    predecessors: frozenset
    lego: Lego
    successors: frozenset

def construct_network(came_from: dict, current: SearchGraphNode, g_score: float) -> Tuple[nx.DiGraph, float]:
    """
    Constructs the Bayesian network given an (approximately) optimal path through the search graph.

    :param came_from: A dict which maps SearchGraphNode objects to SearchGraphNode objects.
    :param current: The terminal SearchGraphNode object in the path.
    :param g_score: The final Bayesian score of the network.
    :return: A tuple (network, network_score) containing the Bayesian network and its Bayesian score.
    """
    # Initialize the graph and populate its with vertices.
    graph = nx.DiGraph()
    for feature in current.predecessors:
        graph.add_node(feature)
    graph.add_node(current.lego.child)

    # Now traverse backwards through our path in the search graph, adding edges from parents to children.
    for parent in current.lego.parents:
        graph.add_edge(parent, current.lego.child)
    while current in came_from:
        current = came_from[current]
        for parent in current.lego.parents:
            graph.add_edge(parent, current.lego.child)

    return graph, g_score

def optimal_network(
        df: pd.DataFrame,
        feature_values_dict: dict,
        rng: np.random.Generator,
        max_num_parents: int=7,
        max_num_check: int=10,
        start_num_parents: int=7,
        saved_scores: dict=None,
        num_samples: int=10,
        optimal_parents_dict: dict=None,
        use_only_saved: bool=False,
        max_allowed_score: float=np.inf,
        top_order=None
) -> Tuple[nx.DiGraph, float]:
    """
    Performs A* on the search graph of Legos in order to find an (approximately) optimal Bayesian network for the
    provided data.

    :param df: The DataFrame containing the data.
    :param feature_values_dict: A dict which maps feature names to its support, i.e., an array of its possible values.
    :param rng: A numpy Generator object used for the initialization step.
    :param max_num_parents: (Optional) The maximum number of parents permitted. The default is None, indicating that
                            we place no cap on the number of parents for child.
    :param max_num_check: (Optional) The maximum number of vertices we explore. The default is 10, since experimentation
                          suggests that the optimal parents are found within 10 exploration steps, or otherwise the
                          algorithm takes impractically long to run.
    :param start_num_parents: (Optional) The (maximum) number of parents that will be selected for each parent set in
                              the random population for the fixed-point method. The default is 7.
    :param saved_scores: (Optional) A dict containing saved Lego scores. The default is None.
    :param num_samples: (Optional) The size of the population in the fixed-point method. The default is 10. Note that
                        if 0 is passed, then the fixed-point method will be skipped.
    :param optimal_parents_dict: (Optional) A dict which maps feature names to a Lego containing the feature as a
                                 child, and the (approximately) optimal parent set out of the remaining features. The
                                 default is None.
    :param use_only_saved: (Optional) A bool that determines whether only those vertices corresponding to Legos in
                           saved_scores are explored during A*. Note that this can be True even if the input
                           saved_scores is None. This is because, in that case, an empty saved_scores dict is
                           initialized and used during the generation of heuristic values. The default is False.
    :param max_allowed_score: (Optional) This is an upper bound on the minimum (negative) Bayesian score, used for
                              cutting branches that are guaranteed to exceed the upper bound. The default is np.inf.
    :param top_order: (Optional) A dict that maps a feature name to a frozenset containing the name of the feature that
                      appears next in the user-specified topological order. The first feature is the value of the key
                      "root", and the value of the last feature is an empty frozenset. The default is None, indicating
                      that the topological order should be discovered during the algorithm's runtime.
    :return: A tuple (network, network_score) containing the (approximately optimal) Bayesian network and its Bayesian
             score.
    """
    # We need to extract the features from the data.
    features = frozenset(df.columns)

    # If the user has not specified a topological order, then starting from the root, we have liberty to pick any
    # successor we wish.
    if top_order is None:
        root = SearchGraphNode(
            predecessors=frozenset(),
            lego=Lego(frozenset(), ""),
            successors=features
        )

    # Otherwise, there can be only one successor from the root.
    else:
        root = SearchGraphNode(
            predecessors=frozenset(),
            lego=Lego(frozenset(), ""),
            successors=top_order["root"]
        )

    # Like Dijkstra, the "open set" of A* refers to the PriorityQueue of vertices to be explored.
    open_set = PriorityQueue()
    open_set.put((0., root))

    # We aren't storing enough information in each vertex to construct the optimal network from just the terminal
    # vertex in our path through the search graph. So we need to keep track of the path itself, which will then let us
    # construct the network.
    came_from = {}

    # Let's initialize a dict to keep track of each vertex's best-known distance from the root.
    distance_from_root = {}
    distance_from_root[root] = 0

    # Let's also initialize a dict to keep track of our heuristic estimate of the weight of the best path which contains
    # the vertex we are currently considering. Note that we initialize root to have a heuristic weight of 0 because we
    # will never revisit root, so it doesn't matter what we fill here.
    heuristic_total_distance = {}
    heuristic_total_distance[root] = 0

    # For efficiency's sake, if the user has not specified a saved_scores dict, let's initialize one here.
    if saved_scores is None:
        saved_scores = {}

    # If the user has not provided a precomputed optimal parents dict that maps features to their optimal parent sets
    # (picking parents from all other features) then we compute that here.
    if optimal_parents_dict is None:
        optimal_parents_dict = parent_selection_all_predecessors(
            df=df,
            feature_values_dict=feature_values_dict,
            rng=rng,
            max_num_parents=max_num_parents,
            max_num_check=max_num_check,
            start_num_parents=start_num_parents,
            saved_scores=saved_scores,
            num_samples=num_samples
        )

    # We now compute the heuristic values associated with each feature.
    heuristic_values = {}
    for feature in features:
        heuristic_values[feature] = -lego_score(
            df, len(feature_values_dict[feature]), optimal_parents_dict[feature], saved_scores=saved_scores
        )

    # If the user has indicated to use only those Legos whose scores have already been computed, then we define a dict
    # that maps children to the Legos in saved_scores which have that child. This is done for efficiency so that we
    # don't need to loop over saved_scores with every iteration during A*.
    if use_only_saved:
        child_to_saved_scores = {}
        for feature in features:
            child_to_saved_scores[feature] = {}
        for lego, score in saved_scores.items():
            child_to_saved_scores[lego.child][lego] = score

    # We now begin the main logic for A*.
    while len(open_set) > 0:
        # We take the highest-priority vertex from the queue.
        _, current = open_set.pop()

        # If the current vertex we're considering has no successors, then we've added all features to the network. This
        # means that we are done searching, and we can simply construct and return the optimal network!
        if len(current.successors) == 0:
            return construct_network(came_from, current, distance_from_root[current])

        # This is a technical check, since the root has no child. When constructing a neighboring vertex to current,
        # we need to know the neighbor's predecessors, but any neighbor of root should have no predecessors.
        if current.lego.child == "":
            proposed_predecessors = current.predecessors
        else:
            proposed_predecessors = current.predecessors.union({current.lego.child})

        # Each successor of the current node can be added to the network next, giving a neighbor vertex in the search
        # graph.
        for proposed_child in current.successors:
            # Find the number of values the child takes in the data set.
            num_vals = len(feature_values_dict[proposed_child])

            # If the user has not passed a topological order, then all of the remaining successors (besides the
            # proposed child) can be immediate successors of the neighbor.
            if top_order is None:
                proposed_successors = current.successors.difference({proposed_child})

            # Otherwise, there is only one option for the immediate successor to the neighbor.
            else:
                proposed_successors = top_order[proposed_child]

            # If the user has specified to use only saved Lego scores, then we pick the Lego with the proposed child
            # and parents from its predecessors that is optimal, out of the Legos that we have already computed (or
            # it'll have no parents).
            if use_only_saved:
                # At the very least, we need to initialize a Lego with no parents. (Sorry if you didn't want this to
                # be computed---you might run into neighborless vertices in the search graph otherwise!)
                proposed_lego = Lego(frozenset(), proposed_child)

                # Compute the baseline score of this Lego and store it (again---sorry if you didn't want to compute
                # this).
                edge_weight = -lego_score(df, num_vals, proposed_lego, saved_scores=saved_scores)
                child_to_saved_scores[proposed_child][proposed_lego] = -edge_weight

                # Now let's iterate over all the Legos in saved_scores that are relevant, and we'll check if any of
                # them have a better score than the proposed. (Note: this can be optimized further, if we store the
                # Legos in sorted order... I didn't think to do that during my initial implementation, but I'm too
                # lazy to add this feature now.)
                for lego, score in child_to_saved_scores[proposed_child].items():
                    # A Lego is relevant if its child is proposed_child, and its parents are all in
                    # proposed_predecessors.
                    if proposed_child == lego.child and len(lego.parents.difference(proposed_predecessors)) == 0:
                        if edge_weight > -score:
                            proposed_lego, edge_weight = lego, -score

            # Otherwise, we will just pick the Lego that has optimal parents for the proposed_child out of its
            # predecessors. This is the slower option.
            else:
                proposed_lego = parent_selection_combined(
                    df=df,
                    num_vals=num_vals,
                    predecessors=proposed_predecessors,
                    child=proposed_child,
                    rng=rng,
                    max_num_parents=max_num_parents,
                    max_num_check=max_num_check,
                    start_num_parents=start_num_parents,
                    saved_scores=saved_scores,
                    num_samples=num_samples
                )
                edge_weight = -lego_score(df, num_vals, proposed_lego, saved_scores=saved_scores)

            # We've computed all data needed for this neighbor of current. Let's populate the vertex.
            proposed_graph_node = SearchGraphNode(
                predecessors=proposed_predecessors,
                lego=proposed_lego,
                successors=proposed_successors
            )

            # How far is this neighbor from root? Well, if we're getting to neighbor from current, it ought to be
            # the distance of current from root, plus the edge weight. This isn't necessarily the shortest way to
            # get to neighbor, though, so let's define an auxiliary variable and compare with what we know.
            new_distance_from_root = distance_from_root[current] + edge_weight

            # If this is a shorter way to reach this neighbor than we've previously seen (or if this is the first
            # time we're reaching this vertex), then we should store its data in the appropriate places.
            if new_distance_from_root < distance_from_root.get(proposed_graph_node, np.inf):
                # Namely, we want to store current as its immediate predecessor in the search graph, as now the best
                # way to reach the proposed neighbor is via current.
                came_from[proposed_graph_node] = current

                # We also want to update the best-known distance from root.
                distance_from_root[proposed_graph_node] = new_distance_from_root

            # Now let's compute the heuristic value for the remaining distance to the end of the search graph.
            heuristic_value = 0
            for child in proposed_graph_node.successors:
                heuristic_value += heuristic_values[child]

            # If this vertex is already in the open set, then let's see if we need to update its priority.
            if not (open_set._elem_idxs.get(proposed_graph_node, None) is None):
                # The heuristic total distance through this vertex is either the best-known heurstic total distance
                # or it's the new heuristic total distance that we just computed.
                heuristic_total_distance[proposed_graph_node] = min(
                    distance_from_root[proposed_graph_node] + heuristic_value,
                    heuristic_total_distance[proposed_graph_node]
                )

                # Update the vertex's priority in the open set.
                open_set.update_elem(
                    proposed_graph_node, (heuristic_total_distance[proposed_graph_node], proposed_graph_node)
                )
            else:
                # We've never seen this vertex, so we should compute and store the heuristic total distance.
                heuristic_total_distance[proposed_graph_node] = (
                    distance_from_root[proposed_graph_node] + heuristic_value
                )

                # If the heuristic total distance exceeds the maximum allowed score, then we'll just not explore it
                # further since we're guaranteed not to match the maximum allowed score.
                if heuristic_total_distance[proposed_graph_node] <= max_allowed_score:
                    open_set.put((heuristic_total_distance[proposed_graph_node], proposed_graph_node))

    # If we've gotten here, then we've screwed something up (maybe the maximum allowed score was too low).
    return nx.DiGraph(), np.inf

def post_processing(
        network: nx.DiGraph,
        network_score: float,
        df: pd.DataFrame,
        feature_values_dict: dict,
        rng: np.random.Generator,
        saved_scores: dict,
        optimal_parents_dict: dict,
        max_num_parents: int=7,
        max_num_check: int=10,
        start_num_parents: int=7,
        num_samples: int=10,
        use_only_saved: bool=False,
        max_num_swap: int=2
) -> Tuple[nx.DiGraph, float]:
    """
    Refines an approximately optimal network, hopefully finding the local optimum.

    :param network: A preliminary Bayesian network to refine.
    :param network_score: The score of the Bayesian network.
    :param df: The DataFrame containing the data.
    :param feature_values_dict: A dict which maps feature names to its support, i.e., an array of its possible values.
    :param rng: A numpy Generator object used for the initialization step.
    :param saved_scores: A dict containing saved Lego scores. The default is None.
    :param optimal_parents_dict: A dict which maps feature names to a Lego containing the feature as a
                                 child, and the (approximately) optimal parent set out of the remaining features.
    :param max_num_parents: (Optional) The maximum number of parents permitted. The default is None, indicating that
                            we place no cap on the number of parents for child.
    :param max_num_check: (Optional) The maximum number of vertices we explore. The default is 10, since experimentation
                          suggests that the optimal parents are found within 10 exploration steps, or otherwise the
                          algorithm takes impractically long to run.
    :param start_num_parents: (Optional) The (maximum) number of parents that will be selected for each parent set in
                              the random population for the fixed-point method. The default is 7.
    :param num_samples: (Optional) The size of the population in the fixed-point method. The default is 10. Note that
                        if 0 is passed, then the fixed-point method will be skipped.
    :param use_only_saved: (Optional) A bool that determines whether only those vertices corresponding to Legos in
                           saved_scores are explored during A*. Note that this can be True even if the input
                           saved_scores is None. This is because, in that case, an empty saved_scores dict is
                           initialized and used during the generation of heuristic values. The default is False.
    :param max_num_swap: (Optional) The maximum number of vertices to swap in topological order during refinement. The
                         default is 2.
    :return: A tuple (network, network_score) containing the (approximately optimal) Bayesian network and its Bayesian
             score.
    """
    # Extract the topological order of the passed network.
    top_order = top_order_dict_from_array(list(nx.topological_sort(network)))

    # Rescore the current network while fixing the topological order.
    best_network, best_score = optimal_network(
        df=df,
        feature_values_dict=feature_values_dict,
        rng=rng,
        max_num_parents=max_num_parents,
        max_num_check=max_num_check,
        start_num_parents=start_num_parents,
        optimal_parents_dict=optimal_parents_dict,
        saved_scores=saved_scores,
        num_samples=num_samples,
        use_only_saved=False,
        max_allowed_score=network_score,
        top_order=top_order
    )

    # Now iteratively check if we can permute any of the vertices and achieve a better graph.
    # We'll terminate once a fixed point has been reached.
    curr_order = list(nx.topological_sort(network))
    prev_order = None

    for num_swap in range(2, max_num_swap + 1):
        while np.any(curr_order != prev_order):
            prev_order = deepcopy(curr_order)

            # Generate a bunch of permutations and propose that we swap the corresponding vertices in topological order.
            # Then rescore.
            for permutation in itertools.permutations(range(len(curr_order)), num_swap):
                proposed_order = deepcopy(curr_order)
                aux = proposed_order[permutation[0]]
                for i in range(num_swap):
                    next_pos = permutation[(i+1) % num_swap]
                    aux, proposed_order[next_pos] = proposed_order[next_pos], aux

                # Score the proposal graph
                proposed_graph, proposed_score = optimal_network(
                    df=df,
                    feature_values_dict=feature_values_dict,
                    rng=rng,
                    max_num_parents=max_num_parents,
                    max_num_check=max_num_check,
                    start_num_parents=start_num_parents,
                    optimal_parents_dict=optimal_parents_dict,
                    saved_scores=saved_scores,
                    num_samples=num_samples,
                    use_only_saved=use_only_saved,
                    max_allowed_score=best_score,
                    top_order=top_order_dict_from_array(proposed_order)
                )

                # Replace the best score and network if we've done better
                if proposed_score < best_score:
                    best_network = proposed_graph
                    best_score = proposed_score
                    curr_order = proposed_order

    return best_network, best_score


def compute(infile, outfile):
    df = pd.read_csv(infile)
    if infile == "medium.csv":
        feature_values_dict = feature_values(df, constant_support=[1,2,3,4,5])
    else:
        feature_values_dict = feature_values(df)
    saved_scores = {}

    start_time = time.time()

    if infile == "small.csv":
        # Small enough that we can exhaustively optimize.
        network, network_score = optimal_network(
            df=df,
            feature_values_dict=feature_values_dict,
            rng=np.random.default_rng(228),
            max_num_parents=5,
            max_num_check=10,
            start_num_parents=7,
            saved_scores=saved_scores,
            num_samples=10,
            optimal_parents_dict=None,
            use_only_saved=False
        )

    elif infile == "medium.csv":
        # A bit larger, so let's establish a baseline that we can use to bound the score.
        _, base_score = optimal_network(
            df=df,
            feature_values_dict=feature_values_dict,
            rng=np.random.default_rng(228),
            max_num_parents=2,
            saved_scores=saved_scores,
            num_samples=0,
            use_only_saved=True
        )

        print(f"Finished establishing a baseline. Time for this step: {time.time() - start_time} seconds.")
        int_time = time.time()

        # Now let's compute the optimal parents
        optimal_parents_dict = parent_selection_all_predecessors(
            df=df,
            feature_values_dict=feature_values_dict,
            rng=np.random.default_rng(228),
            saved_scores=saved_scores,
            max_num_parents=7
        )

        print(f"Finished computing optimal parents with all predecessors. "
              f"Time for this step: {time.time() - int_time} seconds.")
        int_time = time.time()

        # We'll generate a bunch of random Legos just to cover our bases a bit better
        construct_random_legos_with_optimal_parents(
            df=df,
            feature_values_dict=feature_values_dict,
            saved_scores=saved_scores,
            rng=np.random.default_rng(228),
            num_legos=200,
            max_num_parents=7,
            max_num_check=10,
            start_num_parents=7,
            num_samples=10
        )

        print(f"Finished constructing random Legos. Timefor this step: {time.time() - int_time} seconds.")
        int_time = time.time()

        # We'll impose some heuristic constraints.
        network, network_score = optimal_network(
            df=df,
            feature_values_dict=feature_values_dict,
            rng=np.random.default_rng(228),
            max_num_parents=7,
            saved_scores=saved_scores,
            optimal_parents_dict=optimal_parents_dict,
            num_samples=0,
            use_only_saved=True,
            max_allowed_score=base_score
        )

        print(f"Finished finding an approximate local optimum. Time for this step: {time.time() - int_time} seconds.")
        int_time = time.time()

        network, network_score = post_processing(
            network=network,
            network_score=network_score,
            df=df,
            feature_values_dict=feature_values_dict,
            rng=np.random.default_rng(228),
            saved_scores=saved_scores,
            optimal_parents_dict=optimal_parents_dict,
            max_num_parents=7,
            max_num_check=10,
            start_num_parents=7,
            num_samples=10,
            use_only_saved=False,
            max_num_swap=2
        )

    elif infile == "large.csv":
        # Too large to even bother trying for optimal. Let's just run a heuristic search.
        optimal_parents_dict = parent_selection_all_predecessors(
            df=df,
            feature_values_dict=feature_values_dict,
            rng=np.random.default_rng(228),
            saved_scores=saved_scores,
            max_num_parents=7
        )

        print(f"Finished finding optimal parents given all predecessors. "
              f"Time for this step: {time.time() - start_time} seconds.")
        int_time = time.time()

        # Let's generate some random Legos to cover our bases.
        construct_random_legos_with_optimal_parents(
            df=df,
            feature_values_dict=feature_values_dict,
            saved_scores=saved_scores,
            rng=np.random.default_rng(228),
            num_legos=200,
            max_num_parents=7,
            max_num_check=10,
            start_num_parents=7,
            num_samples=10
        )

        print(f"Finished generating random Legos. Time for this step: {time.time() - int_time} seconds.")
        int_time = time.time()

        # We'll just fix a topological order and hope for the best.
        top_order = top_order_dict_from_array(list(df.columns))
        network, network_score = optimal_network(
            df=df,
            feature_values_dict=feature_values_dict,
            rng=np.random.default_rng(228),
            saved_scores=saved_scores,
            optimal_parents_dict=optimal_parents_dict,
            max_num_parents=7,
            top_order=top_order,
            use_only_saved=False
        )

        print(f"Finished finding optimum for fixed topological order. "
              f"Time for this step: {time.time() - int_time} seconds.")
        int_time = time.time()

        network, network_score = post_processing(
            network=network,
            network_score=network_score,
            df=df,
            feature_values_dict=feature_values_dict,
            rng=np.random.default_rng(228),
            saved_scores=saved_scores,
            optimal_parents_dict=optimal_parents_dict,
            max_num_parents=7,
            max_num_check=10,
            start_num_parents=7,
            num_samples=10,
            use_only_saved=True
        )


    print(f"Time elapsed: {time.time() - start_time} seconds.")
    write_gph(network, outfile)
    print(f"Score of Network: {-network_score}.")
    nx.draw(network, with_labels=True)
    plt.savefig(outfile[:-4]+'.png', dpi=300, bbox_inches='tight')

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
