# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Utility functions for procedural scene generation.

This module provides helper functions used across the scene generation pipeline:
1. Configuration loading and validation
2. Parent-child relationship graph construction and analysis
3. Graph traversal utilities for scene generation
4. Subtraversal optimization support

These utilities form the foundation of the procedural scene generation system,
providing core functionality that is used by the traversal and scene generation modules.
"""

import os
import random
import yaml
from collections import defaultdict

import networkx as nx


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file and ensure output directories exist.

    This function:
    1. Reads and parses the YAML configuration file
    2. Creates the output directory structure based on config settings

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary with all scene and generation parameters
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create output directory structure
    save_dir = os.path.join(config["output"]["save_dir"], config["output"]["run_name"])
    os.makedirs(save_dir, exist_ok=True)

    return config


def build_parent_child_graph(config):
    """
    Build a directed graph representing parent-child relationships between objects.

    This function analyzes the scene configuration to create a directed acyclic graph (DAG)
    where edges represent parent-child relationships between objects. The graph is
    represented as an adjacency list where keys are parents and values are lists of children.

    Args:
        config (dict): Configuration dictionary with scene object definitions

    Returns:
        dict: Adjacency list representation of the parent-child graph
    """
    graph = defaultdict(list)

    # Add edges based on parent relationships in config
    for obj_name, obj_data in config["scene"].items():
        # Add edges from parents to this object
        for parent in obj_data["parents"]:
            if parent:  # Skip empty parent entries
                graph[parent].append(obj_name)

        # Ensure objects with no children are included in the graph (leaf nodes)
        if obj_name not in graph:
            graph[obj_name] = []

    return dict(graph)


def compute_node_depths(graph):
    """
    Compute the depth of each node in the graph (distance from root).

    Depth is defined as the distance from the root node:
    - Root nodes (no parents) have depth 0
    - Their children have depth 1, and so on

    This function works with both dictionary graphs and NetworkX DiGraph objects.

    Args:
        graph: Either a dictionary adjacency list or a NetworkX DiGraph

    Returns:
        dict: Mapping of nodes to their depths in the graph
    """
    # Convert NetworkX graph to dictionary if needed
    if isinstance(graph, nx.DiGraph):
        graph_dict = {node: list(graph.successors(node)) for node in graph.nodes()}
    else:
        graph_dict = graph

    # Find root nodes (nodes with no parents)
    all_nodes = set(graph_dict.keys()).union(*[set(children) for children in graph_dict.values() if children])
    child_nodes = set()
    for children in graph_dict.values():
        child_nodes.update(children)
    root_nodes = all_nodes - child_nodes

    # Compute depths with DFS
    depths = {}
    visited = set()

    def dfs(node, depth=0):
        """
        Depth-first search helper function to compute node depths.

        Args:
            node: Current node being processed
            depth (int): Current depth in the traversal
        """
        if node in visited:
            return

        visited.add(node)
        depths[node] = depth

        for child in graph_dict.get(node, []):
            dfs(child, depth + 1)

    # Run DFS from each root
    for root in root_nodes:
        dfs(root)

    return depths


def determine_objects_to_replace(graph, config):
    """
    Determine which objects (and their descendants) should be replaced during subtraversals.

    This function implements the logic for selecting objects to replace based on:
    1. Selective vs non-selective mode (specific objects vs depth-based selection)
    2. Forced vs non-forced mode (all candidates vs single random candidate)

    It also recursively includes all descendants of selected objects.

    Args:
        graph (dict): Parent-child graph structure
        config (dict): Configuration dictionary with subtraversal settings

    Returns:
        set: Set of object types to replace during subtraversals
    """
    objects_to_replace = set()
    sub_config = config["generation"]["subtraversal"]
    node_depths = compute_node_depths(graph)

    # Step 1: Determine candidates based on selection mode
    if sub_config["selective"]:
        # Use explicitly listed objects from config
        candidates = set(sub_config["objects"])
    else:
        # Use objects at the specified depth in the graph
        depth = sub_config["depth"]
        candidates = {node for node, node_depth in node_depths.items() if node_depth == depth}

    # Step 2: Select candidates based on forced/non-forced mode
    if sub_config["forced"]:
        # Take all candidates when in forced mode
        selected = candidates
    else:
        # Take one random candidate in non-forced mode
        selected = {random.choice(list(candidates))} if candidates else set()

    # Step 3: Add selected objects and their descendants
    for obj in selected:
        objects_to_replace.add(obj)
        add_descendants(graph, obj, objects_to_replace)

    return objects_to_replace


def add_descendants(graph, node, result_set):
    """
    Add all descendants of a node to the result set recursively.

    This helper function finds all children and their children recursively,
    adding them to the result set. This ensures that when we replace an object
    in the scene, we also replace all its dependent objects.

    Args:
        graph (dict): Parent-child graph structure
        node: Node to find descendants of
        result_set: Set to add descendants to
    """
    for child in graph.get(node, []):
        result_set.add(child)
        add_descendants(graph, child, result_set)
