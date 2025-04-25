# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Utility functions and classes for procedural scene generation.

This module provides core functionality used throughout the scene generation pipeline:
1. Configuration loading and parsing
2. Parent-child relationship graph building
3. Logging and timing utilities
4. Asset management

The utilities in this module are designed to be reusable across different parts
of the application, providing consistent interfaces for common operations.
"""

import csv
import os
import random
import time
import yaml
from collections import defaultdict

import networkx as nx


class Logger:
    """
    Comprehensive logging system for debug messages and performance timing.

    This class provides structured logging functionality for:
    - Debug messages with customizable verbosity
    - Performance timing metrics for scene generation
    - CSV-formatted timing data for analysis

    The logger creates and manages all log files automatically based on the configuration,
    ensuring consistent logging across the application and preserving run history.
    """

    def __init__(self, config):
        """
        Initialize logger with configuration.

        Sets up log directories and files according to the provided configuration.
        Creates CSV timing logs with appropriate headers when timing is enabled.

        Args:
            config (dict): Configuration dictionary containing:
                - output.save_dir: Base directory for saving output
                - output.run_name: Name of the current run (used for directory naming)
                - debug.log_debug: Whether to enable debug logging
                - debug.log_debug_file: Name of debug log file
                - debug.log_timing: Whether to enable timing logging
                - debug.log_timing_file: Name of timing log file
        """
        self.config = config
        # Create full path to save directory using run name for isolation
        self.save_dir = os.path.join(config["output"]["save_dir"], config["output"]["run_name"])
        # Ensure directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize debug log file path
        self.debug_log_file = os.path.join(self.save_dir, config["debug"]["log_debug_file"])

        # Create time log file with headers if timing is enabled
        if self.config["debug"]["log_timing"]:
            self.time_log_file = os.path.join(self.save_dir, config["debug"]["log_timing_file"])
            # Initialize CSV with appropriate headers
            with open(self.time_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Scene", "Objects", "Time (s)"])

    def debug(self, message):
        """
        Write debug message to log file if debug logging is enabled.

        Messages are written with newlines to ensure proper formatting.
        Logging is controlled by the debug.log_debug configuration flag.

        Args:
            message (str): Debug message to write to the log file
        """
        if self.config["debug"]["log_debug"]:
            with open(self.debug_log_file, "a") as f:
                f.write(f"{message}\n")

    def timing(self, scene_number, total_objects, time_taken):
        """
        Write timing data to CSV log file if timing logging is enabled.

        Records performance metrics for each scene generation:
        - Scene number (1-indexed for user readability)
        - Total number of objects placed in the scene
        - Time taken to generate the scene (in seconds, formatted to 3 decimal places)

        Args:
            scene_number (int): Scene identifier (0-indexed, internally converted to 1-indexed)
            total_objects (int): Total number of objects successfully placed in the scene
            time_taken (float): Time taken to generate scene (in seconds)
        """
        if self.config["debug"]["log_timing"]:
            with open(self.time_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                # Convert scene number to 1-indexed for user readability
                writer.writerow([scene_number + 1, total_objects, f"{time_taken:.3f}"])


def load_config(config_path="config.yaml"):
    """
    Load and parse configuration from YAML file.

    This function reads the YAML configuration file that defines:
    - Scene objects and their properties (size, clearance, etc.)
    - Parent-child relationships between objects
    - Generation parameters (number of scenes, optimization settings)
    - Output settings (file formats, directories)
    - Debug options (logging, timing)

    Args:
        config_path (str): Path to the YAML configuration file (default: 'config.yaml')

    Returns:
        dict: Parsed configuration dictionary with nested structure matching the YAML
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_node_heights(graph):
    """
    Compute the height of each node in a directed acyclic graph (DAG).

    The height of a node is defined as the maximum distance to any leaf node.
    This is used in the scene generation to:
    1. Determine which objects to respawn in subtraversals
    2. Visualize the parent-child graph with proper layering
    3. Process nodes in optimal order when needed

    Implementation uses depth-first search (DFS) to ensure all paths are explored:
    - Leaf nodes (no children) have height 0
    - Parent nodes have height = 1 + max(child_heights)

    Args:
        graph (nx.DiGraph): NetworkX directed graph representation

    Returns:
        dict: Dictionary mapping each node to its height in the graph
    """
    height = {}

    def dfs(node):
        """
        Recursive depth-first search to compute node heights.

        Base case: Leaf nodes (no successors) have height 0
        Recursive case: Node height is 1 + max of all child heights

        Args:
            node: Current node being processed

        Returns:
            int: Height of the current node
        """
        # Base case: node is a leaf or not in graph
        if node not in graph or len(list(graph.successors(node))) == 0:
            height[node] = 0
            return 0
        # Recursive case: height is 1 + max height of children
        h = max(dfs(child) for child in graph.successors(node)) + 1
        height[node] = h
        return h

    # Process all nodes in topological order to ensure dependencies are handled correctly
    for node in nx.topological_sort(graph):
        if node not in height:
            dfs(node)

    return height


def build_parent_child_graph(config):
    """
    Build a directed graph representation of parent-child relationships.

    Creates a directed acyclic graph (DAG) where:
    - Nodes represent object types (Floor, Table, Chair, etc.)
    - Edges represent parent-child relationships (Floor → Table → Chair)
    - Direction is from parent to child (parent is source, child is target)

    This graph is central to the scene generation process as it determines:
    1. The order in which objects are processed (parents before children)
    2. Valid placement relationships (e.g., chairs are placed relative to tables)
    3. Subtraversal optimization strategies

    Args:
        config (dict): Configuration dictionary from YAML containing object definitions

    Returns:
        dict: Adjacency list representation of the graph where:
              - Keys are parent nodes
              - Values are lists of child nodes
    """
    # Initialize the graph as a defaultdict to handle missing keys
    graph = defaultdict(list)

    # Build edges based on parent relationships defined in configuration
    for obj_name, obj_data in config["scene"].items():
        # For each object, add edges from all its parents
        for parent in obj_data["parents"]:
            # Add directed edge from parent to child
            # (This represents "parent is parent of child")
            graph[parent].append(obj_name)

    return graph


def get_random_asset(config, obj_type):
    """
    Get a random USD asset file for the given object type.

    This function:
    1. Locates the asset directory for the specified object type
    2. Finds all USD files in that directory
    3. Randomly selects one USD file for variety in scene generation

    USD (Universal Scene Description) files contain 3D model data that will be
    placed in the scene. Random selection ensures visual variety across instances.

    Args:
        config (dict): Configuration dictionary containing asset directory settings
        obj_type (str): Name of the object type (e.g., "Table", "Chair", "Box")

    Returns:
        str or None: Full path to randomly selected USD file, or None if:
                    - The asset directory doesn't exist
                    - No USD files are found in the directory
    """
    # Get base assets directory from configuration
    assets_dir = config["output"]["asset_dir"]
    # Construct path to specific object type's asset directory
    asset_dir = os.path.join(assets_dir, obj_type)

    # Check if directory exists
    if not os.path.exists(asset_dir):
        return None

    # Get all USD files in the directory
    asset_files = [f for f in os.listdir(asset_dir) if f.endswith(".usd")]

    # If no USD files found, return None
    if not asset_files:
        return None

    # Select random asset file for variety
    selected_file = random.choice(asset_files)
    return os.path.join(asset_dir, selected_file)
