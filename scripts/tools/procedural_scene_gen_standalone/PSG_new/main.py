# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

# =============================================================================
# Import Modules and Initialize Application
# =============================================================================

import argparse

from omni.isaac.lab.app import AppLauncher

# Add argparse arguments to parse command line inputs for the AppLauncher
parser = argparse.ArgumentParser(description="Load Anymal USD in sim and verify physical properties")
# Append AppLauncher CLI args to the parser (adds additional necessary parameters)
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments from the command line
args_cli = parser.parse_args()
# Set headless and offscreen render modes to True for simulation without GUI
args_cli.headless = True
args_cli.offscreen_render = True

# Launch the Omniverse application via the AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Additional Module Imports and Global Configurations
# =============================================================================

import time

from scripts.procedural_scene_gen_standalone.PSG_new.save import SaveManager
from scripts.procedural_scene_gen_standalone.PSG_new.traversal import Traversal
from scripts.procedural_scene_gen_standalone.PSG_new.utils import build_parent_child_graph, load_config


def main():
    """
    Main function that executes the complete scene generation pipeline.

    The pipeline consists of four main stages:
    1. Configuration loading - Parse YAML config with scene definitions
    2. Graph construction - Build parent-child relationships between objects
    3. Scene generation - Traverse the graph to create scenes with proper placement
    4. Scene saving - Output scenes in requested formats (USD, JSON, visualizations)

    Returns:
        int: Status code (0 for success)
    """
    # Stage 1: Load configuration from YAML
    # This contains all object definitions, instance counts, relationships, etc.
    config = load_config("config.yaml")

    # Stage 2: Build parent-child relationship graph
    # This creates a directed acyclic graph (DAG) of object relationships
    graph = build_parent_child_graph(config)

    # Stage 3: Generate scenes using traversal algorithm
    # This traverses the graph to create scenes with proper parent-child relationships
    traversal = Traversal(graph, config)
    scenes = traversal.generate_scenes()

    # Stage 4: Save generated scenes
    # This saves scenes in the formats specified in the configuration
    save_manager = SaveManager(config)
    save_manager.save_manager(scenes, graph)

    return 0


if __name__ == "__main__":
    main()
