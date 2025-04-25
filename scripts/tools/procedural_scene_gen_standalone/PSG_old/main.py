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

import omni.kit.commands

from scripts.procedural_scene_gen_standalone.PSG_old.placement import create_floor
from scripts.procedural_scene_gen_standalone.PSG_old.save import SaveManager
from scripts.procedural_scene_gen_standalone.PSG_old.traversal import SceneTraversal
from scripts.procedural_scene_gen_standalone.PSG_old.utils import build_parent_child_graph, load_config

from omni.isaac.lab.utils.assets import NVIDIA_NUCLEUS_DIR


def main():
    """
    Main entry point for scene generation.

    The complete scene generation pipeline consists of these key steps:
    1. Load configuration from YAML file - defines objects, relationships, and parameters
    2. Build parent-child relationship graph - creates a DAG of object dependencies
    3. Generate scenes using graph traversal - processes objects in topological order
    4. Save results in multiple formats - preserves both visual and data representations

    The system ensures:
    - Configuration is properly loaded and validated
    - Scene generation follows parent-child relationships (parents processed first)
    - All objects respect physical constraints (no overlapping, proper clearances)
    - Parent-child placement rules are followed (e.g., chairs around tables)
    - Performance optimization via subtraversal for multi-scene generation
    - Complete debug information is logged when enabled
    """
    # Step 1: Initialize the pipeline components
    # -----------------------------------------
    # Load configuration from YAML - contains object definitions, scene parameters, output settings
    config = load_config()

    # Build parent-child graph - creates directed acyclic graph of relationships
    # (e.g., Floor -> Table -> Chair, where arrows represent "parent of" relationship)
    parent_child_graph = build_parent_child_graph(config)

    # Initialize traversal system - handles the scene generation logic
    # This component manages the process of creating scenes according to the graph
    traversal = SceneTraversal(parent_child_graph, config)

    # Initialize save manager - handles output generation in various formats
    save_manager = SaveManager(config)

    # Step 2: Generate all scenes
    # --------------------------
    # For scene count > 1, this will optimize using subtraversal
    # First scene is generated with full traversal, subsequent scenes reuse valid positions
    print("Started Creating Scenes.")
    scenes = traversal.generate_scenes()
    print("Finished Creating Scenes.")

    # Step 3: Save results in requested formats
    # ----------------------------------------
    # Process each generated scene
    print("Started Saving Files.")

    for i, scene_data in enumerate(scenes):
        # Save JSON serialization of scene data (positions, rotations, etc.)
        if config["output"]["save"]["positional_info"]:
            save_manager.save_positional_info(scene_data, i)

        # Save USD file for 3D visualization
        if config["output"]["save"]["usd"]:
            save_manager.save_usd(scene_data, i)

    # Save additional visualizations if enabled in config
    # Parent-child graph shows the object relationships visually
    if config["output"]["save"]["parent_child_graph"]:
        save_manager.save_parent_child_graph(traversal.nx_graph)

    print("Finished Saving Files.")


if __name__ == "__main__":
    main()
