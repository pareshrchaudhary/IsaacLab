# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Traversal module for procedural scene generation.

This module handles traversal of the parent-child graph to generate scenes by:
1. Implementing depth-first traversal of the parent-child graph
2. Ensuring objects are processed in the correct order (parents before children)
3. Supporting optimized subtraversal for efficient multi-scene generation
4. Tracking traversal order for scene construction
5. Caching valid positions and scale factors for performance optimization

The traversal logic maintains the hierarchical relationships between objects,
which is critical for proper procedural scene generation.
"""

import copy
import csv
import os
import random
import time

from scripts.procedural_scene_gen_standalone.PSG_new.placement import ObjectManager, PlacementManager
from scripts.procedural_scene_gen_standalone.PSG_new.utils import compute_node_depths, determine_objects_to_replace


class Traversal:
    """
    Main traversal class for scene generation.

    This class implements two key traversal algorithms:
    1. Base traversal - Complete DFS traversal of the parent-child graph
    2. Sub-traversal - Optimized traversal that only replaces selected objects

    The traversal ensures that parent objects are always processed before their children,
    maintaining the hierarchical dependencies required for proper scene construction.

    Optimizations:
    - Caches valid positions for objects that will be replaced in subtraversals
    - Caches scale factors to avoid redundant calculations
    - Prevents object overlap by tracking used positions
    """

    def __init__(self, graph, config):
        """
        Initialize traversal with graph and configuration.

        Sets up the internal state for traversing the graph and processing objects.
        Computes node depths which are needed for certain traversal operations.
        Also initializes optimization data structures for caching valid positions and scales.

        Args:
            graph (dict): Adjacency list of parent-child relationships
            config (dict): Scene configuration from YAML
        """
        self.graph = graph
        self.config = config
        self.node_depths = compute_node_depths(graph)
        self.object_manager = ObjectManager(config)
        self.placement_manager = PlacementManager(config)

        # Performance optimization data structures
        # =======================================

        # Cache of valid positions for objects that will be replaced in subtraversals
        # Format: {obj_type: {parent_id: [positions]}}
        # This avoids recalculating valid positions for the same parent-child pairs
        self.cached_valid_positions = {}

        # Cache of scale values for objects that will be replaced in subtraversals
        # Format: {obj_type: {asset_path: (scale_factor, true_size)}}
        # This avoids recalculating scale factors for the same assets
        self.cached_scale_values = {}

        # Determine which objects will be replaced in subtraversals
        if config["generation"]["num_scenes"] > 1:
            # Get the set of objects to replace from configuration
            self.objects_to_replace = determine_objects_to_replace(graph, config)

            # Initialize caches for objects that will be replaced
            for obj_type in self.objects_to_replace:
                self.cached_valid_positions[obj_type] = {}
                self.cached_scale_values[obj_type] = {}

                # Also initialize for direct children of replaced objects
                # since their scales should also be cached
                for child in graph.get(obj_type, []):
                    self.cached_scale_values[child] = {}
        else:
            self.objects_to_replace = set()

        # Set up timing log file if enabled in configuration
        if self.config["output"].get("log_timing_file"):
            self.time_log_file = os.path.join(
                self.config["output"]["save_dir"],
                self.config["output"]["run_name"],
                self.config["output"]["log_timing_file"],
            )
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.time_log_file), exist_ok=True)

            # Initialize the CSV file with headers
            with open(self.time_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Scene Number", "Object Count", "Generation Time (s)"])

        self.reset_state()

    def reset_state(self):
        """
        Reset traversal state for a new scene generation.

        Clears tracking sets and lists to prepare for a fresh traversal:
        - visited: Tracks nodes already processed to avoid duplication
        - path: Tracks current traversal path for cycle detection
        - traversal_order: Records the order in which nodes are visited
        """
        self.visited = set()
        self.path = set()
        self.traversal_order = []

    def generate_scenes(self):
        """
        Generate scenes based on configuration.

        This is the main entry point for scene generation, which:
        1. Creates an initial base scene with full traversal
        2. Creates additional scenes with optimized subtraversal
        3. Tracks and returns all generated scenes
        4. Records timing information if enabled

        Returns:
            list: List of generated scene data dictionaries
        """
        num_scenes = self.config["generation"]["num_scenes"]
        scenes = []

        # Create a list to store timing data
        timing_data = []

        # PHASE 1: Generate base scene
        # ===========================

        # Start timing for base scene generation
        base_scene_start_time = time.time()

        # Generate base scene using full traversal
        if num_scenes > 0:
            self.reset_state()
            self.base_traversal()

            # Create the base scene by processing objects in traversal order
            base_scene = self.create_base_scene()
            scenes.append(base_scene)

            # End timing for base scene
            base_scene_end_time = time.time()
            base_scene_generation_time = base_scene_end_time - base_scene_start_time

            # If timing is enabled, collect data for later writing
            if self.config["output"]["save"].get("log_timing", False) and hasattr(self, "time_log_file"):
                # Count total objects in scene (excluding Floor)
                total_objects = sum(
                    len(obj_list) for obj_name, obj_list in base_scene["scene"].items() if obj_name != "Floor"
                )

                # Add to timing data
                timing_data.append([1, total_objects, f"{base_scene_generation_time:.10f}"])

        # PHASE 2: Generate additional scenes
        # ==================================

        # Generate additional scenes using optimized subtraversal
        if num_scenes > 1:
            # Determine which objects to replace in subtraversals
            if not hasattr(self, "objects_to_replace"):
                self.objects_to_replace = determine_objects_to_replace(self.graph, self.config)

            # Start timing for additional scenes
            additional_scenes_start_time = time.time()

            # Create each additional scene
            for scene_idx in range(1, num_scenes):
                # Start timing for this scene
                scene_start_time = time.time()

                # Reset traversal state
                self.reset_state()

                # Get the set of objects to be replaced, including the specified objects
                # and all their descendant objects
                objects_to_replace = self.objects_to_replace.copy()

                # Find all children of objects to replace
                for obj_type in list(objects_to_replace):
                    # Add all descendants to the set
                    self._add_all_descendants(obj_type, objects_to_replace)

                print(f"[SUBTRAVERSAL] Scene {scene_idx+1}: Objects to replace: {sorted(list(objects_to_replace))}")

                # Create a deep copy of the base scene
                scene = copy.deepcopy(base_scene)

                # Clear all objects that need to be replaced from the scene
                # This includes both the specified objects and their children
                for obj_type in objects_to_replace:
                    scene["scene"][obj_type] = []
                    print(f"[SUBTRAVERSAL] Scene {scene_idx+1}: Cleared {obj_type} objects")

                # Get a topologically sorted list of objects to ensure parents are processed before children
                topo_sorted_objects = self._get_topological_sort(objects_to_replace)
                print(f"[SUBTRAVERSAL] Scene {scene_idx+1}: Topological order for replacement: {topo_sorted_objects}")

                # Process objects from lowest to highest depth to ensure parents are placed before children
                for obj_type in topo_sorted_objects:
                    if obj_type == "Floor":
                        # Floor is always at the origin, no need to replace
                        continue
                    else:
                        # Place objects based on their parents
                        # is_base_scene=False because we're in a subtraversal
                        print(f"[SUBTRAVERSAL] Scene {scene_idx+1}: Placing {obj_type}")
                        num_instances = self.config["scene"][obj_type]["instances"]
                        print(f"[SUBTRAVERSAL] Scene {scene_idx+1}: {obj_type} has {num_instances} instances")
                        placed_count = self.place_objects(scene, obj_type, is_base_scene=False)
                        print(f"[SUBTRAVERSAL] Scene {scene_idx+1}: Placed {placed_count} {obj_type} objects")

                scenes.append(scene)

                # End timing for this scene
                scene_end_time = time.time()
                scene_generation_time = scene_end_time - scene_start_time

                # If timing is enabled, collect data for later writing
                if self.config["output"]["save"].get("log_timing", False) and hasattr(self, "time_log_file"):
                    # Count total objects in scene (excluding Floor)
                    total_objects = sum(
                        len(obj_list) for obj_name, obj_list in scene["scene"].items() if obj_name != "Floor"
                    )

                    # Add to timing data
                    timing_data.append([scene_idx + 1, total_objects, f"{scene_generation_time:.10f}"])

            # End timing for additional scenes
            additional_scenes_end_time = time.time()
            additional_scenes_time = additional_scenes_end_time - additional_scenes_start_time

            # If timing is enabled, add summary for additional scenes
            if self.config["output"]["save"].get("log_timing", False) and hasattr(self, "time_log_file"):
                timing_data.append(["All Additional", num_scenes - 1, f"{additional_scenes_time:.10f}"])

        # PHASE 3: Write timing data
        # =========================

        # Write all timing data at once
        if self.config["output"]["save"].get("log_timing", False) and hasattr(self, "time_log_file") and timing_data:
            with open(self.time_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Scene Number", "Object Count", "Generation Time (s)"])
                writer.writerows(timing_data)

        return scenes

    def create_base_scene(self):
        """
        Create a scene based on the base traversal order.

        This method processes objects in the traversal_order to build a complete scene,
        placing objects according to their parent-child relationships.

        Returns:
            dict: Complete scene data
        """
        # Initialize an empty scene with a scene dictionary to hold objects by type
        scene_data = {"scene": {}}

        # Process each object type in the traversal order
        for obj_type in self.traversal_order:
            # Initialize list for this object type
            scene_data["scene"][obj_type] = []

            if obj_type == "Floor":
                # Floor is always at the origin
                obj_data = self.object_manager.create_floor()
                scene_data["scene"][obj_type].append(obj_data)
            else:
                # Place other objects based on their parents
                # is_base_scene=True triggers caching of valid positions and scales
                self.place_objects(scene_data, obj_type, is_base_scene=True)

        return scene_data

    def place_objects(self, scene_data, obj_type, is_base_scene=False):
        """
        Place objects of a given type in the scene based on their parent-child relationships.

        This method:
        1. Identifies the parent type for the object
        2. Gets valid positions for the object based on its parent
        3. Randomly selects positions for each instance of the object
        4. Ensures objects don't overlap by tracking used positions
        5. Caches valid positions and scale values for optimization when is_base_scene is True

        Args:
            scene_data (dict): Current scene data
            obj_type (str): Type of object to place
            is_base_scene (bool): Whether this is the base scene generation or not
        """
        # Get number of instances for this object type
        num_instances = self.config["scene"][obj_type]["instances"]

        # Get parent type for this object
        parent_types = self.config["scene"][obj_type]["parents"]

        # Skip if no parents defined (should not happen except for Floor)
        if not parent_types:
            return 0

        # For now, just use the first parent type (this will be expanded later)
        parent_type = parent_types[0]

        # Debug: Check if this is a subtraversal and if the parent was replaced
        is_subtraversal = not is_base_scene
        parent_was_replaced = is_subtraversal and parent_type in self.objects_to_replace
        if is_subtraversal:
            print(
                f"[SUBTRAVERSAL] {obj_type}: Parent type {parent_type} was"
                f" {'replaced' if parent_was_replaced else 'not replaced'}"
            )

        # Get all parent instances from the scene
        parent_instances = scene_data["scene"].get(parent_type, [])

        # Skip if no parent instances available
        if not parent_instances:
            print(f"[PLACE] {obj_type}: No parent instances of type {parent_type} available in the scene")
            return 0

        # Print debug info about parents
        print(f"[PLACE] {obj_type}: Found {len(parent_instances)} parent instances of type {parent_type}")
        for i, parent in enumerate(parent_instances):
            print(
                f"[PLACE] {obj_type}: Parent #{i} position = {parent['position']} rotation ="
                f" {parent.get('rotation', [0, 0, 0])}"
            )

        # Check if strict orientation is required for this object type
        strict_orientation = self.config["scene"][obj_type].get("strict_orientation", False)
        placement_types = self.config["scene"][obj_type]["placement"]

        # Track available positions for each parent to prevent overlaps within this scene
        available_positions_by_parent = {}

        # Track placed objects count
        placed_count = 0

        # Create and place each instance of this object type
        for instance_idx in range(num_instances):
            # Randomly select a parent instance
            parent_data = random.choice(parent_instances)
            parent_idx = parent_instances.index(parent_data)
            parent_id = f"{parent_type}_{parent_idx}"

            print(
                f"[PLACE] {obj_type}: Instance #{instance_idx} - Selected parent #{parent_idx} at"
                f" {parent_data['position']}"
            )

            # Check if we already have calculated available positions for this parent in this scene
            if parent_id in available_positions_by_parent:
                valid_positions = available_positions_by_parent[parent_id]

                # If no valid positions left for this parent, try another parent
                if not valid_positions:
                    print(f"[PLACE] {obj_type}: No valid positions left for parent {parent_id}, trying another parent")
                    # Try to find another parent with available positions
                    if parent_data in parent_instances and len(parent_instances) > 1:
                        # Create a copy to avoid modifying the original list
                        alternative_parents = parent_instances.copy()
                        alternative_parents.remove(parent_data)

                        # Try to find a parent with available positions
                        parent_found = False
                        for alt_parent in alternative_parents:
                            alt_parent_idx = parent_instances.index(alt_parent)
                            alt_parent_id = f"{parent_type}_{alt_parent_idx}"

                            # Check if this parent has positions available or hasn't been tried yet
                            if (
                                alt_parent_id not in available_positions_by_parent
                                or available_positions_by_parent[alt_parent_id]
                            ):
                                parent_data = alt_parent
                                parent_idx = alt_parent_idx
                                parent_id = alt_parent_id

                                # Get positions if already calculated
                                if alt_parent_id in available_positions_by_parent:
                                    valid_positions = available_positions_by_parent[alt_parent_id]
                                else:
                                    valid_positions = None  # Will be calculated below

                                parent_found = True
                                print(
                                    f"[PLACE] {obj_type}: Found alternative parent #{alt_parent_idx} at"
                                    f" {alt_parent['position']}"
                                )
                                break

                        # Skip this instance if no suitable parent found
                        if not parent_found:
                            print(
                                f"[PLACE] {obj_type}: No alternative parents with valid positions found. Skipping"
                                f" instance #{instance_idx}"
                            )
                            continue
                    else:
                        # No other parents available, skip this instance
                        print(
                            f"[PLACE] {obj_type}: No alternative parents available. Skipping instance #{instance_idx}"
                        )
                        continue
            else:
                # First time using this parent, positions will be calculated below
                valid_positions = None

            # Calculate valid positions if not already done for this parent
            if valid_positions is None:
                # In subtraversals, NEVER use cached positions for children of replaced parents
                if not is_base_scene and parent_was_replaced:
                    # ALWAYS recalculate positions based on the new parent positions for children
                    # of replaced parents in subtraversals
                    print(
                        f"[PLACE] {obj_type}: FORCING recalculation of positions for child of replaced parent in"
                        " subtraversal"
                    )
                    valid_positions = self.placement_manager.get_valid_positions(
                        obj_type, parent_type, parent_data, scene_data  # Pass scene_data for overlap checking
                    )
                # Only use cached positions in appropriate cases
                elif (
                    not is_base_scene
                    and obj_type in self.cached_valid_positions
                    and parent_id in self.cached_valid_positions[obj_type]
                ):
                    # Use cached positions but make a copy to avoid modifying the original
                    valid_positions = self.cached_valid_positions[obj_type][parent_id].copy()
                    print(f"[PLACE] {obj_type}: Using {len(valid_positions)} cached positions for parent {parent_id}")
                else:
                    # Calculate new valid positions
                    print(f"[PLACE] {obj_type}: Calculating new valid positions for parent {parent_id}")
                    valid_positions = self.placement_manager.get_valid_positions(
                        obj_type, parent_type, parent_data, scene_data  # Pass scene_data for overlap checking
                    )
                    print(f"[PLACE] {obj_type}: Found {len(valid_positions)} valid positions for parent {parent_id}")

                    # Cache positions if this is the base scene and this object will be replaced in subtraversals
                    if is_base_scene and obj_type in self.objects_to_replace:
                        if obj_type not in self.cached_valid_positions:
                            self.cached_valid_positions[obj_type] = {}
                        # Store a deep copy to preserve the original list
                        self.cached_valid_positions[obj_type][parent_id] = valid_positions.copy()

                # Store available positions for this parent in this scene
                available_positions_by_parent[parent_id] = valid_positions

            # Skip if no valid positions found
            if not valid_positions:
                print(
                    f"[PLACE] {obj_type}: No valid positions found for parent {parent_id}. Skipping instance"
                    f" #{instance_idx}"
                )
                continue

            # Randomly select a position from the available positions
            position = random.choice(valid_positions)
            print(f"[PLACE] {obj_type}: Selected position {position} from {len(valid_positions)} available positions")

            # Remove the used position from available positions to prevent overlap
            # This ensures that subsequent objects of the same type won't use this position
            available_positions_by_parent[parent_id].remove(position)

            # Create the object using either cached scale values or calculating new ones
            obj_data = self.create_object_with_efficient_scaling(obj_type, is_base_scene)

            # Update object position
            obj_data["position"] = position

            # Apply orientation if needed (for objects requiring specific orientation)
            if strict_orientation or "Tuck" in placement_types:
                # Determine which placement type to use for orientation
                current_placement = placement_types[0] if isinstance(placement_types, list) else placement_types

                # Calculate proper orientation based on position relative to parent
                obj_data["rotation"] = self.placement_manager.calculate_orientation(
                    position, parent_data["position"], parent_data["size"], parent_data["rotation"], current_placement
                )

            # Update bounding box vertices based on new position and rotation
            obj_data["vertices"] = self.object_manager.calculate_bounding_box_vertices(
                position, obj_data["size"], obj_data["rotation"]
            )

            # Add object to scene
            scene_data["scene"][obj_type].append(obj_data)
            placed_count += 1
            print(f"[PLACE] {obj_type}: Successfully placed instance #{instance_idx} at {position}")

        print(f"[PLACE] {obj_type}: Placed {placed_count} instances of {num_instances} requested")
        return placed_count

    def create_object_with_efficient_scaling(self, obj_type, is_base_scene):
        """
        Create an object with efficient scale value calculation.

        Uses cached scale values when available to avoid redundant calculations.
        When creating objects in the base scene, caches scale values for future use.
        Also applies rotation from the configuration.

        Args:
            obj_type (str): Type of object to create
            is_base_scene (bool): Whether this is the base scene generation

        Returns:
            dict: Object data with properly scaled dimensions
        """
        # Get a random asset for this object type
        asset_path = self.object_manager.get_random_asset(obj_type)

        # Get rotation from config
        obj_rotation = self.config["scene"][obj_type].get("rotation", [0, 0, 0])

        # Check if we already have cached scale values for this asset
        if (
            not is_base_scene
            and obj_type in self.cached_scale_values
            and asset_path in self.cached_scale_values[obj_type]
        ):
            # Use cached scale values
            scale_factor, true_size = self.cached_scale_values[obj_type][asset_path]

            # Create object with cached values
            position = [0, 0, 0]  # Initially place at origin

            # Calculate bounding box vertices with rotation
            vertices = self.object_manager.calculate_bounding_box_vertices(position, true_size, obj_rotation)

            return {
                "position": position,
                "size": true_size,
                "rotation": obj_rotation,
                "asset_path": asset_path,
                "scale_factor": scale_factor,
                "vertices": vertices,
            }
        else:
            # No cached values, create object normally
            obj_data = self.object_manager.create_object(obj_type)

            # Cache the scale values if appropriate
            if asset_path and (
                is_base_scene
                and (
                    obj_type in self.objects_to_replace
                    or any(obj_type in self.graph.get(parent, []) for parent in self.objects_to_replace)
                )
            ):
                # Make sure the cache is initialized for this object type
                if obj_type not in self.cached_scale_values:
                    self.cached_scale_values[obj_type] = {}

                # Store scale values for future use
                self.cached_scale_values[obj_type][asset_path] = (obj_data["scale_factor"], obj_data["size"])

            return obj_data

    def base_traversal(self):
        """
        Perform base traversal of the graph using depth-first search.

        This is a complete traversal that:
        1. Identifies root nodes (nodes with no parents)
        2. Processes all nodes in depth-first order
        3. Records the order in which nodes are visited

        Returns:
            list: Nodes in traversal order
        """
        # Step 1: Find root nodes (nodes with no parents)
        all_nodes = set(self.graph.keys()).union(*[set(children) for children in self.graph.values()])
        child_nodes = set()
        for children in self.graph.values():
            child_nodes.update(children)
        root_nodes = all_nodes - child_nodes

        # Step 2: Start DFS from each root node
        for root in root_nodes:
            self._dfs_traverse(root)

        return self.traversal_order

    def _dfs_traverse(self, node):
        """
        Recursively traverse the graph using depth-first search.

        This function:
        1. Tracks visited nodes to prevent duplicate processing
        2. Checks for cycles in the graph (should not occur in a DAG)
        3. Processes nodes in depth-first order
        4. Records the traversal order

        Args:
            node: Current node being processed
        """
        # Skip if node has already been visited
        if node in self.visited:
            return

        # Check for cycles (should not happen in a valid DAG)
        if node in self.path:
            return

        # Process current node
        self.path.add(node)  # Mark as part of current path
        self.visited.add(node)  # Mark as visited
        self.traversal_order.append(node)  # Add to traversal order

        # Recursively process all children
        for child in self.graph.get(node, []):
            self._dfs_traverse(child)

        # Remove node from current path when done
        self.path.remove(node)

    def sub_traversal(self, objects_to_replace):
        """
        Perform sub-traversal optimization.

        This is an optimized traversal that:
        1. Visits all nodes in the graph
        2. Only includes selected objects in the traversal order
        3. Maintains the same parent-child visitation order as the base traversal

        Args:
            objects_to_replace (set): Object types to replace

        Returns:
            list: Nodes in traversal order (only those to be replaced)
        """
        # Step 1: Find root nodes (nodes with no parents)
        all_nodes = set(self.graph.keys()).union(*[set(children) for children in self.graph.values()])
        child_nodes = set()
        for children in self.graph.values():
            child_nodes.update(children)
        root_nodes = all_nodes - child_nodes

        # Step 2: Start subtraversal from each root node
        for root in root_nodes:
            self._subtraversal_dfs(root, objects_to_replace)

        return self.traversal_order

    def _subtraversal_dfs(self, node, objects_to_replace):
        """
        Recursively traverse the graph for subtraversal.

        Similar to _dfs_traverse, but only adds nodes to the traversal_order
        if they are in the objects_to_replace set. This allows selective
        regeneration of objects while maintaining the overall traversal structure.

        Args:
            node: Current node
            objects_to_replace (set): Object types to replace
        """
        # Skip if node has already been visited
        if node in self.visited:
            return

        # Mark node as visited
        self.visited.add(node)

        # Only add to traversal order if this object should be replaced
        if node in objects_to_replace:
            self.traversal_order.append(node)

        # Always process all children to maintain traversal structure
        for child in self.graph.get(node, []):
            self._subtraversal_dfs(child, objects_to_replace)

    def _add_all_descendants(self, obj_type, result_set):
        """
        Add all descendants of an object type to the result set recursively.

        Args:
            obj_type (str): Object type to find descendants for
            result_set (set): Set to add descendants to
        """
        for child in self.graph.get(obj_type, []):
            result_set.add(child)
            self._add_all_descendants(child, result_set)

    def _get_topological_sort(self, obj_types):
        """
        Get a topologically sorted list of object types.
        This ensures parents are processed before their children.

        Args:
            obj_types (set): Set of object types to sort

        Returns:
            list: Topologically sorted list of object types (parents before children)
        """
        # Create a graph representation where edges go from parents to children
        # This is the opposite of what we want for topological sort
        child_to_parent = {}
        for obj_type in obj_types:
            child_to_parent[obj_type] = []
            # Find all parents of this object type
            for parent, children in self.graph.items():
                if obj_type in children and parent in obj_types:
                    child_to_parent[obj_type].append(parent)

        # Identify roots (objects with no parents in our set)
        roots = []
        for obj_type in obj_types:
            if not child_to_parent[obj_type]:
                roots.append(obj_type)

        # Always process Floor first if it's a root
        if "Floor" in roots:
            roots.remove("Floor")
            roots.insert(0, "Floor")

        # Create the sorted list using Kahn's algorithm for topological sorting
        sorted_objects = []
        while roots:
            # Remove a root and add to result
            node = roots.pop(0)
            sorted_objects.append(node)

            # For each child of the current node
            for child in self.graph.get(node, []):
                if child in obj_types:
                    # Remove the edge from child to parent
                    child_to_parent[child].remove(node)

                    # If child has no more parents, it's a new root
                    if not child_to_parent[child]:
                        roots.append(child)

        # Reverse the order so parents come before children
        # because our graph is defined with parent->child edges
        return sorted_objects
