# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Scene traversal and generation module for procedural scene generation.

This module contains the core scene traversal algorithm that constructs scenes by:
1. Processing objects in topological order (parents before children)
2. Determining valid positions for each object type
3. Randomly selecting and placing objects at valid positions
4. Implementing subtraversal optimizations for efficient scene generation

The traversal approach creates a tree of possible object arrangements, exploring
different combinations to generate diverse and valid scenes efficiently.
"""

import copy
import csv
import matplotlib.pyplot as plt
import os
import random
import time

import networkx as nx

from scripts.procedural_scene_gen_standalone.PSG_old.placement import (
    ObjectPlacement,
    ParentPlacement,
    calculate_orientation,
    calculate_scale_factor,
    create_floor,
    create_object,
    get_random_valid_position,
)
from scripts.procedural_scene_gen_standalone.PSG_old.utils import Logger, get_random_asset

# Global dictionary to store valid positions for subtraversals
# This cache persists across scene generations to avoid recalculating positions
# Format: {obj_type: {parent_id: [positions]}}
STORED_VALID_POSITIONS = {}

# Global cache for scale factors and true sizes
# Optimization to avoid expensive recalculations
# Format: {obj_type: {asset_path: (scale_factor, true_size)}}
SCALE_FACTOR_CACHE = {}


class SceneTraversal:
    """
    Handles scene generation by traversing a parent-child graph.
    This class manages the process of creating scenes by:
    1. Processing objects in topological order (parents before children)
    2. Managing object placement on the floor using a grid system
    3. Tracking object relationships and dependencies
    4. Logging detailed debug information about the scene creation process

    The scene generation process ensures that:
    - Objects are placed in valid positions (no overlaps)
    - Parent-child relationships are respected
    - Clearance requirements are met
    - All objects are properly positioned relative to their parents
    """

    def __init__(self, graph, config):
        """
        Initialize scene traversal with graph and configuration.

        Args:
            graph (dict): Adjacency list representing parent-child relationships
                        between objects in the scene
            config (dict): Scene configuration from YAML containing:
                         - Object definitions (size, clearance, etc.)
                         - Generation parameters (number of scenes)
                         - Output settings (save directory, debug options)
        """
        self.graph = graph
        self.config = config
        self.nx_graph = nx.DiGraph(graph)  # Convert to NetworkX graph for easier traversal
        self.logger = Logger(config)  # Initialize logger for debug logging
        self.parent_placement = ParentPlacement(config)  # Initialize parent placement handler

        # Set up output directory and file paths
        self.save_dir = os.path.join(config["output"]["save_dir"], config["output"]["run_name"])
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize log files
        self.debug_log_file = os.path.join(self.save_dir, config["debug"]["log_debug_file"])

        # Create time log file with headers if timing is enabled
        if self.config["debug"]["log_timing"]:
            self.time_log_file = os.path.join(self.save_dir, config["debug"]["log_timing_file"])
            with open(self.time_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Scene", "Objects", "Time (s)"])

        # Pre-calculate scale factors for objects that will be respawned
        self._precalculate_scale_factors()

    def _precalculate_scale_factors(self):
        """
        Pre-calculate scale factors for all objects that might be respawned.
        This avoids expensive scale factor calculations during subtraversals.

        Scale factors determine how to resize 3D models to match desired dimensions.
        By caching these values, we avoid recalculating them for each scene generation,
        significantly improving performance for multi-scene generation.
        """
        global SCALE_FACTOR_CACHE
        SCALE_FACTOR_CACHE.clear()  # Clear any existing cache

        # Determine objects that might be respawned based on configuration
        respawn_candidates = set()

        # Add objects from selective subtraversal config if enabled
        if self.config["generation"]["subtraversal"]["selective"]:
            # Use explicit list of objects to respawn from config
            respawn_candidates.update(self.config["generation"]["subtraversal"]["objects"])
        else:
            # Otherwise use height-based selection (objects at specific graph level)
            height = self.config["generation"]["subtraversal"]["height"]
            respawn_candidates.update(self._get_objects_at_height(height))

        # We pre-calculate for all candidates since we don't know which will be chosen

        # Initialize cache for each object type
        for obj_type in respawn_candidates:
            # Skip objects not defined in scene config
            if obj_type not in self.config["scene"]:
                continue

            SCALE_FACTOR_CACHE[obj_type] = {}

            # Get asset directory for this object type
            asset_dir = os.path.join(self.config["output"]["asset_dir"], obj_type)
            if not os.path.exists(asset_dir):
                continue

            # Get all USD model files for this object type
            asset_files = [f for f in os.listdir(asset_dir) if f.endswith(".usd")]
            for asset_file in asset_files:
                asset_path = os.path.join(asset_dir, asset_file)

                # Calculate and cache scale factor and true size for each asset
                scale_factor, true_size = calculate_scale_factor(asset_path, self.config["scene"][obj_type]["size"])
                SCALE_FACTOR_CACHE[obj_type][asset_path] = (scale_factor, true_size)

    def generate_scenes(self):
        """
        Generate multiple scenes based on configuration.
        Each scene is created by traversing the parent-child graph in topological order,
        ensuring objects are placed in the correct sequence.

        The generation process uses an optimization strategy:
        1. Generate a base scene using full traversal
        2. For subsequent scenes, use subtraversal to regenerate only parts of the scene

        Returns:
            list: List of scene data dictionaries, each containing:
                 - Metadata (run name, scene number, timestamp)
                 - Scene objects with their positions and properties
        """
        # Clear stored positions when starting a new generation run
        global STORED_VALID_POSITIONS
        STORED_VALID_POSITIONS.clear()

        scenes = []

        # First generate the base scene (scene 0)
        self.logger.debug(f"\n{'='*50}")
        self.logger.debug(f"Generating Base Scene")
        self.logger.debug(f"{'='*50}\n")

        # Create the initial base scene with full traversal
        base_scene = self._create_scene(0)
        scenes.append(base_scene)

        # If only one scene requested, return early
        if self.config["generation"]["num_scenes"] <= 1:
            return scenes

        # For multiple scenes, use subtraversal optimization
        self.logger.debug(f"\n{'='*50}")
        self.logger.debug(f"Using Subtraversal for Remaining Scenes")
        self.logger.debug(f"{'='*50}\n")

        # Generate the remaining scenes using subtraversal optimization
        remaining_scenes = self.config["generation"]["num_scenes"] - 1
        subtraversal_scenes = self._generate_subtraversals(base_scene, remaining_scenes)
        scenes.extend(subtraversal_scenes)

        return scenes

    def _determine_objects_to_respawn(self):
        """
        Determine which objects will be respawned in subtraversals based on configuration.

        The subtraversal approach has two modes:
        1. Selective: Respawn specific objects listed in configuration
        2. Height-based: Respawn objects at a specific height in the graph

        Additionally, it can operate in two modes:
        - Forced: Respawn all selected objects
        - Non-forced: Randomly select one object from the candidates

        Returns:
            list: List of object types to respawn in the subtraversal
        """
        objects_to_respawn = []

        # Use selective respawn (specific objects listed in config)
        if self.config["generation"]["subtraversal"]["selective"]:
            objects_to_respawn = self.config["generation"]["subtraversal"]["objects"]
            self.logger.debug(f"Using selective respawn for objects: {objects_to_respawn}")
        else:
            # Use height-based respawn (objects at specific level in graph)
            height = self.config["generation"]["subtraversal"]["height"]
            objects_to_respawn = self._get_objects_at_height(height)
            self.logger.debug(f"Using height-based respawn (height {height}) for objects: {objects_to_respawn}")

        # If forced is false, randomly choose just one object to respawn
        if not self.config["generation"]["subtraversal"]["forced"] and objects_to_respawn:
            objects_to_respawn = [random.choice(objects_to_respawn)]
            self.logger.debug(f"Non-forced mode: randomly selected {objects_to_respawn[0]} for respawn")

        return objects_to_respawn

    def _get_objects_at_height(self, height):
        """
        Get objects at a specific height in the graph.
        Height 0 is Floor, height 1 is children of Floor, etc.

        This function:
        1. Calculates the height of each node in the graph
        2. Returns the nodes that match the target height

        The height represents the longest path from a root node,
        which allows targeting specific "layers" of objects
        (e.g., furniture vs. items on furniture).

        Args:
            height (int): Target height in the graph

        Returns:
            list: List of object types at the specified height
        """
        # Compute node heights using dynamic programming
        node_heights = {}

        def compute_height(node):
            """
            Recursive helper function to compute node height.

            Height is defined as the longest path from a root node:
            - Root nodes (no parents) have height 0
            - Other nodes have height 1 + max(parent_heights)

            Args:
                node (str): Node to compute height for

            Returns:
                int: Height of the node
            """
            # Return cached height if already computed
            if node in node_heights:
                return node_heights[node]

            # Find all parents of this node
            parents = []
            for possible_parent in self.nx_graph.nodes():
                if node in self.graph.get(possible_parent, []):
                    parents.append(possible_parent)

            # If node has no parents (root node), height is 0
            if not parents:
                node_heights[node] = 0
                return 0

            # Height is 1 + max parent height
            parent_heights = [compute_height(parent) for parent in parents]
            node_heights[node] = 1 + max(parent_heights)
            return node_heights[node]

        # Compute height for all nodes
        for node in self.nx_graph.nodes():
            if node not in node_heights:
                compute_height(node)

        # Return nodes at target height
        return [node for node, h in node_heights.items() if h == height]

    def _generate_subtraversals(self, base_scene, num_scenes):
        """
        Generate multiple scenes by respawning only certain objects.
        This is a performance optimization that avoids full graph traversal.

        The subtraversal approach:
        1. Identifies which objects to respawn
        2. Creates a template scene keeping fixed objects from the base scene
        3. Efficiently places only the respawned objects in new positions
        4. Uses cached valid positions to avoid recalculations

        This provides significant performance benefits for generating
        multiple scenes with variations of the same basic layout.

        Args:
            base_scene (dict): Base scene to use as template
            num_scenes (int): Number of additional scenes to generate

        Returns:
            list: List of scene data dictionaries
        """
        self.logger.debug(f"Generating {num_scenes} scenes using subtraversal")

        # Determine which objects to respawn based on configuration
        objects_to_respawn = self._determine_objects_to_respawn()
        if not objects_to_respawn:
            self.logger.debug("No objects selected for respawn - falling back to full traversal")
            return [self._create_scene(i + 1) for i in range(num_scenes)]

        self.logger.debug(f"Objects selected for respawn: {objects_to_respawn}")

        # Start timing - ONLY measure scene generation time
        generation_start_time = time.time()

        # ULTRA-OPTIMIZED APPROACH: Create a single template scene
        # This template will have fixed objects copied from base scene
        # and empty lists for objects that will be respawned
        template_scene = {
            "metadata": {
                "run_name": self.config["output"]["run_name"],
                "scene_number": 0,  # Will be updated for each scene
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "scene": {},
        }

        # Copy only the fixed objects (non-respawned) from base scene
        for obj_name in self.config["scene"]:
            if obj_name in objects_to_respawn:
                # Initialize empty lists for objects that will be respawned
                template_scene["scene"][obj_name] = []
            else:
                # Deep copy fixed objects from base scene
                template_scene["scene"][obj_name] = copy.deepcopy(base_scene["scene"][obj_name])

        # Pre-process parent data for quick access
        # This maps parent IDs to their data for efficient lookup
        parent_data_map = {}
        for obj_name in objects_to_respawn:
            if obj_name in base_scene["scene"]:
                for instance in base_scene["scene"][obj_name]:
                    if "parent" in instance and "parent_idx" in instance:
                        parent_type = instance["parent"]
                        parent_idx = instance["parent_idx"]
                        parent_id = f"{parent_type}_{parent_idx}"

                        # Store parent data for reference when respawning objects
                        if parent_id not in parent_data_map and parent_type in base_scene["scene"]:
                            if parent_idx < len(base_scene["scene"][parent_type]):
                                parent_data_map[parent_id] = base_scene["scene"][parent_type][parent_idx]

        # Pre-cache placement type for each object for quick access
        placement_types = {}
        for obj_name in objects_to_respawn:
            placement_types[obj_name] = self.config["scene"][obj_name].get("placement", "")

        # Create array of scenes upfront (more efficient than appending)
        batch_scenes = []

        # Create a copy of valid positions for each scene
        # Format: {scene_idx: {obj_type: {parent_id: [positions]}}}
        # Each scene needs its own copy because positions are consumed when used
        scene_valid_positions = []
        for _ in range(num_scenes):
            scene_positions = {}
            for obj_name in objects_to_respawn:
                if obj_name in STORED_VALID_POSITIONS:
                    scene_positions[obj_name] = {}
                    for parent_id, positions in STORED_VALID_POSITIONS[obj_name].items():
                        # Create deep copy of positions to avoid shared state
                        scene_positions[obj_name][parent_id] = positions.copy()
            scene_valid_positions.append(scene_positions)

        # Define fast object creation function for efficient respawning
        def fast_create_object(obj_name):
            """
            Optimized helper function to create object instances quickly.

            This function:
            1. Gets object config and selects a random asset
            2. Uses cached scale factors when available
            3. Returns object data without position (added later)

            Args:
                obj_name (str): Type of object to create

            Returns:
                dict: Object data without position
            """
            obj_config = self.config["scene"][obj_name]
            asset_path = get_random_asset(self.config, obj_name)

            # Get base rotation from config
            rotation = obj_config["rotation"].copy()

            # Use pre-calculated scale factor and true size from cache if available
            if obj_name in SCALE_FACTOR_CACHE and asset_path in SCALE_FACTOR_CACHE[obj_name]:
                scale_factor, true_size = SCALE_FACTOR_CACHE[obj_name][asset_path]
            else:
                # Fall back to calculation if not cached (should rarely happen)
                scale_factor, true_size = calculate_scale_factor(asset_path, obj_config["size"])

            return {
                "position": [0, 0, 0],  # Placeholder, will be updated later
                "size": true_size,
                "rotation": rotation,
                "asset_path": asset_path,
                "scale_factor": scale_factor,
            }

        # Generate all scenes in parallel
        for scene_idx in range(num_scenes):
            # Create a new scene from template (deep copy to avoid shared state)
            scene = copy.deepcopy(template_scene)
            scene["metadata"]["scene_number"] = scene_idx + 1

            # Process each object type to respawn
            for obj_name in objects_to_respawn:
                if obj_name not in base_scene["scene"]:
                    continue

                # Get configuration for this object type
                obj_config = self.config["scene"][obj_name]
                strict_orientation = obj_config.get("strict_orientation", False)
                placement_type = placement_types[obj_name]
                instances = base_scene["scene"][obj_name]

                # Process each instance of this object type
                for instance_idx, instance in enumerate(instances):
                    # Skip if no parent information
                    if "parent" not in instance or "parent_idx" not in instance:
                        continue

                    # Get parent information
                    parent_type = instance["parent"]
                    parent_idx = instance["parent_idx"]
                    parent_id = f"{parent_type}_{parent_idx}"

                    # Skip if no parent data or valid positions available
                    if parent_id not in parent_data_map:
                        continue

                    if (
                        obj_name not in scene_valid_positions[scene_idx]
                        or parent_id not in scene_valid_positions[scene_idx][obj_name]
                        or not scene_valid_positions[scene_idx][obj_name][parent_id]
                    ):
                        continue

                    # Get valid positions for this object relative to its parent
                    positions = scene_valid_positions[scene_idx][obj_name][parent_id]
                    if not positions:
                        continue

                    # Randomly select a position and remove it to prevent reuse
                    position_idx = random.randrange(len(positions))
                    position = positions[position_idx]
                    positions.pop(position_idx)  # Remove used position

                    # Create object quickly using the optimized function
                    obj_data = fast_create_object(obj_name)
                    obj_data["position"] = position.copy()
                    obj_data["parent"] = parent_type
                    obj_data["parent_idx"] = parent_idx

                    # Handle orientation for objects that need specific facing
                    if (
                        strict_orientation
                        or (isinstance(placement_type, str) and placement_type == "Tuck")
                        or (isinstance(placement_type, list) and "Tuck" in placement_type)
                    ):
                        parent_data = parent_data_map[parent_id]
                        current_placement = placement_type[0] if isinstance(placement_type, list) else placement_type
                        obj_data["rotation"] = calculate_orientation(
                            position, parent_data["position"], parent_data["size"], current_placement
                        )

                    # Add completed object to scene
                    scene["scene"][obj_name].append(obj_data)

            batch_scenes.append(scene)

        # End timing - measure only scene generation time
        generation_end_time = time.time()
        generation_time = generation_end_time - generation_start_time

        # Apply garbage collection to free memory
        import gc

        gc.collect()

        # Log timing information if enabled
        if self.config["debug"]["log_timing"]:
            with open(self.time_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                # Get a representative object count (all scenes should have similar counts)
                first_scene = batch_scenes[0] if batch_scenes else None
                if first_scene:
                    total_objects = sum(
                        len(obj_list) for obj_name, obj_list in first_scene["scene"].items() if obj_name != "Floor"
                    )
                    # Log a single summary line for all scenes 2-N
                    start_scene = 2  # Base scene is 1
                    end_scene = start_scene + num_scenes - 1  # Last scene number
                    writer.writerow([f"{start_scene}-{end_scene}", total_objects, f"{generation_time:.10f}"])

        self.logger.debug(f"Subtraversal generated {num_scenes} scenes in {generation_time:.6f} seconds")
        self.logger.debug(f"Average time per scene: {generation_time/num_scenes:.6f} seconds")

        return batch_scenes

    def _check_overlap(self, pos1, size1, pos2, size2, clearance=0):
        """
        Check if two objects overlap, considering clearance.

        This function:
        1. Calculates the dimensions of each object including clearance
        2. Checks overlap in each dimension (x, y, z)
        3. Returns True if objects overlap in all dimensions

        Objects are treated as axis-aligned bounding boxes (AABBs).

        Args:
            pos1 (list): [x, y, z] position of first object
            size1 (list): [x, y, z] size of first object
            pos2 (list): [x, y, z] position of second object
            size2 (list): [x, y, z] size of second object
            clearance (float): Minimum distance between objects

        Returns:
            bool: True if objects overlap, False otherwise
        """
        # Check overlap in each dimension (x, y, z)
        for i in range(3):
            half_size1 = size1[i] / 2
            half_size2 = size2[i] / 2

            # Calculate distance between centers
            center_distance = abs(pos1[i] - pos2[i])

            # Calculate minimum distance needed to avoid overlap
            min_distance = half_size1 + half_size2 + clearance

            # If distance is less than minimum in any dimension, objects overlap
            if center_distance < min_distance:
                return True

        # No overlap in at least one dimension means objects don't overlap
        return False

    def _place_child_objects(self, node, obj_config, scene_data, placement_handler):
        """
        Place child objects relative to their parent objects.

        This function:
        1. Identifies parent objects for this child type
        2. Collects valid positions around all parent instances
        3. Randomly selects positions for each child instance
        4. Handles orientation relative to the parent
        5. Adds successfully placed objects to the scene

        The placement respects clearance requirements and ensures
        objects are correctly oriented relative to their parents.

        Args:
            node (str): Type of child object to place
            obj_config (dict): Configuration for the child object
            scene_data (dict): Current scene data
            placement_handler (ObjectPlacement): Handler for object placement

        Returns:
            int: Number of instances successfully placed
        """
        parent_name = obj_config["parents"][0]  # Get first parent type
        parent_instances = scene_data["scene"][parent_name]
        placement_type = obj_config.get("placement", "")

        # Validate parent instances and placement type
        if not parent_instances:
            self.logger.debug(f"WARNING: No parent instances found for {node}")
            return 0

        if not placement_type:
            self.logger.debug(f"WARNING: No placement type specified for {node}")
            return 0

        # Step 1: Collect all valid positions from all parent instances
        all_valid_positions = []
        self.logger.debug(f"\nCollecting valid positions for {node} with placement type {placement_type}:")

        for parent_idx, parent_data in enumerate(parent_instances):
            self.logger.debug(f"\nChecking parent {parent_name} instance {parent_idx}")
            self.logger.debug(f"Parent position: {parent_data['position']}")

            # Get valid positions from parent using parent placement handler
            positions = self.parent_placement.get_valid_positions(
                node, parent_data, placement_type, obj_config.get("clearance", 0), parent_type=parent_name
            )

            if positions:
                self.logger.debug(f"Found {len(positions)} valid positions:")
                for pos in positions:
                    self.logger.debug(f"  - {pos}")
                # Store positions with their parent index for reference
                all_valid_positions.extend([(pos, parent_idx) for pos in positions])

                # Store positions for subtraversal optimization
                parent_id = f"{parent_name}_{parent_idx}"
                if node not in STORED_VALID_POSITIONS:
                    STORED_VALID_POSITIONS[node] = {}
                STORED_VALID_POSITIONS[node][parent_id] = positions
            else:
                self.logger.debug("No valid positions found for this parent")

        self.logger.debug(f"\nTotal valid positions found: {len(all_valid_positions)}")

        # Step 2: Try to place each instance using available positions
        if not all_valid_positions:
            self.logger.debug(f"WARNING: No valid positions found for any parent for {node}")
            return 0

        # Make a copy of available positions to avoid modifying the original
        remaining_positions = all_valid_positions.copy()
        instances_placed = 0  # Track how many instances we've placed
        target_instances = obj_config["instances"]  # Total instances to place

        # Track which positions were actually used (for cleanup)
        used_positions = []

        # Try to place instances until we hit our target or run out of positions
        while instances_placed < target_instances and remaining_positions:
            # Randomly choose one of the remaining valid positions
            chosen_idx = random.randrange(len(remaining_positions))
            pos, parent_idx = remaining_positions.pop(chosen_idx)
            parent_id = f"{parent_name}_{parent_idx}"

            self.logger.debug(f"Placing {node} instance {instances_placed} at position {pos}")
            self.logger.debug(f"Remaining positions: {len(remaining_positions)}")

            # Create new object data for this instance
            instance_data = create_object(self.config, node)
            instance_data["position"] = pos
            instance_data["parent"] = parent_name  # Store parent name for subtraversal
            instance_data["parent_idx"] = parent_idx  # Store parent index for subtraversal

            # Handle orientation based on placement type and configuration
            current_placement = placement_type[0] if isinstance(placement_type, list) else placement_type
            strict_orientation = obj_config.get("strict_orientation", False) or current_placement == "Tuck"

            if strict_orientation:
                parent_data = parent_instances[parent_idx]
                # Calculate orientation based on position relative to parent
                instance_data["rotation"] = calculate_orientation(
                    pos, parent_data["position"], parent_data["size"], current_placement
                )
                self.logger.debug(f"Applied strict orientation: {instance_data['rotation']} for position {pos}")

            # Add object to placement handler - use true size from instance_data, not config size
            self.logger.debug(f"Using true size {instance_data['size']} for placement check")
            success = placement_handler.add_object(
                pos,
                instance_data["size"],  # Use the true size after scaling, not the config size
                f"{node}_{instances_placed}",
                obj_config.get("clearance", 0),
                current_placement,
            )
            if success:
                # Add successfully placed object to scene
                scene_data["scene"][node].append(instance_data)
                used_positions.append(pos)
                self.logger.debug(f"Successfully placed {node} instance {instances_placed} at {pos}")
                instances_placed += 1
            else:
                self.logger.debug(f"Failed to add object to placement handler")

        # Log warning if we couldn't place all requested instances
        if instances_placed < target_instances:
            self.logger.debug(
                f"WARNING: Could only place {instances_placed} instances of {node} (wanted {target_instances})"
            )

        # Clear unused positions from placement handler
        self.logger.debug(f"\nClearing unused valid positions for {node}")
        placement_handler.clear_unused_positions(used_positions)

        return instances_placed

    def _create_scene(self, scene_number):
        """
        Create a single scene by traversing the graph in topological order.
        This ensures that parent objects are placed before their children.

        The process:
        1. Initialize scene data structure
        2. Create placement handler for tracking object positions
        3. Process each node in topological order
        4. For each object:
           - Create it at origin
           - If it has Floor as parent, find valid position
           - Mark position as occupied in grid
           - Add to scene data

        Args:
            scene_number (int): Scene identifier (0-based)

        Returns:
            dict: Complete scene data containing:
                 - Metadata (run name, scene number, timestamp)
                 - Scene objects with their positions and properties
        """
        start_time = time.time()

        # Initialize scene data structure
        scene_data = {
            "metadata": {
                "run_name": self.config["output"]["run_name"],
                "scene_number": scene_number,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "scene": {obj_name: [] for obj_name in self.config["scene"]},
        }

        # Create placement handler for tracking object positions
        floor_size = self.config["scene"]["Floor"]["size"]
        placement_handler = ObjectPlacement(floor_size)
        self.logger.debug(f"Created placement handler with floor size: {floor_size}")

        # Process objects in topological order (parents before children)
        for node in nx.topological_sort(self.nx_graph):
            obj_config = self.config["scene"][node]
            self.logger.debug(f"\nProcessing node: {node}")
            self.logger.debug(f"Configuration: {obj_config}")

            # Special case for floor object (always placed at origin)
            if node == "Floor":
                self.logger.debug("Creating floor object")
                scene_data["scene"][node].append(create_floor(self.config))
                continue

            # Handle floor-based objects differently from child objects
            if "Floor" in obj_config["parents"]:
                # For floor-based objects, we try each instance sequentially
                for instance in range(obj_config["instances"]):
                    self.logger.debug(f"\nCreating instance {instance + 1} of {node}")

                    # Floor-based placement using exact position checking
                    clearance = obj_config.get("clearance", 0)
                    self.logger.debug(f"Finding valid position with clearance: {clearance}")

                    # Get placement type, handling if it's a list
                    placement_type = obj_config.get("placement", "")
                    if isinstance(placement_type, list) and placement_type:
                        placement_type = placement_type[0]

                    # Find valid position on floor
                    pos = get_random_valid_position(
                        floor_size, obj_config["size"], placement_handler, clearance, placement_type=placement_type
                    )
                    if pos is None:
                        self.logger.debug(f"WARNING: Could not find valid position for {node} instance {instance}")
                        continue

                    self.logger.debug(f"Found valid position: {pos}")
                    obj_data = create_object(self.config, node)
                    obj_data["position"] = pos
                    obj_data["parent"] = "Floor"  # Store parent for subtraversal
                    obj_data["parent_idx"] = 0  # Floor is always instance 0

                    # Set random 90-degree orientation if strict_orientation is enabled for floor-based objects
                    strict_orientation = obj_config.get("strict_orientation", False)
                    if strict_orientation:
                        # For floor-based objects, use random 90-degree orientation
                        random_rotation = random.choice([0, 90, 180, 270])
                        obj_data["rotation"] = [0, 0, random_rotation]
                        self.logger.debug(f"Applied random strict orientation: {obj_data['rotation']}")

                    # Add object to placement handler to track occupied space
                    success = placement_handler.add_object(
                        pos, obj_config["size"], f"{node}_{instance}", clearance, placement_type
                    )
                    if success:
                        scene_data["scene"][node].append(obj_data)
                        self.logger.debug(f"Added {node} instance {instance + 1} to scene")
                    else:
                        self.logger.debug(f"WARNING: Failed to add object to placement handler")
            else:
                # For child objects, place all instances at once relative to parents
                placed_count = self._place_child_objects(node, obj_config, scene_data, placement_handler)
                self.logger.debug(f"Placed {placed_count} instances of {node}")

        # Calculate scene statistics
        total_objects = sum(len(obj_list) for obj_name, obj_list in scene_data["scene"].items() if obj_name != "Floor")
        time_taken = time.time() - start_time

        # Log timing data if enabled
        if self.config["debug"]["log_timing"]:
            with open(self.time_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([scene_number + 1, total_objects, f"{time_taken:.10f}"])

        # Log scene completion summary
        self.logger.debug(f"\nScene {scene_number + 1} completed:")
        self.logger.debug(f"Total objects: {total_objects}")
        self.logger.debug(f"Time taken: {time_taken:.3f} seconds")

        return scene_data
