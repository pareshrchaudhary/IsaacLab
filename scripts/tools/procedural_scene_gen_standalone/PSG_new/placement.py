# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Object placement module for procedural scene generation.

This module provides functionality for:
1. Creating and placing objects in the scene
2. Determining valid object positions based on parent-child relationships
3. Ensuring objects respect physical constraints (no overlapping)
4. Managing different placement strategies (Top, Under, Tuck, Long, Short)
5. Handling object orientation based on placement type

The placement module contains main classes:
- ClearanceCalculator: Determines proper clearance distances between objects
- PlacementManager: Implements placement strategies and position calculations
- ObjectManager: Handles object creation, scaling, and bounding box calculations
- PlacementGrid: Manages a grid-based system for tracking object placement
"""

import numpy as np
import os
import random

from pxr import Gf, Usd, UsdGeom


class ClearanceCalculator:
    """
    Calculates clearance values between parent and child objects.

    This class determines the appropriate distance to maintain between objects
    when they are placed in relation to each other. It handles:
    1. General clearance values defined for object types
    2. Specific clearance values for particular parent-child pairs
    3. Resolution of clearance conflicts by using the maximum value
    4. Clearance between non-parent-child object pairs

    The clearance values ensure that objects are placed at appropriate
    distances from each other, creating physically plausible scenes.
    """

    def __init__(self, config):
        """
        Initialize the clearance calculator with configuration.

        Args:
            config (dict): Scene configuration containing object clearance values
        """
        self.config = config

    def calculate_parent_child_clearance(self, child_type, parent_type):
        """
        Calculate the appropriate clearance value between parent and child objects.

        This function applies the following precedence rules:
        1. Specific clearance values defined in child-to-parent relationship
        2. Specific clearance values defined in parent-to-child relationship
        3. Child's general clearance value as a fallback

        When multiple clearance values are applicable, the largest value is used
        to ensure sufficient spacing between objects.

        Args:
            child_type (str): Type of the child object
            parent_type (str): Type of the parent object

        Returns:
            float: The appropriate clearance value to use for positioning
        """
        # Get child and parent configurations
        child_config = self.config["scene"].get(child_type, {})
        parent_config = self.config["scene"].get(parent_type, {})

        # Get general clearance value for the child object (default fallback)
        child_general_clearance = child_config.get("general_clearance", 0)

        # Get specific clearance dictionaries
        child_specific_clearances = child_config.get("specific_clearance", {})
        parent_specific_clearances = parent_config.get("specific_clearance", {})

        # Look for specific clearance values in both directions
        child_to_parent_clearance = child_specific_clearances.get(parent_type)
        parent_to_child_clearance = parent_specific_clearances.get(child_type)

        # Collect all applicable clearance values
        clearance_values = []

        # Add specific clearance values if they exist
        if child_to_parent_clearance is not None:
            clearance_values.append(child_to_parent_clearance)

        if parent_to_child_clearance is not None:
            clearance_values.append(parent_to_child_clearance)

        # If no specific clearances found, use the general clearance
        if not clearance_values:
            clearance_values.append(child_general_clearance)

        # Use the maximum clearance value to ensure sufficient spacing
        return max(clearance_values)

    def calculate_non_parent_child_clearance(self, obj_type1, obj_type2):
        """
        Calculate the appropriate clearance value between two unrelated objects.

        For objects with no parent-child relationship, this function determines
        the minimum distance between them based on the following rules:
        1. If both objects have specific clearance defined for each other, use the maximum
        2. If only one object has specific clearance defined for the other, use that value
        3. If no specific clearance exists, use the higher of the two general clearances

        Args:
            obj_type1 (str): Type of the first object
            obj_type2 (str): Type of the second object

        Returns:
            float: The minimum distance to maintain between the objects
        """
        # Get object configurations
        obj1_config = self.config["scene"].get(obj_type1, {})
        obj2_config = self.config["scene"].get(obj_type2, {})

        # Get general clearance values
        obj1_general_clearance = obj1_config.get("general_clearance", 0)
        obj2_general_clearance = obj2_config.get("general_clearance", 0)

        # Get specific clearance dictionaries
        obj1_specific_clearances = obj1_config.get("specific_clearance", {})
        obj2_specific_clearances = obj2_config.get("specific_clearance", {})

        # Look for specific clearance values in both directions
        obj1_to_obj2_clearance = obj1_specific_clearances.get(obj_type2)
        obj2_to_obj1_clearance = obj2_specific_clearances.get(obj_type1)

        # Case 1: Both objects have specific clearance for each other
        if obj1_to_obj2_clearance is not None and obj2_to_obj1_clearance is not None:
            return max(obj1_to_obj2_clearance, obj2_to_obj1_clearance)

        # Case 2: Only one object has specific clearance for the other
        if obj1_to_obj2_clearance is not None:
            return obj1_to_obj2_clearance

        if obj2_to_obj1_clearance is not None:
            return obj2_to_obj1_clearance

        # Case 3: Fall back to the higher of the general clearances
        return max(obj1_general_clearance, obj2_general_clearance)


class FilterPositions:
    """
    Filters valid positions based on overlap detection and clearance requirements.

    This class is responsible for:
    1. Checking if object positions would cause overlaps with existing objects
    2. Filtering out invalid positions that would result in collisions
    3. Ensuring minimum clearance requirements are maintained
    4. Handling rotated objects for accurate overlap detection
    5. Optimizing the filtering process for performance

    The filtering process ensures that objects are placed in physically
    plausible positions without overlapping with other objects.
    """

    def __init__(self, config):
        """
        Initialize the position filter with configuration.

        Args:
            config (dict): Scene configuration containing clearance settings
        """
        self.config = config
        self.clearance_calculator = ClearanceCalculator(config)
        self.object_manager = ObjectManager(config)

        # If we implement an octree, it would be initialized here
        self.octree = None

    def filter_positions(self, obj_type, valid_positions, scene_data, parent_obj_type=None, parent_data=None):
        """
        Filter valid positions to ensure no overlaps with existing objects and among themselves.

        Args:
            obj_type (str): Type of object being placed
            valid_positions (list): List of valid positions to filter
            scene_data (dict): Current scene data
            parent_obj_type (str, optional): Type of parent object
            parent_data (dict, optional): Data of parent object

        Returns:
            list: Filtered list of valid positions
        """
        print(f"[FILTER] {obj_type}: Initial positions: {len(valid_positions)}")

        if not valid_positions:
            print(f"[FILTER] {obj_type}: No valid positions to filter")
            return []

        # Get object size from config
        obj_size = self.config["scene"][obj_type]["size"]

        # STEP 1: Filter positions that overlap with existing scene objects
        positions_without_scene_overlap = []

        for position in valid_positions:
            overlaps_with_scene = False

            # Check against every object in the scene except Floor
            for scene_obj_type, scene_objects in scene_data["scene"].items():
                # Skip Floor since everything is placed on it
                if scene_obj_type == "Floor":
                    continue

                # Check against every instance of this object type
                for scene_obj in scene_objects:
                    # Skip if this is the parent object
                    if parent_data and scene_obj == parent_data:
                        continue

                    # Perform explicit 2D overlap check
                    pos_min_x = position[0] - obj_size[0] / 2
                    pos_max_x = position[0] + obj_size[0] / 2
                    pos_min_y = position[1] - obj_size[1] / 2
                    pos_max_y = position[1] + obj_size[1] / 2

                    obj_min_x = scene_obj["position"][0] - scene_obj["size"][0] / 2
                    obj_max_x = scene_obj["position"][0] + scene_obj["size"][0] / 2
                    obj_min_y = scene_obj["position"][1] - scene_obj["size"][1] / 2
                    obj_max_y = scene_obj["position"][1] + scene_obj["size"][1] / 2

                    # Using strict inequality and a small buffer to avoid floating point issues
                    buffer = 0.01
                    if not (
                        pos_max_x + buffer < obj_min_x
                        or pos_min_x - buffer > obj_max_x
                        or pos_max_y + buffer < obj_min_y
                        or pos_min_y - buffer > obj_max_y
                    ):
                        overlaps_with_scene = True
                        break

                if overlaps_with_scene:
                    break

            if not overlaps_with_scene:
                positions_without_scene_overlap.append(position)

        print(
            f"[FILTER] {obj_type}: After scene overlap check: {len(positions_without_scene_overlap)} positions remain"
        )

        # STEP 2: Ensure no overlaps between valid positions themselves
        final_positions = []

        for position in positions_without_scene_overlap:
            overlaps_with_other_position = False

            # Check against all already accepted positions
            for accepted_position in final_positions:
                # Perform explicit 2D overlap check
                pos_min_x = position[0] - obj_size[0] / 2
                pos_max_x = position[0] + obj_size[0] / 2
                pos_min_y = position[1] - obj_size[1] / 2
                pos_max_y = position[1] + obj_size[1] / 2

                acc_min_x = accepted_position[0] - obj_size[0] / 2
                acc_max_x = accepted_position[0] + obj_size[0] / 2
                acc_min_y = accepted_position[1] - obj_size[1] / 2
                acc_max_y = accepted_position[1] + obj_size[1] / 2

                # Using strict inequality and a small buffer to avoid floating point issues
                buffer = 0.01
                if not (
                    pos_max_x + buffer < acc_min_x
                    or pos_min_x - buffer > acc_max_x
                    or pos_max_y + buffer < acc_min_y
                    or pos_min_y - buffer > acc_max_y
                ):
                    overlaps_with_other_position = True
                    break

            if not overlaps_with_other_position:
                final_positions.append(position)

        print(f"[FILTER] {obj_type}: Final positions: {len(final_positions)}")
        return final_positions

    def _debug_overlap_check(self, pos1, pos2, obj_size, filtered_idx):
        """Helper method to print detailed debug information for overlap checks"""
        print(f"[DEBUG] Checking position {pos1} against filtered position {filtered_idx}: {pos2}")

        # Calculate bounds
        pos1_min_x = pos1[0] - obj_size[0] / 2
        pos1_max_x = pos1[0] + obj_size[0] / 2
        pos1_min_y = pos1[1] - obj_size[1] / 2
        pos1_max_y = pos1[1] + obj_size[1] / 2

        pos2_min_x = pos2[0] - obj_size[0] / 2
        pos2_max_x = pos2[0] + obj_size[0] / 2
        pos2_min_y = pos2[1] - obj_size[1] / 2
        pos2_max_y = pos2[1] + obj_size[1] / 2

        # Calculate distances
        x_distance = min(abs(pos1_min_x - pos2_max_x), abs(pos1_max_x - pos2_min_x))
        y_distance = min(abs(pos1_min_y - pos2_max_y), abs(pos1_max_y - pos2_min_y))

        # Check if bounds overlap
        x_overlap = not (pos1_max_x <= pos2_min_x or pos1_min_x >= pos2_max_x)
        y_overlap = not (pos1_max_y <= pos2_min_y or pos1_min_y >= pos2_max_y)
        overlaps = x_overlap and y_overlap

        print(
            f"[DEBUG] Position 1 bounds: X=[{pos1_min_x:.2f}, {pos1_max_x:.2f}], Y=[{pos1_min_y:.2f}, {pos1_max_y:.2f}]"
        )
        print(
            f"[DEBUG] Position 2 bounds: X=[{pos2_min_x:.2f}, {pos2_max_x:.2f}], Y=[{pos2_min_y:.2f}, {pos2_max_y:.2f}]"
        )
        print(f"[DEBUG] X overlap: {x_overlap}, Y overlap: {y_overlap}, Overall overlap: {overlaps}")
        print(f"[DEBUG] Distance X: {x_distance:.2f}, Y: {y_distance:.2f}")

        # Add a clearer threshold check with a small epsilon to handle floating point issues
        epsilon = 0.001
        if abs(x_distance) < epsilon or abs(y_distance) < epsilon:
            print(f"[DEBUG] WARNING: Positions are very close! Distance below epsilon={epsilon}")

        return overlaps

    def _check_overlap_with_vertices(
        self, obj_type, obj_vertices, scene_data, parent_obj_type=None, parent_data=None, allow_parent_overlap=False
    ):
        """
        Check if an object overlaps with any existing objects in the scene.

        Args:
            obj_type (str): Type of object being placed
            obj_vertices (list): Vertices of the object's bounding box
            scene_data (dict): Current scene data
            parent_obj_type (str, optional): Type of parent object
            parent_data (dict, optional): Data of parent object
            allow_parent_overlap (bool): Whether to allow overlap with parent

        Returns:
            bool: True if overlap detected, False otherwise
        """
        # Get object size from config
        obj_size = self.config["scene"][obj_type]["size"]

        # Calculate object's x-y bounds
        obj_min_x = float("inf")
        obj_max_x = float("-inf")
        obj_min_y = float("inf")
        obj_max_y = float("-inf")

        for vertex in obj_vertices:
            obj_min_x = min(obj_min_x, vertex[0])
            obj_max_x = max(obj_max_x, vertex[0])
            obj_min_y = min(obj_min_y, vertex[1])
            obj_max_y = max(obj_max_y, vertex[1])

        # Check for overlaps with existing objects
        for existing_type, existing_objects in scene_data["scene"].items():
            # Always skip Floor since everything is placed on it
            if existing_type == "Floor":
                continue

            # Skip parent object if we're allowing parent overlap
            if allow_parent_overlap and existing_type == parent_obj_type:
                continue

            # Check ALL objects (except Floor)
            for existing_obj in existing_objects:
                # Skip if this is the parent object
                if parent_data and existing_obj == parent_data:
                    continue

                # Get existing object's vertices
                existing_vertices = existing_obj.get("vertices", [])
                if not existing_vertices:
                    continue

                # Calculate existing object's x-y bounds
                existing_min_x = float("inf")
                existing_max_x = float("-inf")
                existing_min_y = float("inf")
                existing_max_y = float("-inf")

                for vertex in existing_vertices:
                    existing_min_x = min(existing_min_x, vertex[0])
                    existing_max_x = max(existing_max_x, vertex[0])
                    existing_min_y = min(existing_min_y, vertex[1])
                    existing_max_y = max(existing_max_y, vertex[1])

                # Check for overlap in x-y plane
                if not (
                    obj_max_x < existing_min_x
                    or obj_min_x > existing_max_x
                    or obj_max_y < existing_min_y
                    or obj_min_y > existing_max_y
                ):
                    print(f"[OVERLAP] Overlap detected between {obj_type} and {existing_type}")
                    return True

        return False

    def _aabb_overlap(self, min1, max1, min2, max2, clearance=0):
        """
        Check if two axis-aligned bounding boxes overlap, considering clearance.

        Args:
            min1 (list): Minimum coordinates of first bounding box [x, y, z]
            max1 (list): Maximum coordinates of first bounding box [x, y, z]
            min2 (list): Minimum coordinates of second bounding box [x, y, z]
            max2 (list): Maximum coordinates of second bounding box [x, y, z]
            clearance (float): Minimum distance required between bounding boxes

        Returns:
            bool: True if bounding boxes overlap, False otherwise
        """
        # Check if bounding boxes overlap in all dimensions
        # Use strict inequalities (< and >) to ensure that objects that are just touching
        # (with clearance) are not considered to be overlapping
        return (
            min1[0] + clearance < max2[0]
            and max1[0] - clearance > min2[0]
            and min1[1] + clearance < max2[1]
            and max1[1] - clearance > min2[1]
            and min1[2] + clearance < max2[2]
            and max1[2] - clearance > min2[2]
        )

    def _calculate_bounding_box(self, position, size):
        """
        Calculate the bounding box for an object at a given position.

        Args:
            position (list): [x, y, z] position of the object
            size (list): [width, depth, height] dimensions of the object

        Returns:
            dict: Bounding box with min and max points
        """
        x, y, z = position
        half_width = size[0] / 2
        half_depth = size[1] / 2
        half_height = size[2] / 2

        return {
            "min": [x - half_width, y - half_depth, z - half_height],
            "max": [x + half_width, y + half_depth, z + half_height],
        }

    def _check_overlap(
        self, obj_type, obj_bbox, scene_data, parent_obj_type=None, parent_data=None, allow_parent_overlap=False
    ):
        """
        Check if an object's bounding box overlaps with any existing object.

        This method is now deprecated in favor of _check_overlap_with_vertices
        which handles rotated objects.

        Args:
            obj_type (str): Type of object being placed
            obj_bbox (dict): Bounding box of the object to check
            scene_data (dict): Current scene data with existing objects
            parent_obj_type (str, optional): Type of the parent object
            parent_data (dict, optional): Data of the parent object
            allow_parent_overlap (bool): Whether to allow overlap with parent

        Returns:
            bool: True if there is an overlap, False otherwise
        """
        # Iterate through all object types in the scene
        for other_type, other_objects in scene_data.get("scene", {}).items():
            # Skip Floor as we allow objects to overlap with the floor
            if other_type == "Floor":
                continue

            # Check against each object of this type
            for other_obj in other_objects:
                # Check if this is the parent object and overlap is allowed
                if (
                    allow_parent_overlap
                    and parent_obj_type == other_type
                    and parent_data is not None
                    and other_obj == parent_data
                ):
                    # Skip overlap check for the parent if overlap is allowed
                    continue

                # Get clearance between these object types
                clearance = self.clearance_calculator.calculate_parent_child_clearance(obj_type, other_type)

                other_pos = other_obj["position"]
                other_size = other_obj["size"]

                # Calculate other object's bounding box
                other_bbox = self._calculate_bounding_box(other_pos, other_size)

                # Expand bounding box by clearance
                expanded_bbox = self._expand_bbox_by_clearance(other_bbox, clearance)

                # Check if bounding boxes overlap
                if self._do_bboxes_overlap(obj_bbox, expanded_bbox):
                    return True  # Overlap found

        # No overlaps found
        return False

    def _expand_bbox_by_clearance(self, bbox, clearance):
        """
        Expand a bounding box by the specified clearance in all directions.

        Args:
            bbox (dict): Bounding box with min and max points
            clearance (float): Amount to expand the bounding box by

        Returns:
            dict: Expanded bounding box
        """
        return {
            "min": [bbox["min"][0] - clearance, bbox["min"][1] - clearance, bbox["min"][2] - clearance],
            "max": [bbox["max"][0] + clearance, bbox["max"][1] + clearance, bbox["max"][2] + clearance],
        }

    def _do_bboxes_overlap(self, bbox1, bbox2):
        """
        Check if two bounding boxes overlap.

        Args:
            bbox1 (dict): First bounding box with min and max points
            bbox2 (dict): Second bounding box with min and max points

        Returns:
            bool: True if the bounding boxes overlap, False otherwise
        """
        # Check for non-overlap along each axis
        # If there's a non-overlap along any axis, the boxes don't intersect
        if (
            bbox1["max"][0] < bbox2["min"][0]
            or bbox1["min"][0] > bbox2["max"][0]
            or bbox1["max"][1] < bbox2["min"][1]
            or bbox1["min"][1] > bbox2["max"][1]
            or bbox1["max"][2] < bbox2["min"][2]
            or bbox1["min"][2] > bbox2["max"][2]
        ):
            return False

        # If we get here, the boxes overlap
        return True


class PlacementManager:
    """
    Manages placement strategies for different object types.

    This class is responsible for:
    1. Determining valid positions for objects based on parent-child relationships
    2. Implementing different placement strategies (Top, Under, Tuck, Long, Short)
    3. Calculating object orientations based on placement types
    4. Ensuring objects are placed with proper clearance from their parents
    5. Filtering positions to avoid overlaps between objects

    The placement manager is the core component for determining where objects
    should be placed in a procedurally generated scene to maintain physical
    plausibility and aesthetic quality.
    """

    def __init__(self, config):
        """
        Initialize the placement manager with configuration.

        Args:
            config (dict): Scene configuration containing placement rules
        """
        self.config = config
        self.clearance_calculator = ClearanceCalculator(config)
        self.filter_positions = FilterPositions(config)

    def calculate_orientation(self, obj_position, parent_position, parent_size, parent_rotation, placement_type):
        """
        Calculate the proper orientation for an object based on its placement.

        When strict_orientation is enabled, this method orients the object to face
        the appropriate direction based on its placement relative to the parent.
        The orientation accounts for the parent's rotation to ensure consistent
        relative orientation even when parents are rotated.

        The method handles different placement types specially:
        - Tuck: Object faces the nearest edge of the parent
        - Long/Short: Object faces perpendicular to the parent's side it's placed along
        - Top/Under: Object aligns with the parent's rotation

        Args:
            obj_position (list): [x, y, z] position of the object
            parent_position (list): [x, y, z] position of the parent
            parent_size (list): [x, y, z] size of the parent
            parent_rotation (list): [rx, ry, rz] rotation of the parent in degrees
            placement_type (str): Type of placement (Long, Short, Top, Under, Tuck)

        Returns:
            list: [rx, ry, rz] rotation in degrees
        """
        import math

        # Calculate the vector from object to parent center
        dx = parent_position[0] - obj_position[0]
        dy = parent_position[1] - obj_position[1]

        # Get parent rotation around Z-axis (in radians)
        parent_z_rotation_rad = math.radians(parent_rotation[2])

        # Calculate the angle between object and parent center
        base_angle_rad = math.atan2(dy, dx)

        # Adjust the angle based on parent's rotation
        adjusted_angle_rad = base_angle_rad - parent_z_rotation_rad

        # Convert to degrees
        adjusted_angle_deg = math.degrees(adjusted_angle_rad)

        # Handle different placement types
        if placement_type == "Tuck":
            # For tucked objects, they should face the parent's closest edge

            # Calculate distances to each edge of the parent (accounting for rotation)
            parent_half_width = parent_size[0] / 2
            parent_half_depth = parent_size[1] / 2

            # Rotate the vector from parent to object by negative parent rotation
            # to align with parent's local coordinate system
            rotated_dx = dx * math.cos(-parent_z_rotation_rad) - dy * math.sin(-parent_z_rotation_rad)
            rotated_dy = dx * math.sin(-parent_z_rotation_rad) + dy * math.cos(-parent_z_rotation_rad)

            # Calculate distance to each edge
            dist_to_right = abs(parent_half_width - abs(rotated_dx))
            dist_to_left = abs(parent_half_width + abs(rotated_dx))
            dist_to_front = abs(parent_half_depth - abs(rotated_dy))
            dist_to_back = abs(parent_half_depth + abs(rotated_dy))

            # Determine which edge is closest
            min_dist = min(dist_to_right, dist_to_left, dist_to_front, dist_to_back)

            # Orient the object to face the closest edge
            if min_dist == dist_to_right:
                # Object should face right edge of parent
                return [0, 0, parent_rotation[2] + 90]
            elif min_dist == dist_to_left:
                # Object should face left edge of parent
                return [0, 0, parent_rotation[2] - 90]
            elif min_dist == dist_to_front:
                # Object should face front edge of parent
                return [0, 0, parent_rotation[2] + 180]
            else:  # min_dist == dist_to_back
                # Object should face back edge of parent
                return [0, 0, parent_rotation[2]]

        elif placement_type in ["Long", "Short"]:
            # For objects placed along the sides, face perpendicular to the parent

            # Rotate the vector to parent's coordinate system
            rotated_dx = dx * math.cos(-parent_z_rotation_rad) - dy * math.sin(-parent_z_rotation_rad)
            rotated_dy = dx * math.sin(-parent_z_rotation_rad) + dy * math.cos(-parent_z_rotation_rad)

            # Determine which side the object is on
            if abs(rotated_dx) > abs(rotated_dy):
                # Object is on left or right side
                if rotated_dx > 0:
                    # Object is on right side, face left (towards parent)
                    return [0, 0, parent_rotation[2] + 90]
                else:
                    # Object is on left side, face right (towards parent)
                    return [0, 0, parent_rotation[2] - 90]
            else:
                # Object is on front or back side
                if rotated_dy > 0:
                    # Object is on back side, face front (towards parent)
                    return [0, 0, parent_rotation[2] + 180]
                else:
                    # Object is on front side, face back (towards parent)
                    return [0, 0, parent_rotation[2]]

        elif placement_type == "Top":
            # For objects on top, align with the parent's rotation
            return [0, 0, parent_rotation[2]]

        elif placement_type == "Under":
            # For objects underneath, also align with the parent's rotation
            return [0, 0, parent_rotation[2]]

        else:
            # For other placement types, use 90-degree snapping based on angle
            if -45 <= adjusted_angle_deg < 45:
                return [0, 0, parent_rotation[2] + 90]  # Face right relative to parent
            elif 45 <= adjusted_angle_deg < 135:
                return [0, 0, parent_rotation[2] + 180]  # Face up relative to parent
            elif adjusted_angle_deg >= 135 or adjusted_angle_deg < -135:
                return [0, 0, parent_rotation[2] - 90]  # Face left relative to parent
            else:  # -135 <= adjusted_angle_deg < -45
                return [0, 0, parent_rotation[2]]  # Face down relative to parent

    def get_valid_positions(self, obj_type, parent_obj_type, parent_data, scene_data=None):
        """
        Get all valid positions for placing an object relative to its parent.

        This method:
        1. Determines valid positions based on the placement type (Top, Under, etc.)
        2. Filters positions to ensure no overlaps with existing objects
        3. Returns only positions that satisfy both placement and overlap constraints

        Args:
            obj_type (str): Type of object to place
            parent_obj_type (str): Type of parent object
            parent_data (dict): Data of parent object
            scene_data (dict, optional): Current scene data for overlap checking

        Returns:
            list: List of valid positions
        """
        placement_types = self.config["scene"][obj_type]["placement"]

        # Print information about the parent we're using
        print(
            f"[POSITIONS] Getting valid positions for {obj_type} relative to {parent_obj_type} at"
            f" {parent_data['position']} with rotation {parent_data.get('rotation', [0, 0, 0])}"
        )

        valid_positions = []

        # Apply each placement type and collect valid positions
        for placement_type in placement_types:
            if placement_type == "Top":
                # Place object on top of parent
                positions = self._get_top_positions(obj_type, parent_data, parent_obj_type)
                valid_positions.extend(positions)
                print(f"[POSITIONS] {obj_type}: Added {len(positions)} Top positions")

            elif placement_type == "Under":
                # Place object under the parent
                positions = self._get_under_positions(obj_type, parent_data, parent_obj_type)
                valid_positions.extend(positions)
                print(f"[POSITIONS] {obj_type}: Added {len(positions)} Under positions")

            elif placement_type == "Tuck":
                # Place object tucked against the parent
                positions = self._get_tuck_positions(obj_type, parent_data, parent_obj_type)
                valid_positions.extend(positions)
                print(f"[POSITIONS] {obj_type}: Added {len(positions)} Tuck positions")

            elif placement_type == "Long" or placement_type == "Short":
                # Calculate clearance between objects
                clearance = self.clearance_calculator.calculate_parent_child_clearance(obj_type, parent_obj_type)

                if placement_type == "Long":
                    # Place object along the long sides of the parent
                    positions = self._get_long_positions(obj_type, parent_data, clearance, parent_obj_type)
                    valid_positions.extend(positions)
                    print(f"[POSITIONS] {obj_type}: Added {len(positions)} Long positions")

                elif placement_type == "Short":
                    # Place object along the short sides of the parent
                    positions = self._get_short_positions(obj_type, parent_data, clearance, parent_obj_type)
                    valid_positions.extend(positions)
                    print(f"[POSITIONS] {obj_type}: Added {len(positions)} Short positions")

        print(f"[POSITIONS] {obj_type}: Total potential positions before filtering: {len(valid_positions)}")

        # Filter positions to ensure no overlaps if scene_data is provided
        if scene_data is not None:
            valid_positions = self.filter_positions.filter_positions(
                obj_type, valid_positions, scene_data, parent_obj_type, parent_data
            )
            print(f"[POSITIONS] {obj_type}: Positions after overlap filtering: {len(valid_positions)}")

        return valid_positions

    def _get_top_positions(self, obj_type, parent_data, parent_type):
        """
        Get valid positions for placing an object on top of its parent.

        This method calculates a grid of positions on top of the parent with:
        1. Proper spacing between objects based on their size
        2. Objects placed at the top of the parent (z = parent's height)
        3. Evenly distributed positions across the parent's top surface
        4. Proper handling of rotated parent objects
        5. Special handling for Floor parent (z=0)

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data (position, size, rotation, etc.)
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions on top of the parent
        """
        # Extract parent and object data
        parent_pos = parent_data["position"]
        parent_size = parent_data["size"]
        parent_rotation = parent_data.get("rotation", [0, 0, 0])
        obj_size = self.config["scene"][obj_type]["size"]
        floor_size = self.config["scene"]["Floor"]["size"]

        # Calculate object placement height (top of parent)
        # Special case for Floor as parent - place objects at floor level (z=0)
        if parent_type == "Floor":
            z_pos = 0  # Object bottom at z=0 for floor
        else:
            z_pos = parent_pos[2] + parent_size[2]

        valid_positions = []

        # Small gap between objects for better spacing
        gap = obj_size[0] * 0.1

        # Handle rotated parent case
        if parent_rotation[2] != 0:
            import math

            # Get the parent's rotation in radians
            z_rotation_rad = math.radians(parent_rotation[2])

            # Calculate grid spacing and dimensions
            spacing_x = obj_size[0] + gap
            spacing_y = obj_size[1] + gap

            # Calculate available area dimensions (accounting for object size)
            avail_x = parent_size[0] - obj_size[0]
            avail_y = parent_size[1] - obj_size[1]

            # Skip if object doesn't fit on parent
            if avail_x <= 0 or avail_y <= 0:
                return []

            # Calculate number of positions that can fit along each dimension
            num_x = max(1, int(avail_x / spacing_x) + 1)
            num_y = max(1, int(avail_y / spacing_y) + 1)

            # Generate grid of positions in local parent coordinates
            for i in range(num_x):
                for j in range(num_y):
                    # Calculate offset from parent center in local coordinates
                    local_offset_x = (i * spacing_x) - (avail_x / 2)
                    local_offset_y = (j * spacing_y) - (avail_y / 2)

                    # Rotate the local offsets to world space
                    world_offset_x = local_offset_x * math.cos(z_rotation_rad) - local_offset_y * math.sin(
                        z_rotation_rad
                    )
                    world_offset_y = local_offset_x * math.sin(z_rotation_rad) + local_offset_y * math.cos(
                        z_rotation_rad
                    )

                    # Calculate world position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, z_pos])

            return valid_positions

        # For non-rotated parents, continue with original approach
        # Calculate grid spacing and dimensions
        spacing_x = obj_size[0] + gap
        spacing_y = obj_size[1] + gap

        # Calculate available area dimensions (accounting for object size)
        avail_x = parent_size[0] - obj_size[0]
        avail_y = parent_size[1] - obj_size[1]

        # Skip if object doesn't fit on parent
        if avail_x <= 0 or avail_y <= 0:
            return []

        # Calculate number of positions that can fit along each dimension
        num_x = max(1, int(avail_x / spacing_x) + 1)
        num_y = max(1, int(avail_y / spacing_y) + 1)

        # Generate grid of positions
        for i in range(num_x):
            for j in range(num_y):
                x = parent_pos[0] + (i * spacing_x) - (avail_x / 2)
                y = parent_pos[1] + (j * spacing_y) - (avail_y / 2)

                # Check if position is within floor bounds
                if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                    valid_positions.append([x, y, z_pos])

        return valid_positions

    def _get_under_positions(self, obj_type, parent_data, parent_type):
        """
        Get valid positions for placing an object under its parent.

        Places objects underneath their parent objects, with the following considerations:
        1. Uses 80% of the parent's width and length to avoid legs
        2. Creates a grid of positions within the usable area
        3. Ensures positions are within floor bounds
        4. Sets z position at floor level
        5. Properly handles rotated parent objects

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data (position, size, rotation, etc.)
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions under the parent
        """
        # Extract parent and object data
        parent_pos = parent_data["position"]
        parent_size = parent_data["size"]
        parent_rotation = parent_data.get("rotation", [0, 0, 0])
        obj_size = self.config["scene"][obj_type]["size"]
        floor_size = self.config["scene"]["Floor"]["size"]

        valid_positions = []

        # Can't place under Floor
        if parent_type == "Floor":
            return []

        # Set z position at floor level
        z_pos = 0

        # Handle rotated parent case
        if parent_rotation[2] != 0:
            import math

            # Get the parent's rotation in radians
            z_rotation_rad = math.radians(parent_rotation[2])

            # Restrict to 80% of parent's width and length to avoid legs
            usable_parent_width = parent_size[0] * 0.8
            usable_parent_length = parent_size[1] * 0.8

            # Calculate available area under parent (within 80% bounds)
            available_width = usable_parent_width - obj_size[0]
            available_length = usable_parent_length - obj_size[1]

            # If parent is too small, can't place under it
            if available_width <= 0 or available_length <= 0:
                return []

            # Small gap between objects for better spacing
            gap = obj_size[0] * 0.1

            # Calculate grid spacing and dimensions
            spacing_x = obj_size[0] + gap
            spacing_y = obj_size[1] + gap
            num_x = max(1, int(available_width / spacing_x) + 1)
            num_y = max(1, int(available_length / spacing_y) + 1)

            # Generate grid of positions in local space, then transform to world space
            for i in range(num_x):
                for j in range(num_y):
                    # Calculate local offsets from parent center
                    # Restricted to 80% area
                    local_offset_x = ((i * spacing_x) - (available_width / 2)) * 0.8
                    local_offset_y = ((j * spacing_y) - (available_length / 2)) * 0.8

                    # Rotate local coordinates to world space
                    world_offset_x = local_offset_x * math.cos(z_rotation_rad) - local_offset_y * math.sin(
                        z_rotation_rad
                    )
                    world_offset_y = local_offset_x * math.sin(z_rotation_rad) + local_offset_y * math.cos(
                        z_rotation_rad
                    )

                    # Calculate absolute position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, z_pos])

        else:
            # Non-rotated parent case (original implementation)
            # Restrict to 80% of parent's width and length to avoid legs
            usable_parent_width = parent_size[0] * 0.8
            usable_parent_length = parent_size[1] * 0.8

            # Calculate available area under parent (within 80% bounds)
            available_width = usable_parent_width - obj_size[0]
            available_length = usable_parent_length - obj_size[1]

            # If parent is too small, can't place under it
            if available_width <= 0 or available_length <= 0:
                return []

            # Small gap between objects for better spacing
            gap = obj_size[0] * 0.1

            # Calculate grid spacing and dimensions
            spacing_x = obj_size[0] + gap
            spacing_y = obj_size[1] + gap
            num_x = max(1, int(available_width / spacing_x) + 1)
            num_y = max(1, int(available_length / spacing_y) + 1)

            # Generate grid of positions
            for i in range(num_x):
                for j in range(num_y):
                    # Calculate offsets from parent center
                    offset_x = (i * spacing_x) - (available_width / 2)
                    offset_y = (j * spacing_y) - (available_length / 2)

                    # Calculate absolute position
                    x = parent_pos[0] + offset_x
                    y = parent_pos[1] + offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, z_pos])

        # If no valid positions were found, fall back to center if it's above the floor
        if not valid_positions and z_pos >= obj_size[2] / 2:
            valid_positions.append([parent_pos[0], parent_pos[1], z_pos])

        return valid_positions

    def _get_tuck_positions(self, obj_type, parent_data, parent_type):
        """
        Get valid positions for tucking an object under its parent with 50% overlap.

        This method places objects at the sides of their parent with 50% overlap,
        which creates a visually appealing arrangement where objects are:
        1. Positioned along the parent edges
        2. Aligned with the exact parent edge for a clean appearance
        3. Positioned at floor level (z=0)
        4. Checked to ensure they remain within floor bounds
        5. Properly handles rotated parent objects

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data (position, size, rotation, etc.)
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions for tucked placement
        """
        # Extract parent and object data
        parent_pos = parent_data["position"]
        parent_size = parent_data["size"]
        parent_rotation = parent_data.get("rotation", [0, 0, 0])
        obj_size = self.config["scene"][obj_type]["size"]
        floor_size = self.config["scene"]["Floor"]["size"]

        valid_positions = []

        # Can't tuck with Floor as parent
        if parent_type == "Floor":
            return []

        # Handle rotated parent case
        if parent_rotation[2] != 0:
            import math

            # Get the parent's rotation in radians
            z_rotation_rad = math.radians(parent_rotation[2])

            # Determine which side is shorter in local space
            is_x_shorter = parent_size[0] < parent_size[1]

            if is_x_shorter:
                # Tuck along the X-axis (left and right of parent) in local space

                # Calculate local offsets for 50% overlap
                local_left_x = -parent_size[0] / 2  # Left edge in local space
                local_right_x = parent_size[0] / 2  # Right edge in local space
                local_y = 0  # Center along Y axis

                # Transform from local to world coordinates
                # Left side
                world_left_x = local_left_x * math.cos(z_rotation_rad) - local_y * math.sin(z_rotation_rad)
                world_left_y = local_left_x * math.sin(z_rotation_rad) + local_y * math.cos(z_rotation_rad)

                # Right side
                world_right_x = local_right_x * math.cos(z_rotation_rad) - local_y * math.sin(z_rotation_rad)
                world_right_y = local_right_x * math.sin(z_rotation_rad) + local_y * math.cos(z_rotation_rad)

                # Add world positions to parent position
                left_pos_x = parent_pos[0] + world_left_x
                left_pos_y = parent_pos[1] + world_left_y
                right_pos_x = parent_pos[0] + world_right_x
                right_pos_y = parent_pos[1] + world_right_y

                # Only add positions that are within floor bounds
                if (
                    abs(left_pos_x) + obj_size[0] / 2 <= floor_size[0] / 2
                    and abs(left_pos_y) + obj_size[1] / 2 <= floor_size[1] / 2
                ):
                    valid_positions.append([left_pos_x, left_pos_y, 0])

                if (
                    abs(right_pos_x) + obj_size[0] / 2 <= floor_size[0] / 2
                    and abs(right_pos_y) + obj_size[1] / 2 <= floor_size[1] / 2
                ):
                    valid_positions.append([right_pos_x, right_pos_y, 0])

            else:
                # Tuck along the Y-axis (front and back of parent) in local space

                # Calculate local offsets for 50% overlap
                local_x = 0  # Center along X axis
                local_front_y = -parent_size[1] / 2  # Front edge in local space
                local_back_y = parent_size[1] / 2  # Back edge in local space

                # Transform from local to world coordinates
                # Front side
                world_front_x = local_x * math.cos(z_rotation_rad) - local_front_y * math.sin(z_rotation_rad)
                world_front_y = local_x * math.sin(z_rotation_rad) + local_front_y * math.cos(z_rotation_rad)

                # Back side
                world_back_x = local_x * math.cos(z_rotation_rad) - local_back_y * math.sin(z_rotation_rad)
                world_back_y = local_x * math.sin(z_rotation_rad) + local_back_y * math.cos(z_rotation_rad)

                # Add world positions to parent position
                front_pos_x = parent_pos[0] + world_front_x
                front_pos_y = parent_pos[1] + world_front_y
                back_pos_x = parent_pos[0] + world_back_x
                back_pos_y = parent_pos[1] + world_back_y

                # Only add positions that are within floor bounds
                if (
                    abs(front_pos_x) + obj_size[0] / 2 <= floor_size[0] / 2
                    and abs(front_pos_y) + obj_size[1] / 2 <= floor_size[1] / 2
                ):
                    valid_positions.append([front_pos_x, front_pos_y, 0])

                if (
                    abs(back_pos_x) + obj_size[0] / 2 <= floor_size[0] / 2
                    and abs(back_pos_y) + obj_size[1] / 2 <= floor_size[1] / 2
                ):
                    valid_positions.append([back_pos_x, back_pos_y, 0])

        else:
            # Non-rotated parent case (original implementation)
            # Determine which side is shorter
            is_x_shorter = parent_size[0] < parent_size[1]

            # Calculate the exact positions for 50% overlap
            if is_x_shorter:
                # Tuck along the X-axis (left and right of parent)
                # Calculate parent's left and right edges
                parent_left_edge = parent_pos[0] - parent_size[0] / 2
                parent_right_edge = parent_pos[0] + parent_size[0] / 2

                # Calculate object positions for 50% overlap
                # Position objects exactly at the edge for clean alignment
                left_pos_x = parent_left_edge
                right_pos_x = parent_right_edge

                # Only add positions that are within floor bounds
                if abs(left_pos_x) + obj_size[0] / 2 <= floor_size[0] / 2:
                    valid_positions.append([left_pos_x, parent_pos[1], 0])

                if abs(right_pos_x) + obj_size[0] / 2 <= floor_size[0] / 2:
                    valid_positions.append([right_pos_x, parent_pos[1], 0])

            else:
                # Tuck along the Y-axis (front and back of parent)
                # Calculate parent's front and back edges
                parent_front_edge = parent_pos[1] - parent_size[1] / 2
                parent_back_edge = parent_pos[1] + parent_size[1] / 2

                # Calculate object positions for 50% overlap
                # Position objects exactly at the edge for clean alignment
                front_pos_y = parent_front_edge
                back_pos_y = parent_back_edge

                # Only add positions that are within floor bounds
                if abs(front_pos_y) + obj_size[1] / 2 <= floor_size[1] / 2:
                    valid_positions.append([parent_pos[0], front_pos_y, 0])

                if abs(back_pos_y) + obj_size[1] / 2 <= floor_size[1] / 2:
                    valid_positions.append([parent_pos[0], back_pos_y, 0])

        return valid_positions

    def _get_long_positions(self, obj_type, parent_data, clearance, parent_type):
        """
        Get valid positions for placing an object on the longer sides of its parent.

        This method places objects along the two longer sides of the parent, with:
        1. Proper clearance between parent and child
        2. Evenly spaced positions along each side
        3. Appropriate checks to ensure positions remain within floor bounds
        4. Objects placed at floor level (z=0)
        5. Proper handling of rotated parent objects

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data (position, size, rotation, etc.)
            clearance (float): Clearance distance between parent and child
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions along the longer sides
        """
        # Extract parent and object data
        parent_pos = parent_data["position"]
        parent_size = parent_data["size"]
        parent_rotation = parent_data.get("rotation", [0, 0, 0])
        obj_size = self.config["scene"][obj_type]["size"]
        floor_size = self.config["scene"]["Floor"]["size"]

        valid_positions = []

        # Small gap between objects for spacing (10% of object width)
        gap = obj_size[0] * 0.1

        # Handle rotated parent case
        if parent_rotation[2] != 0:
            import math

            # Get the parent's rotation in radians
            z_rotation_rad = math.radians(parent_rotation[2])

            # Determine which dimension is longer in local space
            is_x_longer = parent_size[0] >= parent_size[1]

            if is_x_longer:
                # Parent's long sides are along x-axis in local space
                # These become the front and back sides when rotated

                # Calculate offset from parent edge including clearance
                offset = parent_size[1] / 2 + clearance + obj_size[1] / 2

                # Calculate available length along parent's long dimension
                available_length = parent_size[0] - obj_size[0]

                # If no room along parent's edge, return empty list
                if available_length <= 0:
                    return []

                # Calculate spacing between objects along the edge
                spacing = obj_size[0] + gap

                # Calculate number of positions that can fit
                num_positions = max(1, int(available_length / spacing) + 1)

                # Place objects along both longer sides
                for i in range(num_positions):
                    # Calculate local offset from parent center
                    local_offset_x = (i * spacing) - (available_length / 2)

                    # Place along front side (negative y in local space)
                    local_y = -offset

                    # Rotate the local offsets to world space
                    world_offset_x = local_offset_x * math.cos(z_rotation_rad) - local_y * math.sin(z_rotation_rad)
                    world_offset_y = local_offset_x * math.sin(z_rotation_rad) + local_y * math.cos(z_rotation_rad)

                    # Calculate world position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, 0])

                    # Place along back side (positive y in local space)
                    local_y = offset

                    # Rotate the local offsets to world space
                    world_offset_x = local_offset_x * math.cos(z_rotation_rad) - local_y * math.sin(z_rotation_rad)
                    world_offset_y = local_offset_x * math.sin(z_rotation_rad) + local_y * math.cos(z_rotation_rad)

                    # Calculate world position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, 0])
            else:
                # Parent's long sides are along y-axis in local space
                # These become the left and right sides when rotated

                # Calculate offset from parent edge including clearance
                offset = parent_size[0] / 2 + clearance + obj_size[0] / 2

                # Calculate available length along parent's long dimension
                available_length = parent_size[1] - obj_size[1]

                # If no room along parent's edge, return empty list
                if available_length <= 0:
                    return []

                # Calculate spacing between objects along the edge
                spacing = obj_size[1] + gap

                # Calculate number of positions that can fit
                num_positions = max(1, int(available_length / spacing) + 1)

                # Place objects along both longer sides
                for i in range(num_positions):
                    # Calculate local offset from parent center
                    local_offset_y = (i * spacing) - (available_length / 2)

                    # Place along left side (negative x in local space)
                    local_x = -offset

                    # Rotate the local offsets to world space
                    world_offset_x = local_x * math.cos(z_rotation_rad) - local_offset_y * math.sin(z_rotation_rad)
                    world_offset_y = local_x * math.sin(z_rotation_rad) + local_offset_y * math.cos(z_rotation_rad)

                    # Calculate world position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, 0])

                    # Place along right side (positive x in local space)
                    local_x = offset

                    # Rotate the local offsets to world space
                    world_offset_x = local_x * math.cos(z_rotation_rad) - local_offset_y * math.sin(z_rotation_rad)
                    world_offset_y = local_x * math.sin(z_rotation_rad) + local_offset_y * math.cos(z_rotation_rad)

                    # Calculate world position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, 0])

            return valid_positions

        else:
            # Non-rotated parent case (original implementation)
            # Determine which dimension is longer
            is_x_longer = parent_size[0] >= parent_size[1]

            if is_x_longer:
                # Parent is wider than deep - place along front and back (Y-axis)
                # Calculate offset from parent edge including clearance
                offset_y = parent_size[1] / 2 + clearance + obj_size[1] / 2

                # Calculate available length along parent's X dimension
                available_length = parent_size[0] - obj_size[0]

                # If no room along parent's edge, return empty list
                if available_length <= 0:
                    return []

                # Calculate spacing between objects along the edge
                spacing = obj_size[0] + gap

                # Calculate number of positions that can fit
                num_positions = max(1, int(available_length / spacing) + 1)

                # Place objects along both sides (front and back)
                for i in range(num_positions):
                    # Calculate offset from parent center
                    offset_x = (i * spacing) - (available_length / 2)

                    # Front side position
                    y = parent_pos[1] - offset_y
                    if abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([parent_pos[0] + offset_x, y, 0])

                    # Back side position
                    y = parent_pos[1] + offset_y
                    if abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([parent_pos[0] + offset_x, y, 0])

            else:
                # Parent is deeper than wide - place along left and right (X-axis)
                # Calculate offset from parent edge including clearance
                offset_x = parent_size[0] / 2 + clearance + obj_size[0] / 2

                # Calculate available length along parent's Y dimension
                available_length = parent_size[1] - obj_size[1]

                # If no room along parent's edge, return empty list
                if available_length <= 0:
                    return []

                # Calculate spacing between objects along the edge
                spacing = obj_size[1] + gap

                # Calculate number of positions that can fit
                num_positions = max(1, int(available_length / spacing) + 1)

                # Place objects along both sides (left and right)
                for i in range(num_positions):
                    # Calculate offset from parent center
                    offset_y = (i * spacing) - (available_length / 2)

                    # Left side position
                    x = parent_pos[0] - offset_x
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2:
                        valid_positions.append([x, parent_pos[1] + offset_y, 0])

                    # Right side position
                    x = parent_pos[0] + offset_x
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2:
                        valid_positions.append([x, parent_pos[1] + offset_y, 0])

            return valid_positions

    def _get_short_positions(self, obj_type, parent_data, clearance, parent_type):
        """
        Get valid positions for placing an object on the shorter sides of its parent.

        This method places objects along the two shorter sides of the parent, with:
        1. Proper clearance between parent and child
        2. Evenly spaced positions along each side
        3. Appropriate checks to ensure positions remain within floor bounds
        4. Objects placed at floor level (z=0)
        5. Proper handling of rotated parent objects

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data (position, size, rotation, etc.)
            clearance (float): Clearance distance between parent and child
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions along the shorter sides
        """
        # Extract parent and object data
        parent_pos = parent_data["position"]
        parent_size = parent_data["size"]
        parent_rotation = parent_data.get("rotation", [0, 0, 0])
        obj_size = self.config["scene"][obj_type]["size"]
        floor_size = self.config["scene"]["Floor"]["size"]

        valid_positions = []

        # Small gap between objects for spacing (10% of object width)
        gap = obj_size[0] * 0.1

        # Handle rotated parent case
        if parent_rotation[2] != 0:
            import math

            # Get the parent's rotation in radians
            z_rotation_rad = math.radians(parent_rotation[2])

            # Determine which dimension is shorter in local space
            is_x_shorter = parent_size[0] < parent_size[1]

            if not is_x_shorter:
                # Parent's short sides are along y-axis in local space
                # These become the left and right sides when rotated

                # Calculate offset from parent edge including clearance
                offset = parent_size[0] / 2 + clearance + obj_size[0] / 2

                # Calculate available length along parent's short dimension
                available_length = parent_size[1] - obj_size[1]

                # If no room along parent's edge, return empty list
                if available_length <= 0:
                    return []

                # Calculate spacing between objects along the edge
                spacing = obj_size[1] + gap

                # Calculate number of positions that can fit
                num_positions = max(1, int(available_length / spacing) + 1)

                # Place objects along both shorter sides
                for i in range(num_positions):
                    # Calculate local offset from parent center
                    local_offset_y = (i * spacing) - (available_length / 2)

                    # Place along left side (negative x in local space)
                    local_x = -offset

                    # Rotate the local offsets to world space
                    world_offset_x = local_x * math.cos(z_rotation_rad) - local_offset_y * math.sin(z_rotation_rad)
                    world_offset_y = local_x * math.sin(z_rotation_rad) + local_offset_y * math.cos(z_rotation_rad)

                    # Calculate world position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, 0])

                    # Place along right side (positive x in local space)
                    local_x = offset

                    # Rotate the local offsets to world space
                    world_offset_x = local_x * math.cos(z_rotation_rad) - local_offset_y * math.sin(z_rotation_rad)
                    world_offset_y = local_x * math.sin(z_rotation_rad) + local_offset_y * math.cos(z_rotation_rad)

                    # Calculate world position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, 0])
            else:
                # Parent's short sides are along x-axis in local space
                # These become the front and back sides when rotated

                # Calculate offset from parent edge including clearance
                offset = parent_size[1] / 2 + clearance + obj_size[1] / 2

                # Calculate available length along parent's short dimension
                available_length = parent_size[0] - obj_size[0]

                # If no room along parent's edge, return empty list
                if available_length <= 0:
                    return []

                # Calculate spacing between objects along the edge
                spacing = obj_size[0] + gap

                # Calculate number of positions that can fit
                num_positions = max(1, int(available_length / spacing) + 1)

                # Place objects along both shorter sides
                for i in range(num_positions):
                    # Calculate local offset from parent center
                    local_offset_x = (i * spacing) - (available_length / 2)

                    # Place along front side (negative y in local space)
                    local_y = -offset

                    # Rotate the local offsets to world space
                    world_offset_x = local_offset_x * math.cos(z_rotation_rad) - local_y * math.sin(z_rotation_rad)
                    world_offset_y = local_offset_x * math.sin(z_rotation_rad) + local_y * math.cos(z_rotation_rad)

                    # Calculate world position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, 0])

                    # Place along back side (positive y in local space)
                    local_y = offset

                    # Rotate the local offsets to world space
                    world_offset_x = local_offset_x * math.cos(z_rotation_rad) - local_y * math.sin(z_rotation_rad)
                    world_offset_y = local_offset_x * math.sin(z_rotation_rad) + local_y * math.cos(z_rotation_rad)

                    # Calculate world position
                    x = parent_pos[0] + world_offset_x
                    y = parent_pos[1] + world_offset_y

                    # Check if position is within floor bounds
                    if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                        valid_positions.append([x, y, 0])

            return valid_positions

        # For non-rotated parent case:
        # Determine which dimension is shorter
        is_x_shorter = parent_size[0] < parent_size[1]

        if not is_x_shorter:
            # Parent is wider than deep - place along left and right (X-axis)
            # Calculate offset from parent edge including clearance
            offset_x = parent_size[0] / 2 + clearance + obj_size[0] / 2

            # Calculate available length along parent's Y dimension
            available_length = parent_size[1] - obj_size[1]

            # If no room along parent's edge, return empty list
            if available_length <= 0:
                return []

            # Calculate spacing between objects along the edge
            spacing = obj_size[1] + gap

            # Calculate number of positions that can fit
            num_positions = max(1, int(available_length / spacing) + 1)

            # Place objects along both sides (left and right)
            for i in range(num_positions):
                # Calculate offset from parent center
                offset_y = (i * spacing) - (available_length / 2)

                # Left side position
                x = parent_pos[0] - offset_x
                if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2:
                    valid_positions.append([x, parent_pos[1] + offset_y, 0])

                # Right side position
                x = parent_pos[0] + offset_x
                if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2:
                    valid_positions.append([x, parent_pos[1] + offset_y, 0])

        else:
            # Parent is deeper than wide - place along front and back (Y-axis)
            # Calculate offset from parent edge including clearance
            offset_y = parent_size[1] / 2 + clearance + obj_size[1] / 2

            # Calculate available length along parent's X dimension
            available_length = parent_size[0] - obj_size[0]

            # If no room along parent's edge, return empty list
            if available_length <= 0:
                return []

            # Calculate spacing between objects along the edge
            spacing = obj_size[0] + gap

            # Calculate number of positions that can fit
            num_positions = max(1, int(available_length / spacing) + 1)

            # Place objects along both sides (front and back)
            for i in range(num_positions):
                # Calculate offset from parent center
                offset_x = (i * spacing) - (available_length / 2)

                # Front side position
                y = parent_pos[1] - offset_y
                if abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                    valid_positions.append([parent_pos[0] + offset_x, y, 0])

                # Back side position
                y = parent_pos[1] + offset_y
                if abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                    valid_positions.append([parent_pos[0] + offset_x, y, 0])

        return valid_positions


class ObjectManager:
    """
    Manages object creation, scaling, and bounding box calculations.

    This class is responsible for:
    1. Creating objects with appropriate asset paths
    2. Scaling objects to match their target dimensions
    3. Calculating bounding box vertices for collision detection
    4. Maintaining uniform scale to preserve object proportions

    The ObjectManager ensures that objects have the correct dimensions
    and appearance when placed in the scene.
    """

    def __init__(self, config):
        """
        Initialize the object manager with configuration.

        Args:
            config (dict): Scene configuration containing object definitions
        """
        self.config = config

    def get_random_asset(self, obj_type):
        """
        Get a random USD asset file for the given object type.

        Randomly selects from available USD files in the object's asset directory
        to provide variety in the scene.

        Args:
            obj_type (str): Type of object to get an asset for

        Returns:
            str or None: Path to a random USD asset file, or None if no assets found
        """
        # Get base assets directory from configuration
        assets_dir = self.config["output"]["asset_dir"]
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

    def calculate_scale_factor(self, asset_path, target_size):
        """
        Calculate uniform scale factor to fit an asset within target size.

        This method:
        1. Loads the asset and computes its original bounds
        2. Calculates scale factors for each dimension
        3. Uses the minimum scale factor to maintain aspect ratio
        4. Returns the scale factor and actual size after scaling

        Args:
            asset_path (str): Path to the asset file
            target_size (list): Target [x, y, z] size for the object

        Returns:
            tuple: (scale_factor, true_size) after scaling
        """
        if asset_path is None:
            return 1.0, target_size

        # Load asset and compute bounds
        temp_stage = Usd.Stage.CreateInMemory()
        temp_ref = temp_stage.DefinePrim("/Temp", "Xform")
        temp_ref.GetReferences().AddReference(os.path.abspath(asset_path))
        temp_geom = UsdGeom.Imageable(temp_ref)
        bounds = temp_geom.ComputeLocalBound(0.0, "default")

        # Get original size from bounds
        bbox = bounds.ComputeAlignedBox()
        min_point = bbox.GetMin()
        max_point = bbox.GetMax()
        original_size = max_point - min_point

        # Calculate uniform scale factor
        target_size_vec = Gf.Vec3d(*target_size)

        # Calculate scale factor for each dimension
        scale_factors = [t / o if o != 0 else 1.0 for t, o in zip(target_size_vec, original_size)]

        # Use minimum to maintain aspect ratio (uniform scaling)
        scale_factor = min(scale_factors)

        # Calculate true size after scaling
        true_size = [float(x) for x in (original_size * scale_factor)]

        return float(scale_factor), true_size

    def calculate_bounding_box_vertices(self, position, size, rotation=None):
        """
        Calculate the 8 vertices of an object's bounding box, with optional rotation.

        These vertices are used for collision detection and placing objects
        in relation to each other. When rotation is specified, the bounding box
        is properly rotated to match the object's orientation.

        Args:
            position (list): [x, y, z] center position of the object
            size (list): [width, depth, height] dimensions of the object
            rotation (list, optional): [rx, ry, rz] rotation in degrees

        Returns:
            list: 8 vertices of the bounding box as [x, y, z] coordinates
        """
        # Extract position and half-dimensions
        x, y, z = position
        half_width = size[0] / 2
        half_depth = size[1] / 2
        half_height = size[2] / 2

        # Calculate the 8 vertices of the unrotated bounding box
        vertices = [
            # Bottom face (z is minimum)
            [x - half_width, y - half_depth, z - half_height],  # Bottom-left-back
            [x + half_width, y - half_depth, z - half_height],  # Bottom-right-back
            [x + half_width, y + half_depth, z - half_height],  # Bottom-right-front
            [x - half_width, y + half_depth, z - half_height],  # Bottom-left-front
            # Top face (z is maximum)
            [x - half_width, y - half_depth, z + half_height],  # Top-left-back
            [x + half_width, y - half_depth, z + half_height],  # Top-right-back
            [x + half_width, y + half_depth, z + half_height],  # Top-right-front
            [x - half_width, y + half_depth, z + half_height],  # Top-left-front
        ]

        # If rotation is specified, apply rotation to all vertices
        if rotation:
            import math

            # Convert rotation from degrees to radians
            rx, ry, rz = (math.radians(r) for r in rotation)

            # Only handle z-rotation for now (most common case)
            if rz != 0:
                # Create rotation matrix for z-axis
                cos_z = math.cos(rz)
                sin_z = math.sin(rz)

                # Apply rotation to each vertex
                for i in range(len(vertices)):
                    # Extract coordinates relative to object center
                    vx = vertices[i][0] - x
                    vy = vertices[i][1] - y

                    # Apply rotation
                    rotated_x = vx * cos_z - vy * sin_z
                    rotated_y = vx * sin_z + vy * cos_z

                    # Update vertex with rotated coordinates
                    vertices[i][0] = rotated_x + x
                    vertices[i][1] = rotated_y + y

        return vertices

    def calculate_oriented_bounding_box(self, position, size, rotation=None):
        """
        Calculate an oriented bounding box for an object with rotation.

        Returns a bounding box with orientation information that can be used
        for placement decisions and overlap checks.

        Args:
            position (list): [x, y, z] center position of the object
            size (list): [width, depth, height] dimensions of the object
            rotation (list, optional): [rx, ry, rz] rotation in degrees

        Returns:
            dict: Oriented bounding box with center, size, rotation, and vertices
        """
        # Calculate vertices of the rotated bounding box
        vertices = self.calculate_bounding_box_vertices(position, size, rotation)

        # Create the oriented bounding box
        obb = {
            "center": position.copy(),
            "size": size.copy(),
            "rotation": rotation.copy() if rotation else [0, 0, 0],
            "vertices": vertices,
        }

        return obb

    def get_axis_aligned_bounds(self, vertices):
        """
        Get the axis-aligned bounds from a set of vertices.

        This is useful for getting the min/max extents of a rotated bounding box.

        Args:
            vertices (list): List of [x, y, z] vertex coordinates

        Returns:
            tuple: (min_point, max_point) as [x, y, z] coordinates
        """
        # Initialize min and max points with the first vertex
        min_point = vertices[0].copy()
        max_point = vertices[0].copy()

        # Find min and max for each axis
        for vertex in vertices[1:]:
            for i in range(3):  # For x, y, z
                min_point[i] = min(min_point[i], vertex[i])
                max_point[i] = max(max_point[i], vertex[i])

        return min_point, max_point

    def create_floor(self):
        """
        Create a floor object based on configuration.

        The floor is always created at the origin (0,0,0) with dimensions
        specified in the configuration.

        Returns:
            dict: Floor object data with position, size, and vertices
        """
        floor_config = self.config["scene"]["Floor"]
        position = [0, 0, 0]  # Floor is always at origin
        size = floor_config["size"]
        rotation = floor_config.get("rotation", [0, 0, 0])

        # Calculate bounding box vertices with rotation
        vertices = self.calculate_bounding_box_vertices(position, size, rotation)

        return {
            "position": position,
            "size": size,
            "rotation": rotation,
            "asset_path": None,  # Floor doesn't have an asset file, created procedurally
            "scale_factor": 1.0,
            "vertices": vertices,
        }

    def create_object(self, obj_type):
        """
        Create an object with proper scaling and rotation at origin.

        This method:
        1. Gets a random asset for the object type
        2. Calculates the appropriate scale factor
        3. Applies the configured rotation
        4. Initializes the object at the origin (will be moved later)
        5. Calculates the rotated bounding box vertices

        Args:
            obj_type (str): Type of object to create

        Returns:
            dict: Object data with position, size, rotation, and asset info
        """
        obj_config = self.config["scene"][obj_type]
        position = [0, 0, 0]  # Initially place at origin

        # Get rotation from config (default to [0, 0, 0] if not specified)
        rotation = obj_config.get("rotation", [0, 0, 0])

        # Get random asset and calculate scale factor
        asset_path = self.get_random_asset(obj_type)
        scale_factor, true_size = self.calculate_scale_factor(asset_path, obj_config["size"])

        # Calculate bounding box vertices with rotation
        vertices = self.calculate_bounding_box_vertices(position, true_size, rotation)

        return {
            "position": position,
            "size": true_size,
            "rotation": rotation,
            "asset_path": asset_path,
            "scale_factor": scale_factor,
            "vertices": vertices,
        }


class PlacementGrid:
    """
    A grid system for tracking object placement on the floor.
    This class manages a 2D grid where each cell can be either empty (None) or occupied by an object ID.
    The grid uses center-origin coordinates to match the USD coordinate system.
    """

    def __init__(self, floor_size):
        """
        Initialize grid based on floor size.

        Args:
            floor_size (list): [x, y, z] dimensions of the floor
        """
        self.width = int(floor_size[0])
        self.length = int(floor_size[1])
        self.grid = [[None for _ in range(self.length)] for _ in range(self.width)]
        # Store offsets for converting between world and grid coordinates
        self.x_offset = self.width // 2
        self.y_offset = self.length // 2

    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices.
        Handles both individual coordinates and position lists.
        """
        if isinstance(x, list):
            x, y = x[0], x[1]
        return int(x + self.x_offset), int(y + self.y_offset)

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid indices back to world coordinates.
        """
        return grid_x - self.x_offset, grid_y - self.y_offset

    def _is_area_valid(self, center_x, center_y, half_width, half_length):
        """
        Check if an area is valid for object placement.
        Area must be within bounds and all cells must be unoccupied.
        """
        for x in range(center_x - half_width, center_x + half_width + 1):
            for y in range(center_y - half_length, center_y + half_length + 1):
                if not (0 <= x < self.width and 0 <= y < self.length):
                    return False
                if self.grid[x][y] is not None:
                    return False
        return True

    def mark_occupied(self, position, obj_size, obj_id, clearance=0):
        """
        Mark grid cells as occupied by an object including clearance buffer.
        """
        center_x, center_y = self.world_to_grid(position[0], position[1])
        # Add clearance to object size
        clearance_int = int(round(clearance))
        half_width = int(obj_size[0] / 2) + clearance_int
        half_length = int(obj_size[1] / 2) + clearance_int

        # Check bounds and occupancy
        if not self._is_area_valid(center_x, center_y, half_width, half_length):
            return False

        # Mark all cells as occupied
        for x in range(center_x - half_width, center_x + half_width + 1):
            for y in range(center_y - half_length, center_y + half_length + 1):
                self.grid[x][y] = obj_id

        return True

    def is_position_valid(self, position, obj_size, clearance=0):
        """
        Check if a position is valid for object placement including clearance.
        """
        center_x, center_y = self.world_to_grid(position[0], position[1])
        clearance_int = int(round(clearance))
        half_width = int(obj_size[0] / 2) + clearance_int
        half_length = int(obj_size[1] / 2) + clearance_int
        return self._is_area_valid(center_x, center_y, half_width, half_length)


class ObjectPlacement:
    """
    Handles object placement and overlap checking using a grid-based system.
    Combines exact geometric calculations with efficient grid-based tracking.
    """

    def __init__(self, floor_size):
        """
        Initialize placement handler.

        Args:
            floor_size (list): [x, y, z] dimensions of the floor
        """
        self.floor_size = floor_size
        self.placed_objects = []  # List of {position, size, id, clearance} dicts
        self.grid = PlacementGrid(floor_size)  # Initialize placement grid

    def check_bounds(self, position, obj_size, clearance=0):
        """
        Check if object and its clearance area are within floor bounds.

        Args:
            position (list): [x, y, z] center position to check
            obj_size (list): [x, y, z] size of object
            clearance (float): Clearance distance required around object

        Returns:
            bool: True if within bounds, False otherwise
        """
        # Add clearance to effective size
        effective_width = obj_size[0] + 2 * clearance
        effective_length = obj_size[1] + 2 * clearance

        half_width = effective_width / 2
        half_length = effective_length / 2

        # Check if object bounds plus clearance are within floor bounds
        min_x = position[0] - half_width
        max_x = position[0] + half_width
        min_y = position[1] - half_length
        max_y = position[1] + half_length

        within_bounds = (
            min_x >= -self.floor_size[0] / 2
            and max_x <= self.floor_size[0] / 2
            and min_y >= -self.floor_size[1] / 2
            and max_y <= self.floor_size[1] / 2
        )

        return within_bounds

    def check_overlap(self, position, obj_size, clearance=0, ignore_obj_id=None, placement_type=None):
        """
        Check if object overlaps with any placed objects, considering clearance.
        Uses grid-based system for initial quick check, then detailed geometry for special cases.

        Args:
            position (list): [x, y, z] center position to check
            obj_size (list): [x, y, z] size of object
            clearance (float): Minimum distance required between objects
            ignore_obj_id (str): ID of object to ignore when checking overlap
            placement_type (str): The placement type being used ("Long", "Short", "Top", "Under", "Tuck")

        Returns:
            bool: True if overlaps with any object, False otherwise
        """
        # Special handling for Tuck and Under placement types
        if placement_type in ["Tuck", "Under"]:
            # For these types, we use the existing geometric overlap check
            return self._check_geometric_overlap(position, obj_size, clearance, ignore_obj_id)

        # For all other cases, use the grid system
        return not self.grid.is_position_valid(position, obj_size, clearance)

    def _check_geometric_overlap(self, position, obj_size, clearance=0, ignore_obj_id=None):
        """
        Perform detailed geometric overlap check for special placement types.
        """
        obj_min_x = position[0] - obj_size[0] / 2 - clearance
        obj_max_x = position[0] + obj_size[0] / 2 + clearance
        obj_min_y = position[1] - obj_size[1] / 2 - clearance
        obj_max_y = position[1] + obj_size[1] / 2 + clearance

        for placed in self.placed_objects:
            if placed["id"] == ignore_obj_id:
                continue

            placed_min_x = placed["position"][0] - placed["size"][0] / 2 - clearance
            placed_max_x = placed["position"][0] + placed["size"][0] / 2 + clearance
            placed_min_y = placed["position"][1] - placed["size"][1] / 2 - clearance
            placed_max_y = placed["position"][1] + placed["size"][1] / 2 + clearance

            if not (
                obj_max_x < placed_min_x
                or obj_min_x > placed_max_x
                or obj_max_y < placed_min_y
                or obj_min_y > placed_max_y
            ):
                return True

        return False

    def is_position_valid(self, position, obj_size, clearance=0, ignore_obj_id=None, placement_type=None):
        """
        Check if a position is valid for object placement.
        Position is valid if within bounds and no overlap with other objects.

        Args:
            position (list): [x, y, z] center position to check
            obj_size (list): [x, y, z] size of object
            clearance (float): Minimum distance required between objects
            ignore_obj_id (str): ID of object to ignore when checking overlap
            placement_type (str): Type of placement ("Long", "Short", "Top", "Under", "Tuck")

        Returns:
            bool: True if position is valid, False otherwise
        """
        # Check bounds first (always required)
        if not self.check_bounds(position, obj_size, clearance):
            return False

        # Check overlap using appropriate method based on placement type
        if self.check_overlap(position, obj_size, clearance, ignore_obj_id, placement_type):
            return False

        return True

    def add_object(self, position, obj_size, obj_id, clearance=0, placement_type=None):
        """
        Add object to placed objects list and update the grid.
        Special handling for Tuck placement which allows 50% overlap with parent.

        Args:
            position (list): [x, y, z] center position of object
            obj_size (list): [x, y, z] size of object
            obj_id (str): Unique identifier for the object
            clearance (float): Clearance distance for this object
            placement_type (str): Type of placement ("Long", "Short", "Top", "Under", "Tuck")

        Returns:
            bool: True if successfully added, False otherwise
        """
        # For Tuck and Under placements, we only check geometric validity
        if placement_type in ["Tuck", "Under"]:
            if not self.check_bounds(position, obj_size):
                return False

            if self._check_geometric_overlap(position, obj_size, clearance, obj_id):
                return False

            self.placed_objects.append({"position": position, "size": obj_size, "id": obj_id, "clearance": clearance})
            return True

        # For all other placements, use the grid system
        if not self.grid.mark_occupied(position, obj_size, obj_id, clearance):
            return False

        self.placed_objects.append({"position": position, "size": obj_size, "id": obj_id, "clearance": clearance})
        return True


def get_random_valid_position(floor_size, obj_size, placement_handler, clearance=0, max_attempts=1000, retries=3):
    """
    Find a random valid position for an object on the floor.
    Uses a grid-based approach with rejection sampling and multiple retries.

    Args:
        floor_size (list): [x, y, z] size of the floor
        obj_size (list): [x, y, z] size of object to place
        placement_handler (ObjectPlacement): Current placement handler
        clearance (float): Clearance distance around object
        max_attempts (int): Maximum number of attempts per retry
        retries (int): Number of times to retry with different strategies

    Returns:
        list or None: [x, y, z] position if found, None if no valid position
    """
    # Create a placement grid for this attempt
    grid = PlacementGrid(floor_size)

    # Add existing objects to the grid
    for obj in placement_handler.placed_objects:
        grid.mark_occupied(obj["position"], obj["size"], obj["id"], obj.get("clearance", 0))

    # Calculate bounds for random position
    min_x = -floor_size[0] / 2 + obj_size[0] / 2
    max_x = floor_size[0] / 2 - obj_size[0] / 2
    min_y = -floor_size[1] / 2 + obj_size[1] / 2
    max_y = floor_size[1] / 2 - obj_size[1] / 2

    # Try different strategies
    for retry in range(retries):
        if retry == 0:
            # First try: Pure random sampling
            for _ in range(max_attempts):
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                pos = [x, y, 0]

                if grid.is_position_valid(pos, obj_size, clearance):
                    return pos

        elif retry == 1:
            # Second try: Grid-aligned sampling
            x_step = max(1, obj_size[0] + clearance)
            y_step = max(1, obj_size[1] + clearance)

            for _ in range(max_attempts):
                x = min_x + random.randint(0, int((max_x - min_x) / x_step)) * x_step
                y = min_y + random.randint(0, int((max_y - min_y) / y_step)) * y_step
                pos = [x, y, 0]

                if grid.is_position_valid(pos, obj_size, clearance):
                    return pos

        else:
            # Last try: Quadrant-based sampling
            quadrants = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            random.shuffle(quadrants)

            for qx, qy in quadrants:
                for _ in range(max_attempts // 4):
                    if qx > 0:
                        x = np.random.uniform(0, max_x)
                    else:
                        x = np.random.uniform(min_x, 0)

                    if qy > 0:
                        y = np.random.uniform(0, max_y)
                    else:
                        y = np.random.uniform(min_y, 0)

                    pos = [x, y, 0]
                    if grid.is_position_valid(pos, obj_size, clearance):
                        return pos

    return None
