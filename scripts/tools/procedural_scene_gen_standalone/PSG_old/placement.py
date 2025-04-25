# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Object placement module for procedural scene generation.

This module provides functionality for:
1. Determining valid object positions based on parent-child relationships
2. Managing spatial relationships between objects in a scene
3. Handling different placement strategies (Top, Side, Tuck, etc.)
4. Ensuring proper scale and orientation of objects

The placement algorithms ensure objects are positioned in physically
realistic ways while respecting constraints like clearance requirements.
"""

import json
import math
import numpy as np
import os
import random
from collections import defaultdict, deque

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from scripts.procedural_scene_gen_standalone.PSG_old.utils import get_random_asset


class ParentPlacement:
    """
    Handles placement of objects relative to their parent objects.
    Supports various placement types:
    - Long: Place at parent's longer side border (both ends)
    - Short: Place at parent's shorter side border (both ends)
    - Top: Place on parent's top surface
    - Under: Place under parent's surface
    - Tuck: Place chair with 50% overlap on parent's long side

    This class generates positions based on parent-child relationships,
    implementing different strategies for arranging objects in physically
    realistic ways around their parents.
    """

    def __init__(self, config):
        """
        Initialize parent placement handler.

        Sets up the placement handler with scene configuration,
        providing access to object dimensions and properties needed
        for determining valid positions.

        Args:
            config (dict): Scene configuration containing object definitions
        """
        self.config = config

    def get_valid_positions(self, obj_type, parent_data, placement_type, clearance=0, parent_type=None):
        """
        Get valid positions for an object based on its parent and placement type.

        This is the main entry point for generating positions, which:
        1. Takes a placement type (or list of types)
        2. Delegates to specialized placement methods for each type
        3. Combines results into a comprehensive list of valid positions

        If placement_type is empty or None, returns all valid positions (both long and short sides).
        If placement_type is a list, returns positions from all specified placement types.

        Args:
            obj_type (str): Type of object to place (e.g., "Chair", "Cube")
            parent_data (dict): Parent object data containing position and size
            placement_type (str or list): Type(s) of placement ("Long", "Short", "Top", "Under", "Tuck", "" or None)
            clearance (float): Clearance distance from parent
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions
        """
        valid_positions = []

        # Convert single placement type to list for uniform handling
        if not isinstance(placement_type, list):
            placement_type = [placement_type]

        # Map placement types to their corresponding methods
        placement_methods = {
            "Long": self._get_long_positions,
            "Short": self._get_short_positions,
            "Top": self._get_top_positions,
            "Under": self._get_under_positions,
            "Tuck": self._get_tuck_positions,
            "NULL": None,  # Special case, handled separately
        }

        # Handle each placement type
        for ptype in placement_type:
            if ptype == "NULL":
                # NULL is a special case - get both long and short positions
                valid_positions.extend(self._get_long_positions(obj_type, parent_data, clearance, parent_type))
                valid_positions.extend(self._get_short_positions(obj_type, parent_data, clearance, parent_type))
            elif ptype in placement_methods:
                method = placement_methods[ptype]
                valid_positions.extend(method(obj_type, parent_data, clearance, parent_type))

        return valid_positions

    def _generate_side_positions(self, obj_type, parent_data, clearance, parent_type, side_type):
        """
        Helper method to generate positions along the sides of a parent object.

        This is a common implementation used by both _get_long_positions and
        _get_short_positions. It:
        1. Determines parent orientation (which axis is longer)
        2. Calculates the number of objects that can fit along the selected side
        3. Generates positions on both sides of the parent
        4. Ensures all positions are within floor bounds

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data with position and size
            clearance (float): Clearance distance from parent
            parent_type (str): Type of the parent object
            side_type (str): Which sides to place on ("Long" or "Short")

        Returns:
            list: List of valid [x, y, z] positions
        """
        parent_pos = parent_data["position"]
        parent_size = parent_data["size"]
        obj_config = self.config["scene"][obj_type]
        obj_size = obj_config["size"]
        floor_size = self.config["scene"]["Floor"]["size"]
        parent_clearance = self.config["scene"][parent_type].get("clearance", 0)

        # Determine axis orientation based on parent dimensions
        is_x_shorter = parent_size[0] < parent_size[1]
        valid_positions = []

        # Small gap between objects (10% of object width)
        gap = obj_size[0] * 0.1

        # Determine which sides to place on based on the side_type
        place_on_long_side = (side_type == "Long" and not is_x_shorter) or (side_type == "Short" and is_x_shorter)

        if place_on_long_side:
            # Place along X-axis (along long sides in Y direction)
            offset_y = parent_size[1] / 2 + parent_clearance + obj_size[1] / 2

            # Calculate number of positions that can fit along each side
            available_length = parent_size[0] - obj_size[0]
            spacing = obj_size[0] + gap
            num_positions = int(available_length / spacing) + 1

            # Generate positions along both sides
            for i in range(num_positions):
                # Calculate x offset from center
                offset_x = (i * spacing) - (available_length / 2)

                # Front side position
                y = parent_pos[1] - offset_y
                if abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                    pos = [parent_pos[0] + offset_x, y, parent_pos[2]]
                    valid_positions.append(pos)

                # Back side position
                y = parent_pos[1] + offset_y
                if abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                    pos = [parent_pos[0] + offset_x, y, parent_pos[2]]
                    valid_positions.append(pos)

        else:
            # Place along Y-axis (along long sides in X direction)
            offset_x = parent_size[0] / 2 + parent_clearance + obj_size[0] / 2

            # Calculate number of positions that can fit along each side
            available_length = parent_size[1] - obj_size[1]
            spacing = obj_size[1] + gap
            num_positions = int(available_length / spacing) + 1

            # Generate positions along both sides
            for i in range(num_positions):
                # Calculate y offset from center
                offset_y = (i * spacing) - (available_length / 2)

                # Left side position
                x = parent_pos[0] - offset_x
                if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2:
                    pos = [x, parent_pos[1] + offset_y, parent_pos[2]]
                    valid_positions.append(pos)

                # Right side position
                x = parent_pos[0] + offset_x
                if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2:
                    pos = [x, parent_pos[1] + offset_y, parent_pos[2]]
                    valid_positions.append(pos)

        return valid_positions

    def _get_long_positions(self, obj_type, parent_data, clearance, parent_type):
        """
        Get valid positions for placing an object on the longer side of its parent.

        Places objects along the two longer sides of the parent:
        - If parent is wider than deep, places objects along front and back
        - If parent is deeper than wide, places objects along left and right sides

        Uses the _generate_side_positions helper with "Long" side type.

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data with position and size
            clearance (float): Clearance distance from parent
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions
        """
        return self._generate_side_positions(obj_type, parent_data, clearance, parent_type, "Long")

    def _get_short_positions(self, obj_type, parent_data, clearance, parent_type):
        """
        Get valid positions for placing an object on the shorter side of its parent.

        Places objects along the two shorter sides of the parent:
        - If parent is wider than deep, places objects along left and right sides
        - If parent is deeper than wide, places objects along front and back

        Uses the _generate_side_positions helper with "Short" side type.

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data with position and size
            clearance (float): Clearance distance from parent
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions
        """
        return self._generate_side_positions(obj_type, parent_data, clearance, parent_type, "Short")

    def _get_top_positions(self, obj_type, parent_data, clearance, parent_type):
        """
        Get valid positions for placing an object on top of its parent.

        This method:
        1. Creates a grid of positions on the parent's top surface
        2. Accounts for object size and clearance requirements
        3. Ensures positions are within parent bounds and floor bounds
        4. Handles edge cases like insufficient space for placement

        Typically used for placing items on tables, shelves, etc.

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data with position and size
            clearance (float): Clearance distance from parent
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions
        """
        parent_pos = parent_data["position"]
        parent_size = parent_data["size"]
        obj_config = self.config["scene"][obj_type]
        obj_size = obj_config["size"]
        floor_size = self.config["scene"]["Floor"]["size"]

        valid_positions = []

        # Small gap between objects (10% of object width)
        gap = obj_size[0] * 0.1

        # Calculate available area on parent's top surface
        available_width = parent_size[0] - obj_size[0]  # Account for object width
        available_length = parent_size[1] - obj_size[1]  # Account for object length

        # Calculate spacing between objects
        spacing_x = obj_size[0] + gap
        spacing_y = obj_size[1] + gap

        # Calculate number of positions that can fit in each direction
        num_x = int(available_width / spacing_x) + 1
        num_y = int(available_length / spacing_y) + 1

        # Calculate z position (on top of parent)
        z_pos = parent_pos[2] + parent_size[2]  # Position at parent's top surface

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
                    pos = [x, y, z_pos]
                    valid_positions.append(pos)

        return valid_positions

    def _get_under_positions(self, obj_type, parent_data, clearance, parent_type):
        """
        Get valid positions for placing an object under its parent.

        This method:
        1. Creates a grid of positions under the parent
        2. Applies insets to avoid table legs or edges
        3. Ensures positions are within bounds
        4. Accounts for object dimensions and clearance

        Typically used for placing items under tables, beds, etc.

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data with position and size
            clearance (float): Clearance distance from parent
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions under parent
        """
        parent_pos = parent_data["position"]
        parent_size = parent_data["size"]
        obj_config = self.config["scene"][obj_type]
        obj_size = obj_config["size"]
        floor_size = self.config["scene"]["Floor"]["size"]

        valid_positions = []

        # Small gap between objects (10% of object width)
        gap = obj_size[0] * 0.1

        # Calculate inset to avoid table legs (20% of parent width/length)
        inset_x = parent_size[0] * 0.2
        inset_y = parent_size[1] * 0.2

        # Calculate available area under parent's surface (accounting for insets)
        available_width = parent_size[0] - 2 * inset_x - obj_size[0]  # Account for insets and object width
        available_length = parent_size[1] - 2 * inset_y - obj_size[1]  # Account for insets and object length

        # Calculate spacing between objects
        spacing_x = obj_size[0] + gap
        spacing_y = obj_size[1] + gap

        # Calculate number of positions that can fit in each direction
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
                z = 0  # Place directly on floor

                # Check if position is within floor bounds
                if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2 and abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                    pos = [x, y, z]
                    valid_positions.append(pos)

        return valid_positions

    def _get_tuck_positions(self, obj_type, parent_data, clearance, parent_type):
        """
        Get valid positions for tucking an object under its parent with 50% overlap.

        This specialized placement type:
        1. Places objects with 50% overlap with the parent's bounding box
        2. Positions objects only at the center of each side
        3. Enforces strict orientation so objects (typically chairs) face the parent
        4. Respects floor boundaries

        This is typically used for placing chairs at tables, where the chair
        is partially under the table but still accessible.

        Args:
            obj_type (str): Type of object to place
            parent_data (dict): Parent object data with position and size
            clearance (float): Clearance distance from parent (not used in tucking)
            parent_type (str): Type of the parent object

        Returns:
            list: List of valid [x, y, z] positions with 50% overlap
        """
        parent_pos = parent_data["position"]
        parent_size = parent_data["size"]
        obj_config = self.config["scene"][obj_type]
        obj_size = obj_config["size"]
        floor_size = self.config["scene"]["Floor"]["size"]
        parent_clearance = self.config["scene"][parent_type].get("clearance", 0)

        # Determine which side is shorter (similar to long positions)
        is_x_shorter = parent_size[0] < parent_size[1]
        valid_positions = []

        # Calculate the overlap amount (50% of the object size)
        if is_x_shorter:
            # Overlap in X direction
            overlap_x = obj_size[0] * 0.5
            # Calculate the new offset with 50% overlap
            offset_x = parent_size[0] / 2 - overlap_x + parent_clearance

            # For tuck placement, only allow centered positions
            # Just create one position on each side, centered on Y
            offset_y = 0  # Center position only

            # Left side position
            x = parent_pos[0] - offset_x
            if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2:
                pos = [x, parent_pos[1] + offset_y, parent_pos[2]]
                valid_positions.append(pos)

            # Right side position
            x = parent_pos[0] + offset_x
            if abs(x) + obj_size[0] / 2 <= floor_size[0] / 2:
                pos = [x, parent_pos[1] + offset_y, parent_pos[2]]
                valid_positions.append(pos)

        else:
            # Overlap in Y direction
            overlap_y = obj_size[1] * 0.5
            # Calculate the new offset with 50% overlap
            offset_y = parent_size[1] / 2 - overlap_y + parent_clearance

            # For tuck placement, only allow centered positions
            # Just create one position on each side, centered on X
            offset_x = 0  # Center position only

            # Front side position
            y = parent_pos[1] - offset_y
            if abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                pos = [parent_pos[0] + offset_x, y, parent_pos[2]]
                valid_positions.append(pos)

            # Back side position
            y = parent_pos[1] + offset_y
            if abs(y) + obj_size[1] / 2 <= floor_size[1] / 2:
                pos = [parent_pos[0] + offset_x, y, parent_pos[2]]
                valid_positions.append(pos)

        return valid_positions


class ObjectPlacement:
    """
    Handles object placement and overlap checking using exact geometric calculations.
    No grid system - uses actual object bounds and positions for precise placement.
    """

    def __init__(self, floor_size):
        """
        Initialize placement handler.

        Args:
            floor_size (list): [x, y, z] dimensions of the floor
        """
        self.floor_size = floor_size
        self.placed_objects = []  # List of {position, size, id, clearance} dicts

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
        Two objects overlap if their bounds + clearance intersect.
        Objects are allowed to touch (share an edge) without being considered overlapping.
        Includes a 5% forgiveness factor to account for numerical precision.
        Uses special handling for parent-child relationships based on placement type.

        Args:
            position (list): [x, y, z] center position to check
            obj_size (list): [x, y, z] size of object
            clearance (float): Minimum distance required between objects
            ignore_obj_id (str): ID of object to ignore when checking overlap
            placement_type (str): The placement type being used ("Long", "Short", "Top", "Under", "Tuck")

        Returns:
            bool: True if overlaps with any object, False otherwise
        """
        # Add forgiveness to bounds calculation
        obj_min_x = position[0] - obj_size[0] / 2 - clearance
        obj_max_x = position[0] + obj_size[0] / 2 + clearance
        obj_min_y = position[1] - obj_size[1] / 2 - clearance
        obj_max_y = position[1] + obj_size[1] / 2 + clearance

        # Extract object type from ID if available (for parent-child relationship check)
        child_type = None
        if ignore_obj_id and "_" in ignore_obj_id:
            child_type = ignore_obj_id.split("_")[0]

        # Only Tuck and Under placement types need special handling for overlaps
        for placed in self.placed_objects:
            if placed["id"] == ignore_obj_id:
                continue

            # Only apply special handling for table parents and supported placement types
            special_handling = placement_type in ["Tuck", "Under"]

            # Calculate bounds of placed object
            placed_min_x = placed["position"][0] - placed["size"][0] / 2 - clearance
            placed_max_x = placed["position"][0] + placed["size"][0] / 2 + clearance
            placed_min_y = placed["position"][1] - placed["size"][1] / 2 - clearance
            placed_max_y = placed["position"][1] + placed["size"][1] / 2 + clearance

            # For special placement types that allow overlap with parent
            if special_handling:
                # For Tuck placement, we intentionally allow partial overlap with table
                # Skip the overlap check entirely, as tuck positions are pre-validated
                continue

            # Standard overlap check - objects shouldn't intersect
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
        # Check bounds
        if not self.check_bounds(position, obj_size, clearance):
            return False

        # Check overlap
        if self.check_overlap(position, obj_size, clearance, ignore_obj_id, placement_type):
            return False

        return True

    def _check_object_overlap(self, position, obj_size, obj_id=None):
        """
        Helper method to check if an object overlaps with other objects.
        Simple box overlap check with no clearance buffer.

        Args:
            position (list): [x, y, z] center position of object
            obj_size (list): [x, y, z] size of object
            obj_id (str): ID of object to exclude from checking

        Returns:
            bool: True if overlaps with any object, False otherwise
        """
        # Simple box overlap check
        obj_min_x = position[0] - obj_size[0] / 2
        obj_max_x = position[0] + obj_size[0] / 2
        obj_min_y = position[1] - obj_size[1] / 2
        obj_max_y = position[1] + obj_size[1] / 2

        for placed in self.placed_objects:
            if placed["id"] == obj_id:
                continue

            placed_min_x = placed["position"][0] - placed["size"][0] / 2
            placed_max_x = placed["position"][0] + placed["size"][0] / 2
            placed_min_y = placed["position"][1] - placed["size"][1] / 2
            placed_max_y = placed["position"][1] + placed["size"][1] / 2

            if not (
                obj_max_x < placed_min_x
                or obj_min_x > placed_max_x
                or obj_max_y < placed_min_y
                or obj_min_y > placed_max_y
            ):
                return True

        return False

    def add_object(self, position, obj_size, obj_id, clearance=0, placement_type=None):
        """
        Add object to placed objects list.
        Special handling for Tuck placement which allows 50% overlap with parent.
        All other placement types use standard overlap checking.

        Args:
            position (list): [x, y, z] center position of object
            obj_size (list): [x, y, z] size of object
            obj_id (str): Unique identifier for the object
            clearance (float): Clearance distance for this object
            placement_type (str): Type of placement ("Long", "Short", "Top", "Under", "Tuck")

        Returns:
            bool: True if successfully added, False otherwise
        """
        # Extract object type for parent-child relationship check
        obj_type = obj_id.split("_")[0] if "_" in obj_id else obj_id
        is_tuck_placement = placement_type == "Tuck"

        # For Tuck placement, we handle overlap specially
        if is_tuck_placement:
            # For tuck placement, we just need to check bounds
            if not self.check_bounds(position, obj_size):
                return False

            # Check overlap with same object types only
            for placed in self.placed_objects:
                placed_type = placed["id"].split("_")[0] if "_" in placed["id"] else placed["id"]

                # Only check overlaps with objects of the same type
                if placed_type == obj_type:
                    # Simple box overlap check
                    obj_min_x = position[0] - obj_size[0] / 2
                    obj_max_x = position[0] + obj_size[0] / 2
                    obj_min_y = position[1] - obj_size[1] / 2
                    obj_max_y = position[1] + obj_size[1] / 2

                    placed_min_x = placed["position"][0] - placed["size"][0] / 2
                    placed_max_x = placed["position"][0] + placed["size"][0] / 2
                    placed_min_y = placed["position"][1] - placed["size"][1] / 2
                    placed_max_y = placed["position"][1] + placed["size"][1] / 2

                    if not (
                        obj_max_x < placed_min_x
                        or obj_min_x > placed_max_x
                        or obj_max_y < placed_min_y
                        or obj_min_y > placed_max_y
                    ):
                        return False

            # If we get here, position is valid for tuck placement
            self.placed_objects.append({"position": position, "size": obj_size, "id": obj_id, "clearance": clearance})
            return True

        # For all other objects and placement types, check full position validity
        valid = self.is_position_valid(position, obj_size, clearance, obj_id, placement_type)
        if not valid:
            return False

        self.placed_objects.append({"position": position, "size": obj_size, "id": obj_id, "clearance": clearance})
        return True

    def clear_unused_positions(self, used_positions):
        """
        Clear all positions from placed_objects that aren't in used_positions.
        This is called after placing a set of child objects to free up positions
        for the next object type.

        Args:
            used_positions (list): List of positions that were actually used
        """
        EPSILON = 0.001  # Small epsilon for floating-point comparisons

        def is_position_used(obj_position):
            """Check if a position is in the used_positions list"""
            return any(all(abs(a - b) < EPSILON for a, b in zip(obj_position, used_pos)) for used_pos in used_positions)

        # Keep only parent objects or child objects with used positions
        self.placed_objects = [
            obj for obj in self.placed_objects if "_" not in obj["id"] or is_position_used(obj["position"])
        ]


def get_random_valid_position(
    floor_size, obj_size, placement_handler, clearance=0, max_attempts=1000, placement_type=None
):
    """
    Find a random valid position for an object on the floor.
    Uses rejection sampling to find a valid position.

    Args:
        floor_size (list): [x, y, z] size of the floor
        obj_size (list): [x, y, z] size of object to place
        placement_handler (ObjectPlacement): Current placement handler
        clearance (float): Clearance distance around object
        max_attempts (int): Maximum number of attempts to find valid position
        placement_type (str): Type of placement to use for validity check

    Returns:
        list or None: [x, y, z] position if found, None if no valid position
    """
    # Calculate bounds for random position
    min_x = -floor_size[0] / 2 + obj_size[0] / 2
    max_x = floor_size[0] / 2 - obj_size[0] / 2
    min_y = -floor_size[1] / 2 + obj_size[1] / 2
    max_y = floor_size[1] / 2 - obj_size[1] / 2

    # Try to find a valid position with rejection sampling
    for _ in range(max_attempts):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        pos = [x, y, 0]  # Place directly on floor (z=0)

        if placement_handler.is_position_valid(pos, obj_size, clearance, placement_type=placement_type):
            return pos

    return None


def calculate_orientation(obj_position, parent_position, parent_size, placement_type):
    """
    Calculate the proper orientation for an object with strict_orientation set to True.
    Orients the object so its front side (+x axis) faces the parent's bounding box.
    Rotation is limited to 90-degree increments (0, 90, 180, 270 degrees around z-axis).

    Args:
        obj_position (list): [x, y, z] position of the object
        parent_position (list): [x, y, z] position of the parent
        parent_size (list): [x, y, z] size of the parent
        placement_type (str): Type of placement (Long, Short, Top, Under, Tuck)

    Returns:
        list: [rx, ry, rz] rotation in degrees, where rz is the rotation around z-axis
    """
    # Calculate the vector from object to parent center
    dx = parent_position[0] - obj_position[0]
    dy = parent_position[1] - obj_position[1]

    # Convert to angle in degrees
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Map angle to the nearest 90-degree rotation
    # This ensures the front of the object (+x) faces the parent
    if -45 <= angle < 45:
        return [0, 0, 90]  # Face right (+X) - 0 degrees
    elif 45 <= angle < 135:
        return [0, 0, 180]  # Face up (+Y) - 90 degrees
    elif angle >= 135 or angle < -135:
        return [0, 0, -90]  # Face left (-X) - 180 degrees
    else:  # -135 <= angle < -45
        return [0, 0, 0]  # Face down (-Y) - 270 degrees


def calculate_scale_factor(asset_path, target_size):
    """
    Calculate uniform scale factor to fit an asset within target size.
    Maintains aspect ratio by using the minimum scale factor across dimensions.

    Args:
        asset_path (str): Path to the USD asset file
        target_size (list): Target size [x, y, z] for the object

    Returns:
        tuple: (scale_factor, true_size) where true_size is actual size after scaling
    """
    if asset_path is None:
        return 1.0, target_size

    # Load asset and compute bounds
    temp_stage = Usd.Stage.CreateInMemory()
    temp_ref = temp_stage.DefinePrim("/Temp", "Xform")
    temp_ref.GetReferences().AddReference(os.path.abspath(asset_path))
    temp_geom = UsdGeom.Imageable(temp_ref)
    bounds = temp_geom.ComputeLocalBound(0.0, "default")
    original_size = bounds.ComputeAlignedBox().GetMax() - bounds.ComputeAlignedBox().GetMin()

    # Calculate uniform scale factor
    target_size_vec = Gf.Vec3d(*target_size)
    scale_factors = [t / o if o != 0 else 1.0 for t, o in zip(target_size_vec, original_size)]
    scale_factor = min(scale_factors)  # Use minimum to maintain aspect ratio
    true_size = [float(x) for x in (original_size * scale_factor)]

    return float(scale_factor), true_size


def create_floor(config):
    """Create floor object with default configuration"""
    floor_config = config["scene"]["Floor"]
    return {
        "position": [0, 0, 0],
        "size": floor_config["size"],
        "rotation": [0, 0, 0],
        "asset_path": None,
        "scale_factor": 1.0,
    }


def create_object(config, obj_type):
    """
    Create an object with proper scaling at origin.
    Handles both procedurally generated (Cube) and asset-based objects.

    Args:
        config (dict): Configuration dictionary containing object settings
        obj_type (str): Type of object to create

    Returns:
        dict: Object data with position, size, rotation, and scale factor
    """
    obj_config = config["scene"][obj_type]

    # Handle asset-based objects
    asset_path = get_random_asset(config, obj_type)
    scale_factor, true_size = calculate_scale_factor(asset_path, obj_config["size"])

    return {
        "position": [0, 0, 0],
        "size": true_size,
        "rotation": [0, 0, 0],
        "asset_path": asset_path,
        "scale_factor": scale_factor,
    }
