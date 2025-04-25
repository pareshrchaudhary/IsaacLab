# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Position management module for procedural scene generation.

This module provides functionality for:
1. Creating and manipulating position objects
2. Generating circular arrangements of positions
3. Managing position references and parent-child relationships
4. Converting between coordinate systems

Positions are central to the placement system, providing a structured
way to track object locations and their relationships.
"""

import math
import numpy as np
import random

from pxr import Gf, Sdf, UsdGeom, UsdShade


class Position:
    """
    Represents a position in 3D space with additional metadata.

    This class encapsulates:
    - 3D coordinates (x, y, z)
    - Unique instance identifier
    - Parent reference information

    It forms the foundation for the placement system by tracking both
    spatial coordinates and relational data between objects.
    """

    def __init__(self, x, y, z, instance_id=None, parent_type=None, parent_instance_id=None):
        """
        Initialize a position with coordinates and optional metadata.

        Args:
            x (float): X-coordinate in world space
            y (float): Y-coordinate in world space
            z (float): Z-coordinate in world space
            instance_id (int, optional): Unique identifier for the object at this position
            parent_type (str, optional): Type of parent object
            parent_instance_id (int, optional): Instance ID of parent object
        """
        self.x = x
        self.y = y
        self.z = z
        self.instance_id = instance_id
        self.parent_type = parent_type
        self.parent_instance_id = parent_instance_id

    def __repr__(self):
        """
        Return string representation of the position.

        Provides a concise representation including coordinates and
        relationship information for debugging and logging.

        Returns:
            str: String representation of Position object
        """
        return (
            f"Position({self.x}, {self.y}, {self.z}, instance_id={self.instance_id},"
            f" parent={self.parent_type}_{self.parent_instance_id})"
        )


class PlacementGrid:
    """
    A grid system for tracking object placement on the floor.
    This class manages a 2D grid where each cell can be either empty (None) or occupied by an object ID.
    The grid uses center-origin coordinates to match the USD coordinate system.

    Features:
    - Tracks occupied and available positions
    - Handles clearance requirements
    - Provides coordinate conversion between world and grid space
    - Prevents object overlaps
    - Supports visualization of grid state
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

        Args:
            x (float or list): X coordinate or [x, y] position list
            y (float): Y coordinate (ignored if x is a list)

        Returns:
            tuple: (grid_x, grid_y) indices
        """
        if isinstance(x, list):
            x, y = x[0], x[1]
        return int(x + self.x_offset), int(y + self.y_offset)

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid indices back to world coordinates.

        Args:
            grid_x (int): Grid X index
            grid_y (int): Grid Y index

        Returns:
            tuple: (world_x, world_y) coordinates
        """
        return grid_x - self.x_offset, grid_y - self.y_offset

    def _is_area_valid(self, center_x, center_y, half_width, half_length):
        """
        Check if an area is valid for object placement.
        Area must be within bounds and all cells must be unoccupied.

        Args:
            center_x (int): Grid X coordinate of area center
            center_y (int): Grid Y coordinate of area center
            half_width (int): Half-width of area to check
            half_length (int): Half-length of area to check

        Returns:
            bool: True if area is valid, False otherwise
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

        Args:
            position (list): [x, y, z] center position of object
            obj_size (list): [x, y, z] size of object
            obj_id (str): Unique identifier for the object
            clearance (float): Clearance distance around object (rounded to nearest int)

        Returns:
            bool: True if successfully marked, False if any cell was occupied/out of bounds
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

        Args:
            position (list): [x, y, z] center position to check
            obj_size (list): [x, y, z] size of object
            clearance (float): Clearance distance around object (rounded to nearest int)

        Returns:
            bool: True if position is valid, False otherwise
        """
        center_x, center_y = self.world_to_grid(position[0], position[1])
        clearance_int = int(round(clearance))
        half_width = int(obj_size[0] / 2) + clearance_int
        half_length = int(obj_size[1] / 2) + clearance_int
        return self._is_area_valid(center_x, center_y, half_width, half_length)


def get_all_positions(floor_size):
    """
    Get all possible positions on the floor grid.

    Args:
        floor_size (list): [x, y, z] size of the floor

    Returns:
        list: List of all [x, y, z] positions on the grid
    """
    return [
        [x, y, 0]
        for x in range(-floor_size[0] // 2, floor_size[0] // 2 + 1)
        for y in range(-floor_size[1] // 2, floor_size[1] // 2 + 1)
    ]


def get_valid_floor_positions(floor_size, obj_size, grid, visualize=False):
    """
    Generate all valid positions for an object that has Floor as parent.

    Args:
        floor_size (list): [x, y, z] size of the floor
        obj_size (list): [x, y, z] size of object to place
        grid (PlacementGrid): Current placement grid
        visualize (bool): Whether to return visualization bounds

    Returns:
        list: List of valid positions
        list: (Optional) List of visualization bounds if visualize=True
    """
    valid_positions = []
    viz_bounds = [] if visualize else None

    # Calculate valid range accounting for object size
    x_min = -floor_size[0] / 2 + obj_size[0] / 2
    x_max = floor_size[0] / 2 - obj_size[0] / 2
    y_min = -floor_size[1] / 2 + obj_size[1] / 2
    y_max = floor_size[1] / 2 - obj_size[1] / 2

    # Try each grid position
    for x in range(int(x_min), int(x_max) + 1):
        for y in range(int(y_min), int(y_max) + 1):
            pos = [x, y, 0]
            if grid.is_position_valid(pos, obj_size):
                valid_positions.append(pos)
                if visualize:
                    viz_bounds.append({"center": pos, "size": obj_size})

    return (valid_positions, viz_bounds) if visualize else valid_positions


def get_random_valid_position(floor_size, obj_size, grid, clearance=0, max_attempts=1000, retries=100):
    """
    Find a random valid position for an object on the floor.
    Uses rejection sampling with multiple retries to find a valid position.

    Args:
        floor_size (list): [x, y, z] size of the floor
        obj_size (list): [x, y, z] size of object to place
        grid (PlacementGrid): Current placement grid
        clearance (float): Clearance distance around object
        max_attempts (int): Maximum number of attempts per retry
        retries (int): Number of times to retry finding a valid position

    Returns:
        list or None: [x, y, z] position if found, None if no valid position
    """
    for _ in range(retries):
        for _ in range(max_attempts):
            x = int(np.random.randint(-floor_size[0] // 2, floor_size[0] // 2))
            y = int(np.random.randint(-floor_size[1] // 2, floor_size[1] // 2))
            pos = [x, y, 0]

            if grid.is_position_valid(pos, obj_size, clearance):
                return pos

    return None


def circle_positions(center, radius, count, obj_type, start_idx=0):
    """
    Generate positions arranged in a circle around a center point.

    Creates evenly spaced positions around a center point, ideal for
    arranging objects in a radial pattern around a parent object.

    Args:
        center (Position): Center position of the circle
        radius (float): Radius of the circle
        count (int): Number of positions to generate
        obj_type (str): Type of object to be placed at these positions
        start_idx (int, optional): Starting index for instance IDs

    Returns:
        list: List of Position objects arranged in a circle
    """
    positions = []

    # Generate positions evenly spaced around the circle
    for i in range(count):
        # Calculate angle in radians
        angle = 2 * math.pi * i / count

        # Calculate position using polar coordinates
        x = center.x + radius * math.cos(angle)
        z = center.z + radius * math.sin(angle)

        # Use parent's y-coordinate (same elevation)
        y = center.y

        # Create position with sequential instance ID
        pos = Position(x, y, z, start_idx + i)
        positions.append(pos)

    return positions


def get_circle_positions_on_floor(radius, count, obj_type, start_idx=0):
    """
    Generate positions arranged in a circle on the floor.

    Creates a special case of circle positions centered at the origin
    (floor center), typically used for floor-based objects.

    Args:
        radius (float): Radius of the circle
        count (int): Number of positions to generate
        obj_type (str): Type of object to be placed at these positions
        start_idx (int, optional): Starting index for instance IDs

    Returns:
        list: List of Position objects arranged in a circle on the floor
    """
    # Create a center position at the floor origin
    center = Position(0, 0, 0, 0)

    # Use the standard circle_positions function with floor center
    return circle_positions(center, radius, count, obj_type, start_idx)


def random_position_in_range(x_range, y_range, z_range, obj_type, instance_id=0):
    """
    Generate a random position within specified coordinate ranges.

    Useful for creating random placements within a bounded volume,
    with controlled randomness within the specified ranges.

    Args:
        x_range (tuple): (min_x, max_x) range for x-coordinate
        y_range (tuple): (min_y, max_y) range for y-coordinate
        z_range (tuple): (min_z, max_z) range for z-coordinate
        obj_type (str): Type of object to be placed at this position
        instance_id (int, optional): Instance ID for the position

    Returns:
        Position: Randomly generated position within specified ranges
    """
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    z = random.uniform(z_range[0], z_range[1])

    return Position(x, y, z, instance_id)


def grid_positions(x_start, x_end, z_start, z_end, y, spacing, obj_type, start_idx=0):
    """
    Generate positions arranged in a grid pattern.

    Creates a regular grid of positions with specified spacing,
    which is useful for placing objects in organized layouts.

    Args:
        x_start (float): Starting x-coordinate
        x_end (float): Ending x-coordinate
        z_start (float): Starting z-coordinate
        z_end (float): Ending z-coordinate
        y (float): Fixed y-coordinate (elevation) for all positions
        spacing (float): Distance between adjacent positions
        obj_type (str): Type of object to be placed at these positions
        start_idx (int, optional): Starting index for instance IDs

    Returns:
        list: List of Position objects arranged in a grid
    """
    positions = []
    idx = start_idx

    # Calculate number of positions in each dimension
    x_count = max(1, int((x_end - x_start) / spacing) + 1)
    z_count = max(1, int((z_end - z_start) / spacing) + 1)

    # Adjust spacing to fill the area evenly
    x_spacing = (x_end - x_start) / (x_count - 1) if x_count > 1 else 0
    z_spacing = (z_end - z_start) / (z_count - 1) if z_count > 1 else 0

    # Generate grid of positions
    for i in range(x_count):
        x = x_start + i * x_spacing
        for j in range(z_count):
            z = z_start + j * z_spacing
            positions.append(Position(x, y, z, idx))
            idx += 1

    return positions
