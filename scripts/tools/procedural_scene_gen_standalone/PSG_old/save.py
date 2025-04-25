# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Scene saving module for procedural scene generation.

This module provides functionality for:
1. Saving generated scenes to USD files
2. Creating proper parent-child hierarchies in USD
3. Setting up transformations (position, rotation, scale)
4. Configuring references to USD model assets

The saving process ensures all generated scenes are properly persisted
with the correct transformations and references for visualization.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
from matplotlib.patches import FancyArrowPatch, Patch

import networkx as nx
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from scripts.procedural_scene_gen_standalone.PSG_old.utils import compute_node_heights

from omni.isaac.lab.utils.assets import NVIDIA_NUCLEUS_DIR


class SaveManager:
    """
    Handles saving of scene data, USD files, and visualizations.
    This class manages:
    1. Scene data serialization and storage
    2. USD scene file creation with proper object placement
    3. Visualization of parent-child relationships

    The save process ensures that:
    - All scene data is properly serialized
    - USD files maintain correct object hierarchies
    - Visualizations are clear and informative
    """

    def __init__(self, config):
        """
        Initialize save manager with configuration.
        Creates output directory if it doesn't exist.

        Args:
            config (dict): Scene configuration from YAML containing:
                         - Output settings (save directory, run name)
                         - Save options (USD, JSON, visualizations)
        """
        self.config = config
        self.save_dir = os.path.join(config["output"]["save_dir"], config["output"]["run_name"])
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize texture settings
        self.texture_enabled = config.get("textures", {}).get("enabled", False)
        self.texture_paths = config.get("textures", {}).get("materials", {})
        self.object_materials = config.get("textures", {}).get("object_materials", {})
        self.material_properties = config.get("textures", {}).get("material_properties", {})

    def save_positional_info(self, scene_data, scene_number):
        """
        Save scene data as JSON file.

        Args:
            scene_data (dict): Complete scene data with object placements
            scene_number (int): Scene identifier
        """
        filepath = os.path.join(self.save_dir, f"scene_{scene_number + 1}.json")
        with open(filepath, "w") as f:
            json.dump(scene_data, f, indent=4)

    def save_scene_graph(self):
        """
        Save visualization of the scene graph.
        This is a placeholder for future implementation.

        Returns:
            int: 0 (placeholder)
        """
        return 0

    def save_parent_child_graph(self, graph, filename="parent_child_graph.png"):
        """
        Save visualization of parent-child relationships.
        Creates a hierarchical graph with color-coded height levels.

        Args:
            graph (nx.DiGraph): NetworkX graph of parent-child relationships
            filename (str): Output filename for the visualization
        """
        plt.figure(figsize=(14, 10))
        ax = plt.gca()
        ax.set_aspect("equal")

        # Calculate layout and heights
        pos = nx.spring_layout(graph, k=1, iterations=100)
        heights = compute_node_heights(graph)
        max_height = max(heights.values())

        # Define colors for height levels
        pastel_colors = [
            "#aec6cf",
            "#ffb347",
            "#77dd77",
            "#f49ac2",
            "#cfcfc4",
            "#b39eb5",
            "#ff6961",
            "#fdfd96",
            "#84b6f4",
            "#fdcae1",
        ][: max_height + 1]

        # Draw nodes with height-based colors
        node_colors = [pastel_colors[heights[n]] for n in graph.nodes()]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, edgecolors="black", node_size=1500, linewidths=1.5)
        nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black", font_weight="bold")

        # Draw edges with arrows
        node_radius = 0.08
        for u, v in graph.edges():
            # Calculate arrow start/end points
            x_start, y_start = pos[u]
            x_end, y_end = pos[v]
            dx, dy = x_end - x_start, y_end - y_start
            length = (dx**2 + dy**2) ** 0.5

            # Adjust points to start/end at node boundaries
            x_start += node_radius * dx / length
            y_start += node_radius * dy / length
            x_end -= node_radius * dx / length
            y_end -= node_radius * dy / length

            # Add arrow
            arrow = FancyArrowPatch(
                (x_start, y_start),
                (x_end, y_end),
                arrowstyle="-|>",
                color="black",
                linewidth=1.2,
                mutation_scale=15,
                connectionstyle="arc3",
            )
            ax.add_patch(arrow)

        # Add legend for height levels
        legend_handles = [Patch(color=pastel_colors[i], label=f"Height {i}") for i in range(max_height + 1)]
        ax.legend(
            handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=max_height + 1, frameon=False
        )

        plt.title("Parent-Child Relationship Graph (by Height)", fontsize=14)
        plt.axis("off")
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight", dpi=300)
        plt.close()

    def apply_random_texture(self, prim_path, obj_type, stage, is_cube=False):
        """
        Apply a random texture to a given USD primitive based on its object type.

        Args:
            prim_path (str): USD path of the primitive
            obj_type (str): Type of object (e.g., "Table", "Chair")
            stage (Usd.Stage): The current USD stage
            is_cube (bool): Whether the primitive is a cube (special UV handling)
        """
        if not self.texture_enabled:
            return

        # Get material type for the object
        material_type = self.object_materials.get(obj_type, "Wood")
        if material_type not in self.texture_paths:
            print(f"Warning: Material type {material_type} not found in texture paths")
            return

        # Select random texture from available options
        texture_paths = self.texture_paths[material_type]
        texture_path = os.path.join(NVIDIA_NUCLEUS_DIR, random.choice(texture_paths))

        # Create material and shader
        material = UsdShade.Material.Define(stage, f"{prim_path}/Material")
        shader = UsdShade.Shader.Define(stage, f"{prim_path}/Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Set up UV coordinate reader
        primvar_reader = UsdShade.Shader.Define(stage, f"{prim_path}/Material/PrimvarReader_float2")
        primvar_reader.CreateIdAttr("UsdPrimvarReader_float2")
        primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        # Create texture sampler
        texture_sampler = UsdShade.Shader.Define(stage, f"{prim_path}/Material/diffuseTexture")
        texture_sampler.CreateIdAttr("UsdUVTexture")
        texture_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
        texture_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            primvar_reader.ConnectableAPI(), "result"
        )

        # Connect texture to shader
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture_sampler.ConnectableAPI(), "rgb"
        )

        # Set material properties based on material type
        props = self.material_properties.get(material_type, {})
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(props.get("roughness", 0.5))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(props.get("metallic", 0.0))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(props.get("opacity", 1.0))

        # Connect shader to material
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Handle UV coordinates based on geometry type
        if is_cube:
            self._setup_cube_uvs(prim_path, stage, material)
        else:
            self._setup_mesh_uvs(prim_path, stage, material)

    def _setup_cube_uvs(self, prim_path, stage, material):
        """Set up UV coordinates for cube geometry."""
        prim = stage.GetPrimAtPath(prim_path)
        texCoords = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )

        # UV coordinates for all 6 faces of the cube
        texCoords.Set([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),  # Front
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),  # Back
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),  # Top
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),  # Bottom
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),  # Left
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),  # Right
        ])

        # Indices for the UV coordinates
        texCoords.SetIndices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _setup_mesh_uvs(self, prim_path, stage, material):
        """Set up UV coordinates for mesh geometry."""
        prim = stage.GetPrimAtPath(prim_path)
        for child_prim in Usd.PrimRange(prim):
            if child_prim.GetTypeName() in ["Mesh", "Cube"]:
                if not child_prim.HasAttribute("primvars:st"):
                    texCoords = UsdGeom.PrimvarsAPI(child_prim).CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                    )
                    texCoords.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
                UsdShade.MaterialBindingAPI(child_prim).Bind(material)

    def save_usd(self, scene_data, scene_number):
        """
        Create USD scene file with objects and optional position markers.

        Args:
            scene_data (dict): Scene data with object placements
            scene_number (int): Scene identifier
        """
        filepath = os.path.join(self.save_dir, f"scene_{scene_number + 1}.usd")
        stage = Usd.Stage.CreateNew(filepath)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # Create floor
        floor_data = scene_data["scene"]["Floor"][0]
        floor_path = "/World/Floor"
        floor = UsdGeom.Cube.Define(stage, floor_path)

        # Set floor transform
        floor_size = floor_data["size"]
        floor_xform = floor.AddXformOp(UsdGeom.XformOp.TypeTransform)
        scale_matrix = Gf.Matrix4d().SetScale(Gf.Vec3d(floor_size[0] / 2, floor_size[1] / 2, floor_size[2]))
        translate_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, -1))
        transform_matrix = scale_matrix * translate_matrix
        floor_xform.Set(transform_matrix)

        # Apply texture to floor
        self.apply_random_texture(floor_path, "Floor", stage, is_cube=True)

        def create_scaled_object(obj_name, obj_data, index):
            """
            Create and scale an object in the USD stage.

            Args:
                obj_name (str): Name of the object type
                obj_data (dict): Object placement data
                index (int): Object instance index
            """
            if obj_data["asset_path"] is None:
                return

            obj_path = f"/World/{obj_name}_{index}"
            parent_prim = stage.DefinePrim(obj_path, "Xform")
            asset_scope = UsdGeom.Scope.Define(stage, f"{obj_path}/Asset")
            asset_prim = stage.DefinePrim(f"{obj_path}/Asset/Geom", "Xform")
            abs_path = os.path.abspath(obj_data["asset_path"])
            asset_prim.GetReferences().AddReference(abs_path)

            xformable = UsdGeom.Xformable(parent_prim)
            xformable.ClearXformOpOrder()

            scale = obj_data["scale_factor"]
            pos = obj_data["position"]
            rot = obj_data["rotation"]

            scale_matrix = Gf.Matrix4d().SetScale(Gf.Vec3d(scale, scale, scale))
            translate_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pos))

            if rot[2] != 0:
                rotate_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rot[2]))
                transform_matrix = scale_matrix * rotate_matrix * translate_matrix
            else:
                transform_matrix = scale_matrix * translate_matrix

            transform_op = xformable.AddTransformOp()
            transform_op.Set(transform_matrix)

            # Apply texture to the object
            self.apply_random_texture(f"{obj_path}/Asset/Geom", obj_name, stage, is_cube=False)

        # Create all objects in the scene
        for obj_name, obj_list in scene_data["scene"].items():
            if obj_name != "Floor":
                for i, obj_data in enumerate(obj_list):
                    create_scaled_object(obj_name, obj_data, i)

        stage.Save()
