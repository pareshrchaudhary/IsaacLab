# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Scene API interface for procedural scene generation.

This module provides the main API interface for creating and managing procedurally generated scenes.
"""

from typing import Dict, List

from scripts.procedural_scene_gen_standalone.PSG_new.save import SaveManager
from scripts.procedural_scene_gen_standalone.PSG_new.traversal import Traversal

from omni.isaac.lab.utils.assets import NVIDIA_NUCLEUS_DIR

# Default texture configuration
DEFAULT_TEXTURE_CONFIG = {
    "enabled": True,
    "materials": {
        "Wood": [
            "Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
            "Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
            "Materials/Base/Wood/Oak/Oak_BaseColor.png",
        ],
        "Blank": ["Materials/Base/Metal/Blank.png"],
    },
    "object_materials": {
        "Floor": "Wood",
        "Table": "Wood",
        "Chair": "Wood",
        "Box": "Wood",
        "Cube": "Wood",
        "Bench": "Wood",
    },
    "material_properties": {
        "Wood": {"roughness": 0.7, "metallic": 0.0, "opacity": 1.0},
        "Blank": {"roughness": 0.2, "metallic": 0.8, "opacity": 1.0},
    },
}


class Scene:
    """Main interface for procedural scene generation."""

    def __init__(self, env):
        """Initialize scene with environment."""
        self.env = env
        self.config = {"scene": {}, "generation": None, "output": None, "textures": None}
        self.scene_graph = {}
        self.traversal = None
        self.save_manager = None

    def add_object(self, object_config: dict):
        """Add an object type to the scene with its properties and constraints."""
        object_type = object_config["object_type"]
        self.config["scene"][object_type] = object_config

        # Update scene graph
        for parent in object_config["parents"]:
            if parent not in self.scene_graph:
                self.scene_graph[parent] = []
            if object_type not in self.scene_graph[parent]:
                self.scene_graph[parent].append(object_type)

    def set_config(self, generation_config: dict, output_config: dict, texture_config: dict = None):
        """Set generation, output, and texture configuration."""
        self.config["generation"] = generation_config
        self.config["output"] = output_config

        # Set up texture configuration based on generation config
        texture_enabled = generation_config.get("texture_config", False)
        self.config["textures"] = {
            **DEFAULT_TEXTURE_CONFIG,  # Use default texture settings
            "enabled": texture_enabled,  # Control from generation config
            "randomization_enabled": texture_enabled,  # Keep both flags in sync
        }

    def generate(self) -> list[dict]:
        """Generate scenes based on the configured objects and parameters."""
        if not self.config["generation"] or not self.config["output"]:
            raise ValueError("Generation and output configuration must be set before generating scenes")

        # Initialize traversal with graph and config
        self.traversal = Traversal(self.scene_graph, self.config)

        # Generate scenes
        return self.traversal.generate_scenes()

    def save(self, scenes: list[dict]):
        """Save generated scenes according to output configuration."""
        if not self.config["output"]:
            raise ValueError("Output configuration must be set before saving")

        # Pass complete config including textures to SaveManager
        save_config = {"output": self.config["output"], "textures": self.config["textures"]}
        self.save_manager = SaveManager(save_config)
        self.save_manager.save_manager(scenes, self.scene_graph)

    def _apply_texture(self, stage, prim_path: str, obj_type: str):
        """Apply texture to a prim based on configuration."""
        from pxr import Sdf, UsdShade

        if not (self.config["textures"] and self.config["textures"]["enabled"]):
            return

        # Get material type for object
        material_type = self.config["textures"]["object_materials"].get(obj_type)
        if not material_type:
            return

        # Get available textures for this material
        textures = self.config["textures"]["materials"].get(material_type, [])
        if not textures:
            return

        import random

        # Select random texture and prepend NVIDIA_NUCLEUS_DIR
        texture_path = f"{NVIDIA_NUCLEUS_DIR}/{random.choice(textures)}"

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

        # Set material properties
        props = self.config["textures"]["material_properties"].get(material_type, {})
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(props.get("roughness", 0.5))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(props.get("metallic", 0.0))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(props.get("opacity", 1.0))

        # Connect shader to material
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Bind material to prim
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(material)

    def update(self, stage, scenes: list[dict], environment_path: str):
        """Update Isaac scene with generated object placements."""
        import os

        from pxr import Gf, Usd, UsdGeom

        # Check if texture randomization is enabled at scene level
        texture_randomization = self.config.get("textures", {}).get("randomization_enabled", False)

        for scene_data in scenes:
            for obj_type, objects in scene_data["scene"].items():
                if obj_type in ["Robot", "Floor"]:
                    continue

                for i, obj in enumerate(objects):
                    obj_path = f"{environment_path}/{obj_type}_{i+1}"
                    parent_prim = stage.DefinePrim(obj_path, "Xform")
                    asset_scope = UsdGeom.Scope.Define(stage, f"{obj_path}/Asset")
                    asset_prim = stage.DefinePrim(f"{obj_path}/Asset/Geom", "Xform")

                    asset_path = obj.get("asset_path")
                    if asset_path is None:
                        print(f"[PSG] Warning: No asset path for object type '{obj_type}'")
                        continue

                    if os.path.exists(asset_path):
                        asset_prim.GetReferences().AddReference(asset_path)
                    else:
                        print(f"[PSG] Warning: Asset file not found at '{asset_path}'")
                        continue

                    # Apply textures if enabled at scene level
                    if texture_randomization:
                        self._apply_texture(stage, f"{obj_path}/Asset/Geom", obj_type)

                    xformable = UsdGeom.Xformable(parent_prim)
                    xformable.ClearXformOpOrder()

                    pos = obj["position"]
                    rot = obj["rotation"]
                    scale_factor = obj.get("scale_factor", 1.0)

                    scale_matrix = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_factor, scale_factor, scale_factor))

                    if rot[2] != 0:
                        rotate_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rot[2]))
                        transform_matrix = scale_matrix * rotate_matrix * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pos))
                    else:
                        transform_matrix = scale_matrix * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pos))

                    transform_op = xformable.AddTransformOp()
                    transform_op.Set(transform_matrix)

                    print(f"[PSG] Placed {obj_type}_{i+1} at position: {pos}, rotation: {rot}, scale: {scale_factor}")
