from __future__ import annotations

import os
import subprocess
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from .state_file import StateFile


class ApptainerInterface:
    """A helper class for managing Isaac Lab containers with Apptainer."""

    def __init__(
        self,
        context_dir: Path,
        profile: str = "base",
        envs: list[str] | None = None,
        statefile: StateFile | None = None,
    ):
        """Initialize the Apptainer interface with the given parameters.

        Args:
            context_dir: The context directory for Apptainer operations.
            profile: The profile name for the container. Defaults to "base".
            envs: A list of environment variable files to extend the ``.env.base`` file.
            statefile: An instance of the :class:`Statefile` class to manage state variables.
        """
        # set the context directory
        self.context_dir = context_dir

        # set important directory paths as instance variables
        self.docker_dir = self.context_dir
        self.isaaclab_root = self.docker_dir.parent
        self.repo_root = self.isaaclab_root.parent
        self.vol_root = self.repo_root / "docker_volumes"

        # create a state-file if not provided
        if statefile is None:
            self.statefile = StateFile(path=self.context_dir.parent.parent / "docker_volumes" / "config" / ".container.cfg")
        else:
            self.statefile = statefile

        # set the profile and container name
        self.profile = profile
        if self.profile == "isaaclab":
            # Silently correct from isaaclab to base
            self.profile = "base"

        self.container_name = f"isaac-lab-{self.profile}"
        self.image_name = f"isaac-lab-{self.profile}.sif"
        self.image_path = self.repo_root / "hyak_transfer" / self.image_name

        # keep the environment variables from the current environment
        self.environ = os.environ.copy()

        # load environment variables from .env files
        self._load_env_files(envs)

    def _load_env_files(self, envs: list[str] | None = None):
        """Load .env files and pass them into the container."""
        self.dot_vars: dict[str, str] = {}

        env_files = [".env.base"]
        if self.profile != "base":
            env_files.append(f".env.{self.profile}")
        if envs:
            env_files.extend(envs)

        # Set default headless and asset environment variables
        default_vars = {
            "HEADLESS": "1",
            "ENABLE_CAMERAS": "1",
            "LIVESTREAM": "0",
            "ISAACSIM_ASSET_ROOT": "/isaacsim_assets/Assets/Isaac/4.5",
            "HISTFILE": "/root/.bash_history",
            "HISTSIZE": "1000",
            "HISTFILESIZE": "2000",
            "TMPDIR": "/tmp",
            "TEMP": "/tmp",
            "TMP": "/tmp"
        }
        for k, v in default_vars.items():
            self.dot_vars[k] = v
            self.environ[k] = v
            self.environ[f"APPTAINERENV_{k}"] = v

        for rel_path in env_files:
            env_path = self.context_dir / rel_path
            if not env_path.exists():
                continue
            with env_path.open() as fp:
                for line in fp:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    self.dot_vars[k] = v
                    self.environ[k] = v
                    self.environ[f"APPTAINERENV_{k}"] = v

    def check_isaac_assets(self):
        """Check if Isaac Sim asset packs exist in the required location.
        
        Verifies that Isaac Sim assets are present in docker_volumes/assets/Assets/Isaac/4.5.
        If not found, provides instructions for manual download.
        """
        # Define the target directory
        assets_dir = self.vol_root / "assets" / "Assets" / "Isaac" / "4.5"
        
        # Check if assets already exist (look for both NVIDIA and Isaac folders)
        if (assets_dir.exists() and 
            (assets_dir / "NVIDIA").exists() and 
            (assets_dir / "Isaac").exists()):
            print(f"[INFO] Isaac Sim assets found at {assets_dir}")
            return
            
        print(f"[INFO] Isaac Sim assets not found at {assets_dir}")
        print("[INFO] Download the three Isaac Sim asset packs from:")
        print("[INFO] https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack")
        print(f"[INFO] Extract to: {assets_dir}")
        print("[INFO] Cloud assets will be used by default if local assets are unavailable.")

    def _get_bind_mounts(self) -> list[str]:
        """Return the bind-mount arguments for Apptainer.
        
        Returns:
            A list of strings with the --bind arguments for Apptainer.
        """
        # Ensure all required directories exist
        required_dirs = [
            # === Isaac Sim cache and logs ===
            self.vol_root / "kit/cache",
            self.vol_root / "kit/logs/Kit/Isaac-Sim",
            self.vol_root / "kit/data",
            self.vol_root / "assets/Assets/Isaac/4.5",
            # === Omniverse and NVIDIA caches ===
            self.vol_root / "cache/ov",
            self.vol_root / "cache/pip",
            self.vol_root / "cache/nvidia/GLCache",
            self.vol_root / "cache/compute",
            # === Logging and data volumes ===
            self.vol_root / "logs/omniverse",
            self.vol_root / "data/omniverse",
            self.vol_root / "docs",
            # === Shell history directory ===
            self.vol_root / "shell_history",
            # === Temporary directory ===
            self.vol_root / "tmp",
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"[INFO] Creating directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create bash_history file if it doesn't exist
        bash_history_file = self.vol_root / "shell_history" / ".bash_history"
        if not bash_history_file.exists():
            print(f"[INFO] Creating bash_history file: {bash_history_file}")
            bash_history_file.touch()
        
        # Local volume strategy mapping (host → container)
        volume_map = [
            # === Isaac Sim cache and logs ===
            # Isaac Sim kit cache (shader cache, configs, etc.)
            (self.vol_root / "kit/cache", "/isaac-sim/kit/cache"),
            # Isaac Sim detailed logs
            (self.vol_root / "kit/logs/Kit/Isaac-Sim", "/isaac-sim/kit/logs/Kit/Isaac-Sim"),
            (self.vol_root / "kit/data", "/isaac-sim/kit/data"),

            # === Isaac Sim Assets ===
            # Local Isaac Sim asset packs for improved performance and offline capability
            (self.vol_root / "assets/Assets/Isaac/4.5", "/isaacsim_assets/Assets/Isaac/4.5"),

            # === Omniverse and NVIDIA caches ===
            # Omniverse client cache (materials, assets, etc.)
            (self.vol_root / "cache/ov", "/root/.cache/ov"),
            # Python pip package cache (speeds up package installs)
            (self.vol_root / "cache/pip", "/root/.cache/pip"),
            # NVIDIA OpenGL shader cache (improves rendering performance)
            (self.vol_root / "cache/nvidia/GLCache", "/root/.cache/nvidia/GLCache"),
            # NVIDIA CUDA compute cache (speeds up GPU computations)
            (self.vol_root / "cache/compute", "/root/.nv/ComputeCache"),

            # === Logging and data volumes ===
            # Omniverse application logs
            (self.vol_root / "logs/omniverse", "/root/.nvidia-omniverse/logs"),
            # Omniverse user data (scenes, materials, custom assets)
            (self.vol_root / "data/omniverse", "/root/.local/share/ov/data"),
            # User documentation and configuration files
            (self.vol_root / "docs", "/root/Documents"),

            # === Isaac Lab source code volumes ===
            # These bind mounts allow live editing of Isaac Lab source code
            (self.isaaclab_root / "source", "/workspace/isaaclab/source"),
            (self.isaaclab_root / "scripts", "/workspace/isaaclab/scripts"),
            (self.isaaclab_root / "docs", "/workspace/isaaclab/docs"),
            (self.isaaclab_root / "tools", "/workspace/isaaclab/tools"),
            (self.repo_root / "libraries", "/workspace/isaaclab/libraries"),

            # === Isaac Lab output and build volumes ===
            # Training logs, experiment outputs, saved models
            (self.isaaclab_root / "logs", "/workspace/isaaclab/logs"),
            (self.isaaclab_root / "outputs", "/workspace/isaaclab/outputs"),
            (self.isaaclab_root / "data_storage", "/workspace/isaaclab/data_storage"),

            # === Shell history ===
            # Persistent bash history for convenience during development
            (self.vol_root / "shell_history/.bash_history", "/root/.bash_history"),

            # === Temporary directory ===
            # Ensure /tmp is accessible for temporary file operations
            (self.vol_root / "tmp", "/tmp"),
        ]

        # Format the bind mounts for Apptainer
        binds = []
        for host_path, container_path in volume_map:
            host_path = str(host_path.resolve())
            binds.append(f"{host_path}:{container_path}")
        
        return binds

    def _get_gpu_args(self) -> list[str]:
        """Get GPU arguments for Apptainer.
        
        Returns --nv flag to enable NVIDIA GPU support when running on GPU nodes.
        """
        return ["--nv"]

    def _get_x11_args(self) -> list[str]:
        """Get X11 forwarding arguments, only if X11 forwarding is enabled in the config file."""
        x11_args = []

        # Check if config file exists, if not, default to disabled
        if not self.statefile.path.exists():
            return x11_args

        # Set namespace for X11 variables
        original_namespace = self.statefile.namespace
        self.statefile.namespace = "X11"
        is_x11_forwarding_enabled = self.statefile.get_variable("X11_FORWARDING_ENABLED")
        # Restore original namespace
        self.statefile.namespace = original_namespace

        # Only enable X11 forwarding if explicitly enabled, DISPLAY is set, and X11 socket exists
        if (is_x11_forwarding_enabled == "1" and 
            "DISPLAY" in os.environ and 
            os.path.exists("/tmp/.X11-unix")):
            x11_args.extend([
                "--env", f"DISPLAY={os.environ['DISPLAY']}",
                "--bind", "/tmp/.X11-unix:/tmp/.X11-unix:rw"
            ])
        return x11_args

    def start(self):
        """Check if the Apptainer image exists and create it if not."""
        
        # Create Isaac Lab directories
        host_dirs = [
            self.isaaclab_root / "logs",
            self.isaaclab_root / "outputs", 
            self.isaaclab_root / "data_storage"
        ]
        
        for host_dir in host_dirs:
            if not host_dir.exists():
                print(f"[INFO] Creating directory: {host_dir}")
                host_dir.mkdir(parents=True, exist_ok=True)

        # Check Isaac Sim assets
        self.check_isaac_assets()
        
        if not self.image_path.exists():
            print(f"[INFO] Apptainer image '{self.image_name}' does not exist. Creating it...")
            
            # Path to the create_sif.sh script
            create_sif_script = self.docker_dir / "cluster" / "create_sif.sh"
            
            if not create_sif_script.exists():
                raise RuntimeError(f"create_sif.sh script not found at: {create_sif_script}")
            
            try:
                # Execute the create_sif.sh script
                result = subprocess.run(
                    [str(create_sif_script)],
                    cwd=create_sif_script.parent,
                    env=self.environ,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"[INFO] Successfully created Apptainer image '{self.image_name}'")
                print(result.stdout)
                
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to create Apptainer image '{self.image_name}'")
                print(f"Return code: {e.returncode}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                raise RuntimeError(f"Failed to create SIF image: {e}")
                
        else:
            print(f"[INFO] Image '{self.image_name}' already exists.")

    def enter(self):
        """Enter the container by executing a shell."""
        if not self.image_path.exists():
            raise RuntimeError(f"Image '{self.image_name}' does not exist. Run 'start' first.")

        print(f"[INFO] Entering '{self.container_name}' container...")
        # build the command
        cmd = ["apptainer", "exec", "--contain"]
        
        # Add GPU support
        cmd.extend(self._get_gpu_args())
        
        # Add X11 forwarding if enabled
        x11_args = self._get_x11_args()
        if x11_args:
            print("[INFO] X11 forwarding is enabled.")
            cmd.extend(x11_args)
        else:
            print("[INFO] X11 forwarding is disabled. No action taken.")
        
        # Get bind mounts
        bind_mounts = self._get_bind_mounts()
        
        # Add bind mounts to the command
        for bind_mount in bind_mounts:
            cmd.extend(["--bind", bind_mount])
        
        # Add the image and command to start bash with proper history setup
        cmd.extend([str(self.image_path), "/bin/bash", "-c", """
                cd /workspace/isaaclab
                export HISTFILE=/root/.bash_history
                export HISTSIZE=1000
                export HISTFILESIZE=2000
                shopt -s histappend
                PROMPT_COMMAND='history -a'
                # Configure Isaac Sim to use local assets
                export ISAAC_SIM_ASSET_ROOT="/isaacsim_assets/Assets/Isaac/4.5"
                exec /bin/bash --rcfile /root/.bashrc -i
                """.strip()])
        
        # Run the command
        subprocess.run(cmd, cwd=self.context_dir, env=self.environ)

    def stop(self):
        """Stop operation (no-op for Apptainer since containers are ephemeral)."""
        print("[INFO] Apptainer containers are ephemeral - nothing to stop.")

    def hard_stop(self):
        """Hard stop operation (no-op for Apptainer)."""
        print("[INFO] Apptainer containers are ephemeral - nothing to hard stop.")

    def cleanup(self):
        """Remove the Apptainer image and associated cache directories."""
        import shutil

        # Step 1: Remove the Apptainer image file if it exists
        if self.image_path.exists():
            print(f"[INFO] Removing Apptainer image '{self.image_name}'...")
            self.image_path.unlink()
        else:
            print(f"[INFO] Image '{self.image_name}' does not exist.")

        # Step 2: Remove cache and data directories under docker_volumes
        cache_dirs_to_remove = [
            # Isaac Sim cache and logs
            self.vol_root / "kit",
            # All cache directories
            self.vol_root / "cache",
            # Configuration directory
            self.vol_root / "config",
            # Logging and data volumes
            self.vol_root / "logs",
            self.vol_root / "data",
            self.vol_root / "docs",
            # Shell history and temp directories
            self.vol_root / "shell_history",
            self.vol_root / "tmp",
        ]
        for cache_dir in cache_dirs_to_remove:
            if cache_dir.exists():
                print(f"[INFO] Removing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
            else:
                print(f"[INFO] Cache directory does not exist: {cache_dir}")

        # Step 3: Remove workspace directories (logs, outputs, data_storage) with user confirmation
        workspace_dirs_to_remove = [
            self.isaaclab_root / "logs",
            self.isaaclab_root / "outputs",
            self.isaaclab_root / "data_storage"
        ]
        for workspace_dir in workspace_dirs_to_remove:
            if workspace_dir.exists():
                response = input(f"Remove workspace directory '{workspace_dir}'? [y/N]: ")
                if response.lower() in ['y', 'yes']:
                    print(f"[INFO] Removing workspace directory: {workspace_dir}")
                    shutil.rmtree(workspace_dir)
                else:
                    print(f"[INFO] Keeping workspace directory: {workspace_dir}")
            else:
                print(f"[INFO] Workspace directory does not exist: {workspace_dir}")

        # Step 4: Clean any remaining Apptainer cache on the system
        try:
            import subprocess
            print("[INFO] Cleaning Apptainer cache...")
            subprocess.run(["apptainer", "cache", "clean", "--force"], 
                          capture_output=True, check=False)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[INFO] Could not clean Apptainer cache (apptainer command not found or failed)")

    def copy(self, output_dir: Path | None = None):
        """Copy artifacts (not applicable for Apptainer with bind mounts)."""
        print("[INFO] Copy operation not needed with Apptainer bind mounts.")
        
    def config(self, output_yaml: Path | None = None):
        """Config operation not supported for Apptainer."""
        print("[INFO] Config operation not supported with Apptainer.")

