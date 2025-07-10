# Isaac Lab Docker

This directory contains all Docker-related files and utilities for Isaac Lab deployment, providing containerized environments for development, training, and cluster execution.

## Overview

Isaac Lab uses Docker containers to provide consistent, reproducible environments across different platforms. The Docker setup includes:

- **Base Container**: Isaac Lab with Isaac Sim integration
- **Local Asset Support**: Optional local Isaac Sim asset packs for improved performance
- **Cluster Support**: Utilities for running on HPC clusters with Apptainer
- **Development Tools**: Live code editing and persistent data storage

## Quick Start

### Local Development
```bash
# Build and start the base container
./docker/container.py start

# Enter the running container
./docker/container.py enter

# Stop the container
./docker/container.py stop
```

## Container Profiles

### Base Profile (`isaac-lab-base`)
- Isaac Sim 4.5.0 integration
- Python development environment
- GPU support with NVIDIA drivers
- Live code editing capabilities
- Persistent cache and data storage

## Key Components

### Core Files
- `docker-compose.yaml` - Main container orchestration
- `Dockerfile.base` - Base Isaac Lab container definition
- `container.py` - Main Docker management utility

### Environment Configuration
- `.env.base` - Base environment variables
- `.env.cluster` - Cluster deployment configuration

### Cluster Support
- `cluster/cluster_interface.sh` - Cluster deployment interface
- `cluster/run_singularity.sh` - Singularity execution script
- `cluster/submit_job_*.sh` - Job scheduler templates (SLURM/PBS)

### Utilities
- `utils/` - Python utilities for container management
- `x11.yaml` - X11 forwarding configuration
- `.ros/` - ROS2 middleware configuration files

## Volume Strategy

Isaac Lab uses **bind mounts** instead of named volumes to store all cache, logs, and data in a local `docker_volumes` directory at the project root. This approach provides:

1. **Direct access** to files from the host system
2. **Easy backup and migration** between machines  
3. **Persistent data** that survives container rebuilds
4. **Development**: Live code editing with immediate container reflection
5. **Debugging**: Easy access to logs and cache files for troubleshooting

### Volume Directory Structure

All Isaac Lab and Isaac Sim data is stored in bind mounts pointing to the local `docker_volumes` directory:

```
docker_volumes/
├── kit/cache/                 - Isaac Sim kit cache (shader cache, configs, etc.)
├── kit/logs/Kit/Isaac-Sim/    - Isaac Sim detailed logs
├── cache/ov/                  - Omniverse client cache (materials, assets, etc.)
├── cache/pip/                 - Python package cache (speeds up package installs)
├── cache/nvidia/GLCache/      - OpenGL shader cache (improves rendering performance)
├── cache/compute/             - CUDA compute cache (speeds up GPU computations)
├── logs/omniverse/            - Omniverse application logs
├── data/omniverse/            - Omniverse user data (scenes, materials, custom assets)
├── docs/                      - User documentation and configuration files
├── docs_build/                - Built documentation artifacts (prevents root-owned files)
├── shell_history/             - Persistent bash shell history
└── assets/Assets/Isaac/4.5/   - Local Isaac Sim asset packs (optional)
```

### Bind Mount Mappings

#### Isaac Sim Cache and Data Volumes
These volumes follow NVIDIA's recommendations for persistent storage:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../../docker_volumes/kit/cache` | `${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache` | Isaac Sim kit cache |
| `../../docker_volumes/kit/logs/Kit/Isaac-Sim` | `${DOCKER_ISAACSIM_ROOT_PATH}/kit/logs/Kit/Isaac-Sim` | Isaac Sim detailed logs |

#### Omniverse and NVIDIA Cache Volumes
These store Omniverse client cache and NVIDIA graphics/compute cache:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../../docker_volumes/cache/ov` | `${DOCKER_USER_HOME}/cache/ov` | Omniverse client cache |
| `../../docker_volumes/cache/pip` | `${DOCKER_USER_HOME}/cache/pip` | Python package cache |
| `../../docker_volumes/cache/nvidia/GLCache` | `${DOCKER_USER_HOME}/cache/nvidia/GLCache` | OpenGL shader cache |
| `../../docker_volumes/cache/compute` | `${DOCKER_USER_HOME}/cache/compute` | CUDA compute cache |

#### Logging and Data Volumes

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../../docker_volumes/logs/omniverse` | `${DOCKER_USER_HOME}/logs/omniverse` | Omniverse application logs |
| `../../docker_volumes/data/omniverse` | `${DOCKER_USER_HOME}/data/omniverse` | Omniverse user data |
| `../../docker_volumes/docs` | `${DOCKER_USER_HOME}/docs` | User documentation |

#### Isaac Lab Source Code Volumes
These bind mounts allow live editing of Isaac Lab source code:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../source` | `${DOCKER_ISAACLAB_PATH}/source` | Isaac Lab source code |
| `../scripts` | `${DOCKER_ISAACLAB_PATH}/scripts` | Isaac Lab scripts |
| `../docs` | `${DOCKER_ISAACLAB_PATH}/docs` | Isaac Lab documentation |
| `../tools` | `${DOCKER_ISAACLAB_PATH}/tools` | Isaac Lab tools |
| `../../libraries` | `${DOCKER_ISAACLAB_PATH}/libraries` | External libraries |

#### Isaac Lab Output and Build Volumes

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../logs` | `${DOCKER_ISAACLAB_PATH}/logs` | Training logs, experiment outputs |
| `../outputs` | `${DOCKER_ISAACLAB_PATH}/outputs` | Training outputs, saved models |
| `../data_storage` | `${DOCKER_ISAACLAB_PATH}/data_storage` | Data storage |
| `../../docker_volumes/docs_build` | `${DOCKER_ISAACLAB_PATH}/docs/_build` | Built documentation artifacts |

#### Isaac Sim Assets Volume

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../../docker_volumes/assets/Assets/Isaac/4.5` | `/isaacsim_assets/Assets/Isaac/4.5` | Local Isaac Sim asset packs |

### Directory Population Lifecycle

**IMPORTANT**: When do directories get populated?

- **NOT during container build** (Dockerfile) - directories created but EMPTY
- **NOT during container start** (docker-compose up) - bind mounts established but EMPTY
- **ONLY during RUNTIME** when Isaac Lab applications actually run:
  - First Isaac Lab command: Creates basic cache files
  - First Isaac Sim usage: Populates kit cache, shader cache, GPU cache
  - First pip install: Populates Python package cache
  - Training runs: Creates logs, outputs, and training data
  - Omniverse operations: Downloads assets and creates client cache

## Isaac Sim Asset Management

Isaac Lab now supports **local Isaac Sim asset packs** to improve performance and reduce dependency on cloud asset downloads. The container automatically checks for local assets and provides guidance when they are not found.

### Asset Strategy

Isaac Lab containers are configured to use local assets when available, with automatic fallback to cloud assets. This provides:

1. **Faster loading** - No network downloads during simulation startup
2. **Offline capability** - Run simulations without internet connectivity
3. **Consistent assets** - Guaranteed asset versions across different environments
4. **Reduced bandwidth** - Avoid repeated downloads of large asset files

### Asset Configuration

The container configures Isaac Sim to use local assets through multiple methods:

#### 1. Configuration File Method
The `Dockerfile.base` automatically modifies the Isaac Sim configuration file (`isaacsim.exp.base.kit`) to set:
- `persistent.isaac.asset_root.default = "/isaacsim_assets/Assets/Isaac/4.5"`
- Asset browser folder paths pointing to local directories

#### 2. Command-Line Flag Method  
The container also provides wrapper scripts that include the asset root flag:
- `isaac-sim-assets` - Isaac Sim with local assets configured
- `python-assets` - Python launcher with local assets configured

#### 3. Environment Variable
The container sets `ISAACSIM_ASSET_ROOT=/isaacsim_assets/Assets/Isaac/4.5` to inform applications of the local asset location.

### Setting Up Local Assets

When you start the container, it automatically checks for local assets and provides detailed instructions if they are missing:

#### Manual Asset Download

1. **Download Asset Packs**: Visit the [Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack) and download the three asset packs:
   - `isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005.zip`
   - `isaac-sim-assets-2@4.5.0-rc.36+release.19112.f59b3005.zip`
   - `isaac-sim-assets-3@4.5.0-rc.36+release.19112.f59b3005.zip`

2. **Extract Assets**: Extract all three zip files to `docker_volumes/assets/Assets/Isaac/4.5`:
   ```bash
   cd docker_volumes/assets/Assets/Isaac/4.5
   unzip isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005.zip
   unzip isaac-sim-assets-2@4.5.0-rc.36+release.19112.f59b3005.zip
   unzip isaac-sim-assets-3@4.5.0-rc.36+release.19112.f59b3005.zip
   ```

3. **Verify Structure**: Ensure the final directory contains both `NVIDIA` and `Isaac` folders:
   ```
   docker_volumes/assets/Assets/Isaac/4.5/
   ├── NVIDIA/
   └── Isaac/
   ```

#### Automatic Asset Checking

The container interface automatically:
- Checks for local assets when starting the container
- Provides step-by-step download instructions if assets are missing
- Verifies the correct directory structure exists
- Continues with cloud assets if local assets are not available

### Asset Usage

Once local assets are configured:

#### Using Local Assets (Default)
```bash
# Isaac Sim will automatically use local assets when available
./docker/container.py start
./docker/container.py enter
isaaclab  # Uses local assets by default
```

#### Using Cloud Assets (Override)
If you need to use cloud assets instead of local ones, you can override the configuration:
```bash
# Override asset root to use cloud assets
./isaac-sim/python.sh --/persistent/isaac/asset_root/default="" your_script.py
```

### Troubleshooting Assets

#### Asset Not Found Errors
If you encounter "asset not found" errors:
1. Check that assets are properly extracted to `docker_volumes/assets/Assets/Isaac/4.5`
2. Verify both `NVIDIA` and `Isaac` folders exist in the assets directory
3. Restart the container after adding assets
4. Check container logs for asset-related warnings

#### Network Issues
If cloud asset downloads fail:
1. Verify internet connectivity from within the container
2. Check firewall settings for Omniverse asset servers
3. Consider using local assets for offline environments

#### Storage Space
Local assets require approximately 10-15 GB of disk space. Monitor the `docker_volumes/assets` directory size and ensure sufficient disk space is available.

## Advanced Usage

### Container Management Commands
```bash
# Generate configuration without starting
./docker/container.py config

# Copy artifacts from container
./docker/container.py copy

# Deep cleanup (removes all images and volumes)
./docker/container.py cleanup
```

### Custom Environment Files
```bash
# Use custom environment files
./docker/container.py start --env-files .env.custom

# Use custom docker-compose extensions
./docker/container.py start --files custom.yaml
```

### X11 Forwarding
The container automatically detects and configures X11 forwarding for GUI applications. Configuration is stored in `docker_volumes/config/.container.cfg`.

## Migration Instructions

To move to a new machine, simply copy the entire project directory including the `docker_volumes` folder. All cache, configurations, and data will be preserved exactly as they were on the original machine.

## Singularity Compatibility

The volume strategy is also compatible with Singularity containers. The Dockerfile.base creates additional directories and NVIDIA binary placeholders specifically for Singularity compatibility.

## Environment Variables

Key environment variables used in the volume strategy:

- `DOCKER_ISAACSIM_ROOT_PATH`: Path to Isaac Sim root folder (typically `/isaac-sim`)
- `DOCKER_ISAACLAB_PATH`: Path to Isaac Lab directory (typically `/workspace/isaaclab`)
- `DOCKER_USER_HOME`: Home directory of docker user (typically `/root`)

## Troubleshooting

### Permission Issues
If you encounter permission issues with cache directories, ensure the directories exist in the container before bind mounting. The Dockerfile.base creates these directories with proper ownership.

### Cache Corruption
If cache becomes corrupted, you can safely delete the `docker_volumes` directory and restart the container. The cache will be regenerated during the next run.

### Disk Space
Monitor the `docker_volumes` directory size, especially the cache subdirectories. Isaac Sim and Omniverse can accumulate significant cache data over time.

### Common Issues
- **Container won't start**: Check Docker and NVIDIA driver compatibility
- **GPU not detected**: Ensure NVIDIA Container Toolkit is installed
- **X11 forwarding fails**: Install xauth on the host system
- **Permission denied**: Check file ownership in docker_volumes directory
- **Asset not found errors**: Check local assets are properly downloaded and extracted to `docker_volumes/assets/Assets/Isaac/4.5`
- **Missing NVIDIA/Isaac folders**: Re-extract all three asset pack zip files to the assets directory 