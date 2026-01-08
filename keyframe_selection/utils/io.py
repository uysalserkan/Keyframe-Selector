"""
I/O utilities: seeding, logging, video metadata, config loading.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def set_global_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # For full reproducibility (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging with consistent formatting.
    
    Args:
        level: Logging level (e.g., logging.INFO).
        log_file: Optional path to write logs to file.
        format_string: Custom format string.
    
    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Configure root logger
    handlers: list = [logging.StreamHandler(sys.stdout)]
    
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    
    # Create package logger
    logger = logging.getLogger("keyframe_selection")
    logger.setLevel(level)
    
    return logger


def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from a video file.
    
    Args:
        video_path: Path to video file.
    
    Returns:
        Dictionary containing video metadata:
            - fps: Frames per second
            - frame_count: Total number of frames
            - duration: Duration in seconds
            - width: Frame width
            - height: Frame height
            - codec: Video codec
    """
    import cv2
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        
        duration = frame_count / fps if fps > 0 else 0.0
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
            "codec": codec_str,
            "path": str(video_path),
        }
    finally:
        cap.release()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file.
    
    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    content = config_path.read_text()
    
    if config_path.suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
        return yaml.safe_load(content)
    elif config_path.suffix == ".json":
        return json.loads(content)
    else:
        # Try YAML first, then JSON
        if HAS_YAML:
            try:
                return yaml.safe_load(content)
            except yaml.YAMLError:
                pass
        return json.loads(content)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save config.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Path objects to strings for serialization
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(v) for v in obj]
        return obj
    
    config = convert_paths(config)
    
    if config_path.suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError("PyYAML is required to save YAML configs. Install with: pip install pyyaml")
        content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(config, indent=2)
    
    config_path.write_text(content)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
