#!/usr/bin/env python3

import argparse
import os
import glob
from pathlib import Path
from typing import List, Dict
import yaml

# Import the base configurator to reuse functionality
from deepstream_auto_configurator import DeepStreamAutoConfigurator


class InferenceBuilderAutoConfigurator(DeepStreamAutoConfigurator):
    """
    Auto-configurator for inference-builder based on DeepStream auto-configurator.
    Generates ds_mv3dt.yaml and source_list_static.yaml instead of config_deepstream.txt.
    """
    
    def generate_ds_mv3dt_config(self, video_files: List[str]) -> str:
        """Generate ds_mv3dt.yaml with correct max_batch_size."""
        template_path = Path("config_templates/ds_mv3dt.yaml")
        if not template_path.exists():
            # If no template exists, use the 4cam sample as template
            template_path = Path("samples/inference_builder/4cam/ds_mv3dt.yaml")
            if not template_path.exists():
                raise FileNotFoundError(f"Template not found: ds_mv3dt.yaml")
        
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Detect video resolution from the first video file
        first_video_path = self.dataset_dir / "videos" / video_files[0]
        video_width, video_height = self.get_video_resolution(first_video_path)
        print(f"Detected video resolution: {video_width}x{video_height}")
        
        # Update configuration
        num_videos = len(video_files)
        model_config = config['models'][0]
        model_config['max_batch_size'] = num_videos
        model_config['parameters']['resize_video'] = [video_height, video_width]
        
        # Update tracker resolution
        if 'tracker_config' in model_config['parameters']:
            model_config['parameters']['tracker_config']['width'] = video_width
            model_config['parameters']['tracker_config']['height'] = video_height
        
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    def generate_source_list_static(self, video_files: List[str]) -> str:
        """Generate source_list_static.yaml based on video files."""
        header = """source-list:"""
        
        # Generate source entries
        source_entries = []
        for i, video_file in enumerate(sorted(video_files), 1):
            source_entries.extend([
                f'- uri: "file:///workspace/inputs/videos/{video_file}"',
                f'  sensor-id: Camera{i}',
                f'  sensor-name: UniqueSensorName{i}'
            ])
        
        footer = """source-config:
  source-bin: "nvurisrcbin"
  properties:
    file-loop: false"""
        
        return '\n'.join([header] + source_entries + [footer])
    
    def generate_nvdsinfer_config(self, video_files: List[str]) -> str:
        """Generate nvdsinfer_config.yaml based on config_pgie.txt template."""
        template_path = Path("config_templates/config_pgie.txt")
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r') as f:
            content = f.read()
        
        num_videos = len(video_files)
        
        # Simple format conversion: INI to YAML with batch-size adjustment
        yaml_lines = []
        current_section = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elif line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]  # Remove brackets
                yaml_lines.append(f'{current_section}:')
            elif '=' in line and current_section:
                key, value = line.split('=', 1)
                if key == 'batch-size':
                    yaml_lines.append(f'  {key}: {num_videos}')
                else:
                    yaml_lines.append(f'  {key}: {value}')
        
        return '\n'.join(yaml_lines)
    
    def generate_configs(self, enabled_sinks: List[str] = None, config_overrides: str = None, tracker_config: str = None) -> Dict[str, str]:
        """Generate inference-builder configs instead of DeepStream configs."""
        # Detect MP4 video files
        video_files = sorted([os.path.basename(f) for f in glob.glob(str(self.dataset_dir / "videos" / "*.mp4"))])
        if not video_files:
            raise ValueError(f"No MP4 files found in {self.dataset_dir}/videos")
        
        # Detect YML calibration files
        calib_files = sorted([os.path.basename(f) for f in glob.glob(str(self.dataset_dir / "camInfo" / "*.yml"))])
        
        print(f"Found {len(video_files)} videos, {len(calib_files)} calibration files")
        
        return {
            'ds_mv3dt.yaml': self.generate_ds_mv3dt_config(video_files),
            'config_tracker.yml': self.generate_tracker_config(video_files, calib_files, config_overrides, tracker_config),
            'source_list_static.yaml': self.generate_source_list_static(video_files),
            'nvdsinfer_config.yaml': self.generate_nvdsinfer_config(video_files),
            'config_msgconv.txt': self.generate_msgconv_config(video_files)
        }


def main():
    parser = argparse.ArgumentParser(description='Inference Builder Auto-Configurator')
    parser.add_argument('--dataset-dir', default='datasets/mtmc_4cam', help='Dataset directory with videos/ and camInfo/')
    parser.add_argument('--output-dir', default='temp_outputs', help='Output directory')
    parser.add_argument('--config-overrides', type=str, help='YAML file with section overrides (e.g., override_tracker_4cam.yml, override_tracker_12cam.yml)')
    parser.add_argument('--tracker-config', type=str, default='config_tracker.yml', help='Tracker configuration template (e.g., config_tracker_2d.yml)')
    parser.add_argument('--num_vision_neighbor', type=int, default=None, help='Number of vision neighbors per camera')
    parser.add_argument('--use_debug_communicator', action='store_true', help='Use debug communicator for pub_sub config')
    
    args = parser.parse_args()
    
    try:
        configurator = InferenceBuilderAutoConfigurator(args.dataset_dir, args.output_dir)
        
        # Generate pub_sub config (reuse from parent class)
        video_files = sorted([os.path.basename(f) for f in glob.glob(str(Path(args.dataset_dir) / "videos" / "*.mp4"))])
        if args.num_vision_neighbor is None:
            args.num_vision_neighbor = len(video_files) - 1
        configurator.generate_pub_sub_config(args.dataset_dir, video_files, args.num_vision_neighbor, args.use_debug_communicator)
        
        # Generate inference-builder configs
        configs = configurator.generate_configs(config_overrides=args.config_overrides, tracker_config=args.tracker_config)
        configurator.save_configs(configs)
        configurator.copy_static_configs()
        
        # Get video count for summary
        num_videos = len(video_files)
        
        print(f"✅ Generated inference-builder configs for {num_videos} videos")
        print(f"📁 Output: {args.output_dir}")
        print(f"🤝 Vision neighbors: {args.num_vision_neighbor}")
        print(f"🐛 Debug communicator: {'enabled' if args.use_debug_communicator else 'disabled'}")
        if args.config_overrides:
            print(f"🔧 Config overrides: {args.config_overrides}")
        print(f"📋 Generated files:")
        print(f"   - ds_mv3dt.yaml (inference config with max_batch_size: {num_videos})")
        print(f"   - config_tracker.yml (3D tracker config)")
        print(f"   - source_list_static.yaml (source configuration)")
        print(f"   - nvdsinfer_config.yaml (inference engine config with batch_size: {num_videos})")
        print(f"   - config_msgconv.txt (message converter config)")
        print(f"   - pub_sub_info_config_0.yml (communication config)")
        print(f"\n🚀 Run with inference builder using the generated configs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())