#!/usr/bin/env python3
import os
import argparse
import math
import glob
import re
import shutil
import subprocess
import yaml
import cv2
from pathlib import Path
from typing import List, Tuple, Dict

class DeepStreamAutoConfigurator:
    def __init__(self, dataset_dir: str, output_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
    
    def compute_grid_layout(self, num_videos: int) -> Tuple[int, int]:
        if num_videos <= 1: return 1, 1
        elif num_videos == 2: return 1, 2
        elif num_videos <= 4: return 2, 2
        elif num_videos <= 6: return 2, 3
        elif num_videos <= 9: return 3, 3
        elif num_videos <= 12: return 3, 4
        elif num_videos <= 16: return 4, 4
        else:
            side = math.ceil(math.sqrt(num_videos))
            return side, side
    
    def get_video_resolution(self, video_path: str) -> Tuple[int, int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid resolution detected: {width}x{height}")
        return width, height
    
    
    def generate_deepstream_config(self, video_files: List[str], enabled_sinks: List[str]) -> str:
        template_path = Path("config_templates/config_deepstream.txt")
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r') as f:
            content = f.read()
        
        num_videos = len(video_files)
        rows, columns = self.compute_grid_layout(num_videos)
        
        # Detect video resolution from the first video file
        first_video_path = self.dataset_dir / "videos" / video_files[0]
        video_width, video_height = self.get_video_resolution(first_video_path)
        
        print(f"Detected video resolution: {video_width}x{video_height}")
        
        # Update source sections and URIs
        content = self._update_sources(content, video_files)
        
        # Simple line-by-line replacements
        lines = content.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped.startswith('[') and stripped.endswith(']'):
                current_section = stripped[1:-1]
            elif stripped.startswith('batch-size='):
                lines[i] = f"batch-size={num_videos}"
            elif stripped.startswith('model-engine-file='):
                lines[i] = re.sub(r'_b\d+_', f'_b{num_videos}_', line)
            elif stripped.startswith('rows='):
                lines[i] = f"rows={rows}"
            elif stripped.startswith('columns='):
                lines[i] = f"columns={columns}"
            elif stripped.startswith('enable=') and current_section in ['sink0', 'sink1', 'sink2', 'sink3']:
                lines[i] = f"enable={'1' if current_section in enabled_sinks else '0'}"
            # Update streammux section resolution
            elif stripped.startswith('width=') and current_section == 'streammux':
                lines[i] = f"width={video_width}"
            elif stripped.startswith('height=') and current_section == 'streammux':
                lines[i] = f"height={video_height}"
            # Update tracker section resolution
            elif stripped.startswith('tracker-width=') and current_section == 'tracker':
                lines[i] = f"tracker-width={video_width}"
            elif stripped.startswith('tracker-height=') and current_section == 'tracker':
                lines[i] = f"tracker-height={video_height}"
        
        return '\n'.join(lines)
    
    def _update_sources(self, content: str, video_files: List[str]) -> str:
        lines = content.split('\n')
        result = []
        i = 0
        sources_added = 0
        
        while i < len(lines):
            line = lines[i]
            
            if line.strip().startswith('[source') and sources_added < len(video_files):
                # Keep/add source section
                result.append(line)
                i += 1
                while i < len(lines) and not (lines[i].strip().startswith('[') and lines[i].strip().endswith(']')):
                    if lines[i].strip().startswith('uri='):
                        result.append(f"uri=file:///workspace/inputs/videos/{video_files[sources_added]}")
                    else:
                        result.append(lines[i])
                    i += 1
                sources_added += 1
                continue
            elif line.strip().startswith('[source') and sources_added >= len(video_files):
                # Skip extra source sections
                i += 1
                while i < len(lines) and not (lines[i].strip().startswith('[') and lines[i].strip().endswith(']')):
                    i += 1
                continue
            elif line.strip().startswith('[streammux]'):
                # Add missing sources before streammux
                while sources_added < len(video_files):
                    result.extend([
                        "", f"[source{sources_added}]", "type=3", "enable=1", 
                        "cudadec-memtype=0", "gpu-id=0", "num-sources=1",
                        f"uri=file:///workspace/inputs/videos/{video_files[sources_added]}"
                    ])
                    sources_added += 1
                
                # Add blank line before streammux section
                result.append("")
                result.append(line)
            else:
                result.append(line)
            i += 1
        
        return '\n'.join(result)
    
    def generate_msgconv_config(self, video_files: List[str]) -> str:
        """Generate message converter configuration based on video files."""
        lines = []
        
        # Sort video files to ensure consistent ordering
        sorted_video_files = sorted(video_files)
        
        for i, video_file in enumerate(sorted_video_files, 1):
            # Use index-based camera IDs (1, 2, 3, 4...)
            camera_id = i
            
            lines.extend([
                f"[sensor{i-1}]",
                "enable=1", 
                "type=Camera",
                f"id=Camera{camera_id}",
                ""  # Empty line between sections
            ])
        
        # Add extra empty line at the end to match template format
        lines.append("")
        
        return '\n'.join(lines)

    def generate_tracker_config(self, video_files: List[str], calib_files: List[str], config_overrides: str = None, tracker_config: str = None) -> str:
        # Use specified tracker config or default to config_tracker.yml
        tracker_template = tracker_config or "config_tracker.yml"
        template_path = Path("config_templates") / tracker_template
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r') as f:
            content = f.read()
            
        # check if this is a 2d config
        is_2d_config = self._is_2d_tracker_config(content)
        
        # Load config overrides if provided
        overrides = self._load_section_overrides(config_overrides)
        if overrides and not is_2d_config:
            print(f"Applying section overrides from: {config_overrides}")
            content = self._apply_section_overrides(content, overrides)
        
        # Augment 2D config if needed
        if is_2d_config:
            print("Detected 2D tracker config - adding ObjectModelProjection, MultiViewAssociator, and Communicator sections")
            # Always add default multiview sections first
            content = self._add_multiview_sections(content, calib_files)
            # Then apply any user overrides on top
            if overrides:
                content = self._apply_section_overrides(content, overrides)
        
        lines = content.split('\n')
        num_videos = len(video_files)
        i = 0
        
        while i < len(lines):
            stripped = lines[i].strip()
            
            if stripped.startswith('cameraModelFilepath:'):
                i += 1
                # Remove existing entries
                start_idx = i
                while i < len(lines) and lines[i].startswith('    - '):
                    i += 1
                # Delete the old entries
                del lines[start_idx:i]
                i = start_idx  # Reset index to where we deleted
                
                # Add new camera paths - one for each video file
                print(f"Mapping calibration files to {num_videos} videos:")
                for j, video_file in enumerate(video_files):
                    # Try to find corresponding calibration file
                    calib_file = self._find_matching_calib_file(video_file, calib_files, j)
                    print(f"  Video {j+1}: {video_file} -> {calib_file}")
                    lines.insert(i, f"    - /workspace/inputs/camInfo/{calib_file}")
                    i += 1
                continue
            if stripped.startswith('stateEstimatorType:') and is_2d_config:
                old_value = lines[i].split(':')[1].strip()
                if old_value != '3':
                    lines[i] = "  stateEstimatorType: 3"
                    print(f"stateEstimatorType updated to 3")
            elif stripped.startswith('visualTrackerType:') and is_2d_config:
                old_value = lines[i].split(':')[1].strip()
                if old_value != '2':
                    lines[i] = "  visualTrackerType: 2"
                    print(f"visualTrackerType updated to 2")

            i += 1
        
        return '\n'.join(lines)
    
    def extract_camera_ids(self, video_files: List[str]) -> List[int]:
        """Extract camera IDs by using sorted index + 1."""
        sorted_files = sorted(video_files)
        return [i + 1 for i in range(len(sorted_files))]
    
    def generate_deployment_config(self, cam_ids: List[int]) -> str:
        """Generate deployment config file for production mode."""
        deployment_config = {
            'mqtt_broker_per_instance': ["127.0.0.1:1883"],
            'topic_template': "/trck/cam%d",
            'ds_instance_cam_assignment': [cam_ids],
            'ds_instance_gpu_assignment': [0]
        }
        
        # Save to temporary file
        deployment_config_path = self.output_dir / "deployment_config.yml"
        with open(deployment_config_path, 'w') as f:
            yaml.dump(deployment_config, f, default_flow_style=False)
        
        print(f"Generated deployment config: {deployment_config_path}")
        return str(deployment_config_path)
    
    def _is_2d_tracker_config(self, content: str) -> bool:
        """Check if the tracker config is missing 3D multi-view sections."""
        return not any(section in content for section in [
            'ObjectModelProjection:', 'MultiViewAssociator:', 'Communicator:'
        ])
    
    def _load_section_overrides(self, override_file: str) -> Dict:
        """Load section overrides from a YAML file."""
        if not override_file:
            return {}
        
        override_path = Path(override_file)
        if not override_path.exists():
            # Try relative to config_templates directory
            override_path = Path("config_templates") / override_file
            if not override_path.exists():
                print(f"Warning: Override file not found: {override_file}")
                return {}
        
        try:
            with open(override_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading override file {override_path}: {e}")
            return {}
    
    def _apply_section_overrides(self, content: str, overrides: Dict) -> str:
        """Apply section overrides to the config content."""
        if not overrides:
            return content
        
        lines = content.split('\n')
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            section_name = None
            
            # Check if this line starts a section that has an override
            stripped = line.strip()
            if stripped.endswith(':') and not stripped.startswith('-') and not stripped.startswith('#'):
                potential_section = stripped[:-1]  # Remove the colon
                if potential_section in overrides:
                    section_name = potential_section
            
            if section_name:
                # Skip the original section
                result_lines.append(f"# Original {section_name} section replaced by override")
                i += 1
                # Skip all lines until next section
                while i < len(lines):
                    next_line = lines[i].strip()
                    # Stop if we hit another top-level section (no leading spaces and ends with :)
                    if (next_line.endswith(':') and not next_line.startswith('-') and 
                        not next_line.startswith('#') and not lines[i].startswith(' ')):
                        break
                    i += 1
                
                # Add the override section
                result_lines.append(f"{section_name}:")
                section_data = overrides[section_name]
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, list):
                            result_lines.append(f"  {key}:")
                            for item in value:
                                result_lines.append(f"    - {item}")
                        else:
                            result_lines.append(f"  {key}: {value}")
            else:
                result_lines.append(line)
                i += 1
        
        return '\n'.join(result_lines)
    
    def _add_multiview_sections(self, content: str, calib_files: List[str]) -> str:
        """Add ObjectModelProjection, MultiViewAssociator, Communicator, and PoseEstimator sections to 2D config."""
        lines = content.split('\n')
        
        # Check which sections already exist
        has_object_model_projection = any('ObjectModelProjection:' in line for line in lines)
        has_multi_view_associator = any('MultiViewAssociator:' in line for line in lines)
        has_communicator = any('Communicator:' in line for line in lines)
        has_pose_estimator = any('PoseEstimator:' in line for line in lines)
        
        # Append to the bottom of the config
        insertion_point = len(lines)
        
        # Add ObjectModelProjection section if it doesn't exist
        if not has_object_model_projection:
            print("Adding ObjectModelProjection section")
            object_model_sections = [
                'ObjectModelProjection:',
                '  outputFootLocation: 1',
                '  outputVisibility: 1',
                '  outputConvexHull: 0',
                '  objectModelType: 0',
                '  cameraModelFilepath:'
            ]
            
            # Add camera model file paths
            for calib_file in calib_files:
                object_model_sections.append(f'    - /workspace/inputs/camInfo/{calib_file}')
            
            # Insert ObjectModelProjection section
            for j, section_line in enumerate(object_model_sections):
                lines.insert(insertion_point + j, section_line)
            insertion_point += len(object_model_sections)
        
        # Add MultiViewAssociator section if it doesn't exist
        if not has_multi_view_associator:
            print("Adding MultiViewAssociator section")
            multiview_sections = [
                'MultiViewAssociator:',
                '  multiViewAssociatorType: 1',
                '  enableLatePeerReAssoc: 1',
                '  enableIDCorrection: 1',
                '  enableSeeThrough: 1',
                '  enableMsgSync: 1',
                '  maxPeerTrackletSize: 50',
                '  recentlyActiveAge: 178',
                '  minCommonFrames4MatchScore: 2',
                '  minPeerToPredDistance4Fusion: 1.35',
                '  minPeerVisibility4Fusion: 0.15',
                '  minPeerTrackletMatchScore: 0.48',
                '  maxTrackletMatchingTimeSearchRange: 1',
                '  maxPeerFrameDiff4NoDet: 2',
                '  communicatorInitSleepTime: 0',
            ]
            
            # Insert MultiViewAssociator section
            for j, section_line in enumerate(multiview_sections):
                lines.insert(insertion_point + j, section_line)
            insertion_point += len(multiview_sections)
        
        # Add Communicator section if it doesn't exist
        if not has_communicator:
            print("Adding Communicator section")
            communicator_sections = [
                'Communicator:',
                '  communicatorType: 2',
                '  pubSubInfoConfigPath: /workspace/experiments/pub_sub_info_config_0.yml',
                '  mqttProtoAdaptorConfigPath: /workspace/experiments/config_mqtt.txt',
            ]
            
            # Insert Communicator section
            for j, section_line in enumerate(communicator_sections):
                lines.insert(insertion_point + j, section_line)
            insertion_point += len(communicator_sections)
        
        # Add PoseEstimator section if it doesn't exist
        if not has_pose_estimator:
            print("Adding PoseEstimator section")
            pose_estimator_sections = [
                'PoseEstimator:',
                '  poseEstimatorType: 1',
                '  useVPICropScaler: 1',
                '  batchSize: 1',
                '  workspaceSize: 1000',
                '  inferDims: [3, 256, 192]',
                '  networkMode: 1',
                '  inputOrder: 0',
                '  colorFormat: 0',
                '  offsets: [123.6750, 116.2800, 103.5300]',
                '  netScaleFactor: 0.00392156',
                '  onnxFile: /workspace/models/BodyPose3DNet/bodypose3dnet_accuracy.onnx',
                '  modelEngineFile: /workspace/models/BodyPose3DNet/bodypose3dnet_accuracy.onnx_b1_gpu0_fp16.engine',
                '  poseInferenceInterval: -1',
                ''  # Empty line
            ]
            
            # Insert PoseEstimator section
            for j, section_line in enumerate(pose_estimator_sections):
                lines.insert(insertion_point + j, section_line)
        
        return '\n'.join(lines)
    
    def _find_matching_calib_file(self, video_file: str, calib_files: List[str], index: int) -> str:
        """Find the corresponding calibration file for a video file."""
        # Remove extension from video file to get base name
        video_base = os.path.splitext(video_file)[0]
        
        # Try to find exact match by replacing video extension with .yml
        yml_candidate = video_base + ".yml"
        if yml_candidate in calib_files:
            return yml_candidate
        
        # Try to find match by extracting camera number/ID from video filename
        video_match = re.search(r'[Cc]am(\d+)', video_file)
        if video_match:
            cam_id = video_match.group(1)
            # Look for calibration files with the same camera ID
            for calib_file in calib_files:
                calib_match = re.search(r'[Cc]am(\d+)', calib_file)
                if calib_match and calib_match.group(1) == cam_id:
                    return calib_file
        
        # Fallback: use index-based matching (sorted order)
        if index < len(calib_files):
            return calib_files[index]
        
        # Last resort: generate expected filename based on pattern
        # If we have calib files, try to follow their naming pattern
        if calib_files:
            # Get the pattern from the first calibration file
            first_calib = calib_files[0]
            calib_match = re.search(r'(.+[Cc]am)(\d+)(\.yml)$', first_calib)
            if calib_match:
                prefix, _, suffix = calib_match.groups()
                cam_id = str(index + 1).zfill(len(calib_match.group(2)))
                expected_file = f"{prefix}{cam_id}{suffix}"
                print(f"    Warning: Expected calibration file {expected_file} not found!")
                return expected_file
        
        # Final fallback: generate a generic name
        expected_file = f"camera_{index + 1:03d}.yml"
        print(f"    Warning: Generated fallback calibration filename {expected_file}")
        return expected_file
    
    def generate_pub_sub_config(self, dataset_dir: str, video_files: List[str], num_vision_neighbor: int = 3, use_debug_communicator: bool = False):
        """Generate pub_sub_info_config_0.yml using the existing script."""
        cam_ids = self.extract_camera_ids(video_files)
        cam_subset = ','.join([str(cam_id) for cam_id in cam_ids])
        
        # Check if the script exists
        script_path = Path("utils/generate_pub_sub_configs.py")
        if not script_path.exists():
            print(f"Error: Script not found at {script_path}")
            print(f"Current working directory: {os.getcwd()}")
            return
        
        command = [
            "python", str(script_path),
            "--cam_info_path", os.path.join(dataset_dir, "camInfo"),
            "--neighbor_criteria", f"top_N:{num_vision_neighbor}",
            "--output_path", str(self.output_dir)
        ]
        
        if use_debug_communicator:
            command.extend(["--use_debug_communicator", "--cam_subset", cam_subset])
        else:
            # Generate deployment config for production mode
            deployment_config_path = self.generate_deployment_config(cam_ids)
            command.extend(["--deployment_config_path", deployment_config_path])
        
        try:
            print(f"Generating pub_sub config for cameras: {cam_subset} (vision neighbors: {num_vision_neighbor})")
            print(f"Command: {' '.join(command)}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            # Check if file was actually created
            pub_sub_file = self.output_dir / "pub_sub_info_config_0.yml"
            if pub_sub_file.exists():
                print(f"Generated: {pub_sub_file}")
            else:
                print(f"Warning: pub_sub_info_config_0.yml not found at {pub_sub_file}")
                print(f"Command stdout: {result.stdout}")
                print(f"Command stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to generate pub_sub config: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Command error: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during pub_sub generation: {e}")
    
    def generate_configs(self, enabled_sinks: List[str] = None, config_overrides: str = None, tracker_config: str = None) -> Dict[str, str]:
        if enabled_sinks is None:
            enabled_sinks = ['sink0']
        
        # Detect MP4 video files
        video_files = sorted([os.path.basename(f) for f in glob.glob(str(self.dataset_dir / "videos" / "*.mp4"))])
        if not video_files:
            raise ValueError(f"No MP4 files found in {self.dataset_dir}/videos")
        
        # Detect YML calibration files
        calib_files = sorted([os.path.basename(f) for f in glob.glob(str(self.dataset_dir / "camInfo" / "*.yml"))])
        
        print(f"Found {len(video_files)} videos, {len(calib_files)} calibration files")
        
        # Generate pub_sub config (will be called from main with num_peers)
        # Note: num_peers will be passed from main function
        
        return {
            'config_deepstream.txt': self.generate_deepstream_config(video_files, enabled_sinks),
            'config_tracker.yml': self.generate_tracker_config(video_files, calib_files, config_overrides, tracker_config),
            'config_msgconv.txt': self.generate_msgconv_config(video_files)
        }
    
    def save_configs(self, configs: Dict[str, str]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in configs.items():
            path = self.output_dir / filename
            with open(path, 'w') as f:
                f.write(content)
            print(f"Generated: {path}")
    
    def copy_static_configs(self):
        """Copy static config files from templates to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # List of static config files to copy
        static_files = ['config_mqtt.txt', 'config_pgie.txt']
        
        for filename in static_files:
            src_path = Path('config_templates') / filename
            dst_path = self.output_dir / filename
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {dst_path}")
            else:
                print(f"Warning: Template file not found: {src_path}")

def main():
    parser = argparse.ArgumentParser(description='DeepStream Auto-Configurator')
    parser.add_argument('--dataset-dir', default='datasets/mtmc_4cam', help='Dataset directory with videos/ and camInfo/')
    parser.add_argument('--output-dir', default='temp_outputs', help='Output directory')
    parser.add_argument('--enable-osd', action='store_true', help='Enable OSD sink (sink1 - EglSink)')
    parser.add_argument('--enable-file-output', action='store_true', help='Enable video file output (sink2 - MP4)')
    parser.add_argument('--enable-msg-broker', action='store_true', help='Enable message broker output (sink3 - Kafka)')
    parser.add_argument('--num_vision_neighbor', type=int, default=None, help='Number of vision neighbors per camera')
    parser.add_argument('--use_debug_communicator', action='store_true', help='Use debug communicator for pub_sub config')
    parser.add_argument('--config-overrides', type=str, help='YAML file with section overrides (e.g., override_tracker_4cam.yml, override_tracker_12cam.yml)')
    parser.add_argument('--tracker-config', type=str, default='config_tracker.yml', help='Tracker configuration template (e.g., config_tracker_2d.yml)')
    
    args = parser.parse_args()
    
    enabled_sinks = ['sink0']  # sink0 always enabled (fake sink, type=1)
    if args.enable_osd: enabled_sinks.append('sink1')  # sink1 (OSD/EglSink, type=2)
    if args.enable_file_output: enabled_sinks.append('sink2')  # sink2 (MP4 file output, type=3)
    if args.enable_msg_broker: enabled_sinks.append('sink3')  # sink3 (Kafka message broker, type=6)
    
    try:
        configurator = DeepStreamAutoConfigurator(args.dataset_dir, args.output_dir)
        
        # Get video files for pub_sub generation
        video_files = sorted([os.path.basename(f) for f in glob.glob(str(Path(args.dataset_dir) / "videos" / "*.mp4"))])
        if args.num_vision_neighbor is None:
            args.num_vision_neighbor = len(video_files) - 1
        configurator.generate_pub_sub_config(args.dataset_dir, video_files, args.num_vision_neighbor, args.use_debug_communicator)
        
        configs = configurator.generate_configs(enabled_sinks, args.config_overrides, args.tracker_config)
        configurator.save_configs(configs)
        configurator.copy_static_configs()
        
        # Get video count for summary
        num_videos = len([f for f in glob.glob(str(Path(args.dataset_dir) / "videos" / "*.mp4"))])
        rows, columns = configurator.compute_grid_layout(num_videos)
        
        print(f"✅ Generated configs for {num_videos} videos ({rows}x{columns} grid)")
        print(f"📁 Output: {args.output_dir}")
        print(f"🔧 Sinks: {', '.join(enabled_sinks)}")
        print(f"🤝 Vision neighbors: {args.num_vision_neighbor}")
        print(f"📋 Generated files:")
        print(f"   - config_deepstream.txt (main pipeline config)")
        print(f"   - config_tracker.yml (3D tracker config)")
        print(f"   - config_msgconv.txt (message converter config)")
        print(f"   - pub_sub_info_config_0.yml (communication config)")
        print(f"\n🚀 Run with DeepStream container using the generated configs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())