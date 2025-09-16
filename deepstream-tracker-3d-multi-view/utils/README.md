## Python Utility Scripts

This directory contains Python utility scripts to help with configuration generation and visualization for the DeepStream Multi-View 3D Tracking project.


### `deepstream_auto_configurator.py`

The Deepstream auto-configurator automatically generates DeepStream configuration files based on your dataset.

**Usage:**
```bash
python deepstream_auto_configurator.py --dataset-dir DATASET_DIR [OPTIONS]
```

**Arguments:**
```
- `--dataset-dir`: Dataset directory containing `videos/` and `camInfo/` subdirectories
- `--output-dir`: Output directory for generated configs (default: `temp_outputs`)
- `--enable-osd`: Enable OSD display sink
- `--enable-file-output`: Enable video file output
- `--enable-msg-broker`: Enable Kafka message broker output
- `--num_vision_neighbor`: Number of vision neighbors per camera
- `--tracker-config`: Base tracker configuration file (default: `config_tracker.yml`)
- `--config-overrides`: YAML file containing section overrides for tracker configuration
```


### `inference_builder_auto_configurator.py`

The Inference Builder auto-configurator automatically generates Inference Builder config files based on your dataset.

**Usage:**
```bash
python inference_builder_auto_configurator.py --dataset-dir DATASET_DIR [OPTIONS]
```

**Arguments:**
```
- `--dataset-dir`: Dataset directory containing `videos/` and `camInfo/` subdirectories
- `--output-dir`: Output directory for generated configs (default: `temp_outputs`)
- `--tracker-config`: Base tracker configuration file (default: `config_tracker.yml`)
- `--config-overrides`: YAML file containing section overrides for tracker configuration
```



### `kafka_bev_visualizer.py`

Real-time bird's-eye view visualization of 3D tracking data from Kafka streams.

**Usage:**
```bash
python kafka_bev_visualizer.py [OPTIONS]
```

**Arguments:**
```
- `--dataset-path`: Path to dataset containing map.png and transforms.yml 
- `--msgconv-config`: Path to message converter config file 
- `--output-path`: Output directory for videos and screenshots
- `--offline`: Run in offline mode to save video from all messages instead of real-time visualization
- `--show-ids`: Show object IDs near trajectory heads
- `--average-multi-cam`: Average trajectory points from multiple cameras for the same object (shows 1 point per object instead of multiple points from different cameras)
```

**Interactive Controls (Real-time mode):**
- `q`: Quit application
- `s`: Save current frame as screenshot
- `c`: Clear all trajectories
- `r`: Start/stop recording video



## Additional Utilities

### `generate_pub_sub_configs.py`

Generates communication configurations for multi-camera tracking systems, including peer-to-peer relationships and camera neighbor mappings.

**Usage:**
```bash
python generate_pub_sub_configs.py --deployment_config_path CONFIG_PATH [OPTIONS]
```

**Arguments:**
```
- `--deployment_config_path`: Path to YAML file containing deployment configurations
- `--cam_info_path`: Directory containing camera calibration information 
- `--output_path`: Directory to store output configuration files
- `--neighbor_criteria`: Criteria for selecting neighboring cameras 
  - Format: `"top_N:{N}"` or `"overlap_threshold:{threshold}"`
- `--minimum_object_size`: Minimum object size in pixels for visibility 
- `--range_of_interest`: Range of interest in world coordinates
  - Format: `"x1,y1,x2,y2"` (min and max corners)
```


**Note:** This script is typically called automatically by the `deepstream_auto_configurator.py` script.


### `schema_pb2.py`

Contains Protocol Buffer schema definitions for message serialization used in Kafka communication.
