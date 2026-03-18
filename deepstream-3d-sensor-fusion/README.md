## **What's DS3D Multi-modal sensor fusion**

The ``deepstream-3d-lidar-sensor-fusion`` sample application showcases multi-modal sensor fusion pipelines for LiDAR and camera data using the DS3D framework. This appliation with DS3D framework could setup different LiDAR/RADAR/Camera sensor fusion models, late fusion inference pipelines with several key features.

- Camera processing pipeline leveraging DeepStream’s generic 2D video pipeline with batchMeta.
- Custom ds3d::dataloader for LiDAR capture with pre-processing options.
- Custom ds3d::databridge converts DeepStream NvBufSurface and GstNvDsPreProcessBatchMeta data into shaped based tensor data s3d::Frame2DGuard and ds3d::FrameGuard formats, and embeds key-value pairs within ds3d::datamap.
- ds3d::mixer for efficient merging of camera, LiDAR and any sensor data into ds3d::datamap.
- ds3d::datatfiler followed by libnvds_tritoninferfilter.so for multi-modal ds3d::datamap inference and custom pre/post-processing.
- ds3d::datasink with ds3d_gles_ensemble_render for 3D detection result visualization with a multi-view display.

### Requirements:

 - [Deepstream SDK 7.0+](https://developer.nvidia.com/deepstream-sdk)

### Sample and source code location:

```bash
cd /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion/
```
### Further details:

Refer to the DeepStream documentation for a detailed explanation of this sample application:
[DS3D Multimodal Sensor Fusion](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_3D_MultiModal_Lidar_Sensor_Fusion.html)


## **About the ds3d-bevfusion NuScenes dataset (data/nuscene.tar.gz)**

This dataset is a subset of the original NuScenes dataset containing LiDAR and camera samples for the multi-modal sensor fusion demo. It's licensed for non-commercial use. See [The Terms of Use](https://www.nuscenes.org/terms-of-use)

### Downloading the dataset:

Before running the BEVFusion demo, users need to download ``data/nuscene.tar.gz``. After clone this repo, Users can achieve this using git-lfs commands:

```bash
git lfs install
git lfs fetch
git lfs checkout
```

### Extracting the dataset:

```bash
mkdir /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion/data/
cp data/nuscene.tar.gz /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion/data/
cd /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion/data/
tar -pxvf nuscene.tar.gz
```

Instructions for setting up the demo follow in the next section.


## **What is DS3D BEVFusion pipeline of Sensor Fusion**

The pipeline efficiently processes data from 6 cameras and 1 LiDAR, leveraging a pre-trained PyTorch [BEVFusion model](https://github.com/mit-han-lab/bevfusion). This model is optimized for NVIDIA GPUs using TensorRT and CUDA, as showcased by the [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion) project. Enhancing integration, a multi-modal inference module based on [PyTriton](https://github.com/triton-inference-server/pytriton) simplifies the incorporation of the Python BEVFusion model. The pipeline seamlessly employs ``ds3d::datatfiler`` for triton inference through gRPC. Ultimately, users can visualize the results by examining the ``ds3d::datamap`` with 6 camera views. The pipeline also projects LiDAR data and 3D bounding boxes into each view. Furthermore, the same LiDAR data is thoughtfully visualized in both a top view and a front view for enhanced comprehension.

### **Quick setup**

Follow the instructions in DeepStream documentation [DS3D Multimodal Sensor Fusion](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_3D_MultiModal_Lidar_Sensor_Fusion.html)

### **Detailed setup**

For a more detailed explanation of the BEVFusion pipeline with NuScenes calibration and other datasets, refer to [DS3D Multimodal Lidar Camera BEVFusion](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_3D_MultiModal_Lidar_Camera_BEVFusion.html)

## **Instructions to collect NuScenes dataset for this specific demo scene**

This dataset ``data/nuscene.tar.gz`` could be re-collected through the following scripts from official NuScenes dataset.

```bash
cd /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion
python3 python/triton_lmm/helper/nuscene_data_setup.py  --data_dir=dataset/nuscene \
--ds3d_fusion_workspace=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion --print_calibration
```

The script ``nuscene_data_setup.py`` is typically included within the DeepStream SDK installation (version 7.0 or later). You can search it by:

```bash
find /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion/python -name nuscene_data_setup.py
```

## **License of NuScenes dataset (data/nuscene.tar.gz)**

This tiny dataset ``data/nuscene.tar.gz`` is collected from the official nuScenes website, is licensed under
```bash
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (“CC BY-NC-SA 4.0”).
```
[term-of-use](https://www.nuscenes.org/terms-of-use)

## **Citation**

- BEVFusion model is originally from [github bevfusion](https://github.com/mit-han-lab/bevfusion)

```bibtex
@inproceedings{liu2022bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```

- nuScenes Dataset

```bibtex
@article{nuscenes2019,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and
          Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and
          Giancarlo Baldan and Oscar Beijbom},
  journal={arXiv preprint arXiv:1903.11027},
  year={2019}
}
```

```bibtex
@article{fong2021panoptic,
  title={Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking},
  author={Fong, Whye Kit and Mohan, Rohit and Hurtado, Juana Valeria and Zhou, Lubing and Caesar, Holger and
          Beijbom, Oscar and Valada, Abhinav},
  journal={arXiv preprint arXiv:2109.03805},
  year={2021}
}
```
