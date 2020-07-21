# ScanNet-Layout Dataset

## Data

Dataset Link:

[a link](https://files.icg.tugraz.at/f/2cefdc3a5b9a48d7aaaa/?dl=1)

After downloading the dataset, you can extract it into default ScanNet_Layout_annotations folder. We assume the following folder hierarchy:

|--ScanNet_Layout_annotations
|----SCENE_ID
|--------color # These are the original color images from the ScanNet dataset (*.jpg)
|--------depth # These are the corresponding depth maps from the ScanNet dataset (*.png)
|--------labels_json # These are the corresponding 2D polygon annotations (*.json)
|--------layout_depth # These are the corresponding generated layout depth maps annotations (*.npy)
|--------layout_depth_vis # These are the corresponding generated layout depth maps visualisations (*.jpg)
|--------polygon_vis # These are the corresponding annotated polygons visualisations (*.jpg)
|--------valid_masks # These are the corresponding masks used for the depth map generation (*.json)


## Evaluation scripts

?
