# ScanNet-Layout Dataset

<p align="center">
  <img src="" alt="ScanNet-Layout">
</p>

We introduce a dataset for evaluating the quality of general 3D room layouts. The dataset includes views from the [ScanNet](http://www.scan-net.org/)
 dataset that span different layout settings, are equally distributed to represent both cuboid and general room
layouts, challenging views that are neglected in previous room layout datasets,
and in some cases we include similar viewpoints to evaluate effects of noise (e.g.
motion blur)

## Data

The dataset can be obtained here: [Dataset Link](https://files.icg.tugraz.at/f/2cefdc3a5b9a48d7aaaa/?dl=1)

After downloading the dataset, you can extract it into default ScanNet_Layout_annotations folder. We assume the following folder hierarchy:

* |--ScanNet_Layout_annotations  
  * |----SCENE_ID  
    * |--------color - These are the original color images from the ScanNet dataset (*.jpg)  
    * |--------depth - These are the corresponding depth maps from the ScanNet dataset (*.png)  
    * |--------labels_json - These are the corresponding 2D polygon annotations (*.json)  
    * |--------layout_depth - These are the corresponding generated layout depth maps annotations clipped to 10m (*.npy)  
    * |--------layout_depth_vis - These are the corresponding generated layout depth maps visualisations (*.jpg)  
    * |--------polygon_vis - These are the corresponding annotated polygons visualisations (*.jpg)  
    * |--------valid_masks - These are the corresponding masks used for the depth map generation (*.json)


## Evaluation scripts

For evaluating the quality of recovered layouts you can run the following script:

```
 python evaluate_3d_layouts.py --pred PRED_PATH --gt GT_PATH --out COMP_OUTPUT_PATH --eval_2D --eval_3D
```

PRED_PATH points to the prediction folder

GT_PATH points to the ground truth folder. The default value for GT_PATH is ./ScanNet_Layout_annotations

COMP_OUTPUT_PATH points to the output folder.

where eval_2D, eval_3D specify whether to evaluate the quality of layouts in 2D and 3D. 


The script assumes the following hierarchy for PRED_PATH:

* |--ScanNet_Layout_annotations  
    * |----labels_json
      * |------SCENE_ID  
        * |------ *.json  
    * |----layout_depth 
      * |------SCENE_ID  
        * |------ *.npy      
        
Predicted *.json files in labels_json folder contain the 2D polygons and should have the following format:
```
{"shapes": [{"points": LIST_OF_POINTS_IN_POLY1}, 
            {"points": LIST_OF_POINTS_IN_POLY2},
            {"points": LIST_OF_POINTS_IN_POLY_N}]}
```

e.g.:

```
{"shapes": [{"points": [[275, 379], [283, 0], [0, 0], [0, 479], [94, 479], [275, 379]], "shape_type": "polygon", "plane": [0.7948528396971843, 0.17106515939675068, -0.5821904108329907, 1.660079846881285], "label": "wall"}, {"points": [[275, 379], [283, 0], [639, 0], [639, 479], [519, 479], [275, 379]], "shape_type": "polygon", "plane": [-0.5562020385107806, 0.1743546045935068, -0.8125513917368514, 2.0026558618881487], "label": "wall"}, {"points": [[275, 379], [94, 479], [519, 479], [275, 379]], "shape_type": "polygon", "plane": [0.056890187628746154, -0.960978788526898, -0.2707088372272719, 1.3907352987241348], "label": "floor"}]}
```

Predicted *.npy files in layout_depth folder contain layout depth maps.
