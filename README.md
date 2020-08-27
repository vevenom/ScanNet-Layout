# ScanNet-Layout Dataset

<p align="center">
  <img src="https://github.com/vevenom/ScanNet-Layout/blob/master/image_ex/ex.gif" alt="ScanNet-Layout" width="300">
</p>

We introduce the ScanNet-Layout dataset for benchmarking general 3D room layout estimation from single view. The benchmark includes 293 views from the [ScanNet](http://www.scan-net.org/)
 dataset that span different layout settings, are equally distributed to represent both cuboid and general room
layouts, challenging views that are neglected in existing room layout datasets,
and in some cases we include similar viewpoints to evaluate effects of noise (e.g.
motion blur). Our benchmark supports evaluation metrics both in 2D and 3D. Please refer to our original paper 
[General 3D Room Layout from a Single View by Render-and-Compare](https://arxiv.org/abs/2001.02149) for more information.

#### Note

If you are interested into the implementation of our approach, General 3D Room Layout from a Single View by Render-and-Compare, please visit:

https://github.com/vevenom/RoomLayout3D_RandC



## Data

The dataset can be obtained here: [Dataset Link](https://files.icg.tugraz.at/f/2cefdc3a5b9a48d7aaaa/?dl=1)

After downloading the dataset, you can extract it into GT_PATH folder (default ./ScanNet_Layout_annotations). We assume the following folder hierarchy:

* |--GT_PATH  
  * |----SCENE_ID  
    * |--------color - original color images from the ScanNet dataset (*.jpg)  
    * |--------depth - corresponding depth maps from the ScanNet dataset (*.png)  
    * |--------labels_json - corresponding 2D polygon annotations (*.json)  
    * |--------layout_depth - corresponding generated layout depth maps annotations clipped to 10m (*.npy)  
    * |--------layout_depth_vis - corresponding generated layout depth maps visualisations (*.jpg)  
    * |--------polygon_vis - corresponding annotated polygons visualisations (*.jpg)  
    * |--------valid_masks - corresponding masks used for the depth map generation (*.json)


## Running the Evaluation Script

For evaluating the quality of recovered layouts you can run the following script:

```
 python evaluate_3d_layouts.py --pred PRED_PATH --gt GT_PATH --out COMP_OUTPUT_PATH --eval_2D --eval_3D
```

PRED_PATH points to the prediction folder.

GT_PATH points to the ground truth folder. The default value for GT_PATH is ./ScanNet_Layout_annotations.

COMP_OUTPUT_PATH points to the output folder.

eval_2D, eval_3D specify whether to evaluate the quality of layouts in 2D and 3D. 

#### Predictions Format

The script assumes the following hierarchy for PRED_PATH:

* |--PRED_PATH
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
{"shapes": 
[
{"points": [[275, 379], [283, 0], [0, 0], [0, 479], [94, 479], [275, 379]]}, 
{"points": [[275, 379], [283, 0], [639, 0], [639, 479], [519, 479], [275, 379]]}, 
{"points": [[275, 379], [94, 479], [519, 479], [275, 379]]}
]
}
```

Predicted *.npy files in layout_depth folder contain layout depth maps.

## Citation

If you use this dataset, please cite:

```
@inproceedings{dai2017scannet,  
    title={ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes},  
    author={Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},  
    booktitle = {Proc. Computer Vision and Pattern Recognition (CVPR), IEEE},  
    year = {2017}  
}
```

```
@article{stekovic2020general,
  title={{General 3D Room Layout from a Single View by Render-and-Compare}},  
  author={Stekovic, Sinisa and Hampali, Shreyas and Rad, Mahdi and Deb Sarkar, Sayan and Fraundorfer, Friedrich and Lepetit, Vincent},  
  journal={{European Conference on Computer Vision (ECCV)}},  
  year={2020}  
}
```

# ScanNet Terms of Use

As our dataset is based on images from the ScanNet dataset, you should also agree to the [ScanNet Terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf):

> If you would like to download the ScanNet data, please fill out an agreement to the ScanNet Terms of Use and send it to scannet@googlegroups.com.

For more information, please visit [ScanNet/ScanNet](https://github.com/ScanNet/ScanNet).
