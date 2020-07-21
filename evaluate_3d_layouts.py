import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import cv2
import json
import argparse

from draw_custom import draw_label, _validate_colormap

w,h = (640,480)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pred', type=str)
parser.add_argument('--gt', type=str, default="./ScanNet_Layout_annotations/")
parser.add_argument('--out', type=str, default="./comparisons/")
parser.add_argument('--eval_2D', help="Evaluate in 2D", action='store_true')
parser.add_argument('--eval_3D', help="Evaluate in 3D", action='store_true')

args = parser.parse_args()

evaluate_3D = args.eval_3D
evaluate_2D = args.eval_2D

dataset_path = args.gt
pred_path = args.pred # Our
output_path = args.out

if not os.path.isdir(output_path):
    os.makedirs(output_path)

# evaluate_3D = True
# evaluate_2D = True
#
# dataset_path = "/media/sinisa/Sinisa_hdd_data/Sinisa_Projects/corridor_localisation/Datasets/ScanNetRL_testset/ECCV_set/ECCV_accepted_dataset_v2/merged/data/"
# pred_path = "/media/sinisa/Sinisa_hdd_data/Sinisa_Projects/corridor_localisation/Datasets/ScanNet_Layout/final_dataset/ours_2020/inference_results_depth_gt" # Our
# output_path = "comparison/"

scenes_list = os.listdir(dataset_path)

rmse_list = []
us_rmse_list = []

image_iou_list = []
image_pe_list = []
image_edgee_list = []


def sort_scenes(x):
    return int(x[6:9])
scenes_list = sorted(scenes_list)

for scene in scenes_list:
    # Folder variables
    print("Scene: ", scene)
    color_path = dataset_path + scene + "/color/"
    depth_path = dataset_path + scene + "/depth/"
    labels_json_path = dataset_path + scene + "/labels_json/"
    valid_masks_path = dataset_path + scene + "/valid_masks/"

    gt_layout_depth_path = dataset_path + scene + "/layout_depth/"
    gt_layout_planes_path = dataset_path + scene + "/layout_planes/"

    pred_label_json_folder = pred_path + "labels_json/" + scene + "/"
    pred_layout_depth_folder = pred_path + "layout_depth/" + scene + "/"

    pred_layout_depth_comparison_path = output_path + "layout_depth_comparison/" + scene + "/"
    if not os.path.isdir(pred_layout_depth_comparison_path):
        os.makedirs(pred_layout_depth_comparison_path)
    pred_layout_depth_comparison_path_final = output_path + "0000_final_3d_result.txt"

    pred_layout_2d_comparison_path = output_path + "layout_2d_comparison/" + scene + "/"
    if not os.path.isdir(pred_layout_2d_comparison_path):
        os.makedirs(pred_layout_2d_comparison_path)
    pred_layout_2d_comparison_path_final = output_path + "0000_final_2d_result.txt"


    def sort_images(x):
        return int(x[:-4])
    images = os.listdir(color_path)
    images.sort(key=sort_images)

    # Evaluate in 2D
    if evaluate_2D:
        for col_img_name in images:

            # Get color
            color_img_path = color_path + col_img_name
            col_img = plt.imread(color_img_path)
            col_img_h, col_img_w = (col_img.shape[0], col_img.shape[1])

            col_img = cv2.resize(col_img, (w, h))

            scale_x = float(w) / col_img_w
            scale_y = float(h) / col_img_h

            scale = np.array([scale_x, scale_y]).reshape(1, 1, 2)

            # Get .json labels GT
            llabel_json_path = labels_json_path + col_img_name.replace('.jpg', '.json')
            labelme_data = json.load(open(llabel_json_path))

            # Parse GT polys
            gt_polys_masks = []
            # gt_polys_types = []
            gt_polys_edges_mask = np.zeros((h, w))
            edge_thickness = 1
            for polygon_dict in labelme_data['shapes']:
                polygon = polygon_dict['points']
                # polygon_type = polygon_dict['label']
                polygon = np.array(polygon, dtype=np.float64)
                polygon = polygon.reshape((-1, 1, 2))
                polygon *= scale
                polygon = polygon.astype(np.int32)

                gt_poly_mask = np.zeros((h, w))
                cv2.fillPoly(gt_poly_mask, [polygon], color=[1.])
                gt_polys_masks.append(gt_poly_mask)
                # gt_polys_types.append(polygon_type)

                cv2.polylines(gt_polys_edges_mask, [polygon], isClosed=True, color=[1.], thickness=edge_thickness)


            def sortPolyBySize(mask):
                return mask.sum()


            gt_polys_masks.sort(key=sortPolyBySize, reverse=True)

            # Get .json labels predictions
            pred_llabel_json_path = pred_label_json_folder + col_img_name.replace('.jpg', '.json')
            pred_labelme_data = json.load(open(pred_llabel_json_path))

            # Parse predictions
            pred_polys_masks = []
            # pred_polys_types = []
            pred_polys_edges_mask = np.zeros((h, w))
            for polygon_dict in pred_labelme_data['shapes']:
                polygon = polygon_dict['points']
                # polygon_type = polygon_dict['label']
                polygon = np.array(polygon, dtype=np.float64)
                polygon = polygon.reshape((-1, 1, 2))
                polygon = polygon.astype(np.int32)

                pred_poly_mask = np.zeros((h, w))
                cv2.fillPoly(pred_poly_mask, [polygon], color=[1.])
                pred_polys_masks.append(pred_poly_mask)
                # pred_polys_types.append(polygon_type)

                cv2.polylines(pred_polys_edges_mask, [polygon], isClosed=True, color=[1.], thickness=edge_thickness)

            if len(pred_polys_masks) == 0.:
                pred_polys_edges_mask[edge_thickness:-edge_thickness, edge_thickness:-edge_thickness] = 1
                pred_polys_edges_mask = 1 - pred_polys_edges_mask

                pred_poly_mask = np.ones((h, w))
                pred_polys_masks = [pred_poly_mask]
                # pred_polys_types = [-1]

            pred_polys_masks_cand = copy.copy(pred_polys_masks)

            # Assign predictions to ground truth polygons
            best_pred_ind = []
            ordered_preds = []
            for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
                best_iou_score = 0.3
                best_pred_ind = None
                best_pred_poly_mask = None
                if len(pred_polys_masks_cand) == 0:
                    break
                for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand):
                    gt_pred_add = gt_poly_mask + pred_poly_mask
                    inter = np.equal(gt_pred_add, 2.).sum()
                    union = np.greater(gt_pred_add, 0.).sum()
                    iou_score = inter / union

                    if iou_score > best_iou_score:
                        best_iou_score = iou_score
                        best_pred_ind = pred_ind
                        best_pred_poly_mask = pred_poly_mask
                ordered_preds.append(best_pred_poly_mask)

                pred_polys_masks_cand = [pred_poly_mask for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand)
                                         if pred_ind != best_pred_ind]
                if best_pred_poly_mask is None:
                    continue

            ordered_preds += pred_polys_masks_cand
            class_num = max(len(ordered_preds), len(gt_polys_masks))
            colormap = _validate_colormap(None, class_num + 1)

            # Generate GT poly mask
            gt_layout_mask = np.zeros((h, w))
            gt_layout_mask_colored = np.zeros((h, w, 3))
            for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
                gt_layout_mask = np.maximum(gt_layout_mask, gt_poly_mask * (gt_ind + 1))
                gt_layout_mask_colored += gt_poly_mask[:, :, None] * colormap[gt_ind + 1]

            # Generate pred poly mask
            pred_layout_mask = np.zeros((h, w))
            pred_layout_mask_colored = np.zeros((h, w, 3))
            for pred_ind, pred_poly_mask in enumerate(ordered_preds):
                if pred_poly_mask is not None:
                    pred_layout_mask = np.maximum(pred_layout_mask, pred_poly_mask * (pred_ind + 1))
                    pred_layout_mask_colored += pred_poly_mask[:, :, None] * colormap[pred_ind + 1]

            # Calc IOU
            ious = []
            for layout_comp_ind in range(1, len(gt_polys_masks) + 1):
                inter = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                       np.equal(pred_layout_mask, layout_comp_ind)).sum()
                fp = np.logical_and(np.not_equal(gt_layout_mask, layout_comp_ind),
                                    np.equal(pred_layout_mask, layout_comp_ind)).sum()
                fn = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                    np.not_equal(pred_layout_mask, layout_comp_ind)).sum()
                union = inter + fp + fn
                iou = inter / union
                ious.append(iou)

            image_iou = sum(ious) / class_num
            image_iou_list.append(image_iou)

            # Calc PE
            image_pe = np.equal(gt_layout_mask, pred_layout_mask).sum() / (h * w)
            image_pe_list.append(image_pe)

            comparison_layout_2d_img_path = pred_layout_2d_comparison_path + col_img_name

            # Calc edge error

            # ignore edges at image borders
            img_bound_mask = np.zeros_like(pred_polys_edges_mask)
            img_bound_mask[10:-10, 10:-10] = 1

            pred_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - pred_polys_edges_mask)).astype(np.uint8),
                                                    cv2.DIST_L2, 3)
            gt_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - gt_polys_edges_mask)).astype(np.uint8),
                                                  cv2.DIST_L2, 3)

            chamfer_dist = pred_polys_edges_mask * gt_dist_trans + gt_polys_edges_mask * pred_dist_trans
            merror_edge = 0.5 * np.sum(chamfer_dist) / np.sum(
                np.greater(img_bound_mask * (gt_polys_edges_mask), 0))
            image_edgee_list.append(merror_edge)

            # Export visualization
            plt.figure(figsize=(30, 6), frameon=False)
            plt.subplot(151)
            plt.title("Input image", fontsize=20)
            plt.imshow(col_img)
            plt.subplot(152)
            plt.title("Prediction", fontsize=20)
            plt.imshow(np.clip(pred_layout_mask_colored, 0, 1))
            plt.subplot(153)
            plt.title("GT", fontsize=20)
            plt.imshow(np.clip(gt_layout_mask_colored, 0, 1))
            plt.subplot(154)
            plt.title("IOU: " + str(np.round(image_iou, decimals=3)) + ", PE: " + str(np.round(image_pe, decimals=3)),
                      fontsize=20)
            plt.imshow(np.equal(gt_layout_mask, pred_layout_mask).astype(np.float))
            plt.subplot(155)
            # plt.title("Chamf. Dist : " + str(np.round(chamfer_dist_error, decimals=3)), fontsize=20)
            plt.title("Edge Err. : " + str(np.round(merror_edge, decimals=3)), fontsize=20)
            plt.imshow(chamfer_dist.astype(np.float))
            plt.tight_layout()
            plt.savefig(comparison_layout_2d_img_path, dpi=200)
            # plt.show()
            plt.close()

    # Evaluate in 3D
    if evaluate_3D:
        for col_img_name in images:

            color_img_path = color_path + col_img_name
            col_img = plt.imread(color_img_path)
            col_img = cv2.resize(col_img, (w,h))

            depth_img_path = depth_path + col_img_name.replace('.jpg', '.png')
            depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.
            depth_img = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)

            gt_layout_depth_img_path = gt_layout_depth_path + col_img_name.replace('.jpg', '.npy')
            gt_layout_depth_img = np.load(gt_layout_depth_img_path)
            gt_layout_depth_img = cv2.resize(gt_layout_depth_img, (w,h), interpolation=cv2.INTER_NEAREST)
            gt_layout_depth_img_mask = np.greater(gt_layout_depth_img, 0.)

            gt_layout_depth_med = np.median(gt_layout_depth_img[gt_layout_depth_img_mask])

            pred_layout_depth_img_path = pred_layout_depth_folder + col_img_name.replace('.jpg', '.npy')
            pred_layout_depth_img = np.load(pred_layout_depth_img_path)
            pred_layout_depth_img = cv2.resize(pred_layout_depth_img, (w, h), interpolation=cv2.INTER_NEAREST)

            pred_layout_depth_med = np.median(pred_layout_depth_img)

            # Calc MSE
            ms_error_image = gt_layout_depth_img_mask * (pred_layout_depth_img - gt_layout_depth_img) ** 2
            rmse = np.sqrt(np.sum(ms_error_image) / np.sum(gt_layout_depth_img_mask))

            # Calc up to scale MSE
            if np.isnan(pred_layout_depth_med) or pred_layout_depth_med == 0:
                d_scale = 1.
            else:
                d_scale = gt_layout_depth_med / pred_layout_depth_med
            us_ms_error_image = gt_layout_depth_img_mask * (d_scale * pred_layout_depth_img - gt_layout_depth_img) ** 2
            us_rmse = np.sqrt(np.sum(us_ms_error_image) / np.sum(gt_layout_depth_img_mask))

            # Export comparison
            comparison_layout_img_path = pred_layout_depth_comparison_path + col_img_name
            plt.figure(figsize=(30,6), frameon = False)
            plt.subplot(141)
            plt.title("Input image", fontsize=20)
            plt.imshow(col_img)
            plt.subplot(142)
            plt.title("Prediction", fontsize=20)
            plt.imshow(pred_layout_depth_img, vmin=0., vmax=10.)
            plt.subplot(143)
            plt.title("GT", fontsize=20)
            plt.imshow(gt_layout_depth_img, vmin=0., vmax=10.)
            plt.subplot(144)
            plt.title("RMSE: " + str(np.round(rmse, decimals=3)), fontsize=20)
            plt.imshow(ms_error_image)
            plt.tight_layout()
            plt.savefig(comparison_layout_img_path, dpi = 200)
            # plt.show()
            plt.close()

            # Append
            rmse_list.append(rmse)
            us_rmse_list.append(us_rmse)


if evaluate_2D:
    mean_iou = sum(image_iou_list) / len(image_iou_list) * 100
    mean_pe = (1. - sum(image_pe_list) / len(image_pe_list)) * 100
    mean_edgee = sum(image_edgee_list) / len(image_edgee_list)
    print("IOU mean: ", np.round(mean_iou, decimals=3))
    print("IOU std: ", np.round(np.std(np.array(image_iou_list) * 100), decimals=3))
    print("PE mean: ", np.round(mean_pe, decimals=3))
    print("PE std: ", np.round(np.std(np.array(image_pe_list) * 100), decimals=3))
    print("EE mean: ", np.round(mean_edgee, decimals=3))
    print("EE std: ", np.round(np.std(np.array(image_edgee_list)), decimals=3))
    with open(pred_layout_2d_comparison_path_final, "w") as f:
        iou_string = "IOU mean: " + str(np.round(mean_iou, decimals=3)) + str(" % \n")
        iou_string = "IOU std: " + str(np.round(np.std(np.array(image_iou_list)), decimals=3)) + str(" % \n")
        pe_string = "PE mean: " + str(np.round(mean_pe, decimals=3)) + str(" % \n")
        pe_string = "PE std: " + str(np.round(np.std(np.array(image_pe_list)), decimals=3)) + str(" % \n")
        edgee_string = "EE mean: " + str(np.round(mean_edgee, decimals=3)) + str("\n")
        edgee_string = "EE std: " + str(np.round(np.std(np.array(image_edgee_list)), decimals=3)) + str("\n")
        f.writelines([iou_string, pe_string, edgee_string])

if evaluate_3D:
    print("RMSE mean: ", np.round(sum(rmse_list) / len(rmse_list), decimals=3))
    print("RMSE std: ", np.round(np.std(np.array(rmse_list)), decimals=3))
    print("Up-to-Scale RMSE mean: ", np.round(sum(us_rmse_list) / len(us_rmse_list), decimals=3))
    print("Up-to-Scale RMSE std: ", np.round(np.std(np.array(us_rmse_list)), decimals=3))
    with open(pred_layout_depth_comparison_path_final, "w") as f:
        rmse_string = "RMSE mean: " + str(np.round(sum(rmse_list) / len(rmse_list), decimals=3)) + str("\n")
        rmse_string_std = "RMSE: std" + str(np.round(np.std(np.array(rmse_list)), decimals=3)) + str("\n")
        rmse_uts_string = "Up-to-Scale RMSE mean: " + str(np.round(sum(us_rmse_list) / len(us_rmse_list), decimals=3)) + str("\n")
        rmse_uts_string_std = "Up-to-Scale RMSE mean: " + str(np.round(sum(us_rmse_list) / len(us_rmse_list), decimals=3)) + str("\n")
        rmse_check_string = "RMSE check: std" + str(np.round(np.std(np.array(us_rmse_list)), decimals=3)) + str("\n")
        f.writelines([rmse_string, rmse_check_string])




