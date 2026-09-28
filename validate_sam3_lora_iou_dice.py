#!/usr/bin/env python3
"""
Validation script for SAM3 LoRA model using foreground-mask IoU and Dice.
"""

import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import contextlib

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights, count_parameters

from torchvision.transforms import v2

# COCO/RLE utilities are still used to load segmentation annotations.
from sam3.eval.cgf1_eval import CGF1Evaluator, COCOCustom
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
from sam3.train.masks_ops import rle_encode

# Import SAM3's NMS
from sam3.perflib.nms import nms_masks

class COCOSegmentDataset(Dataset):
    """Dataset class for COCO format segmentation data"""
    def __init__(self, cfg, data_dir, split="train"):
        """
        Args:
            data_dir: Root directory containing train/valid/test folders
            split: One of 'train', 'valid', 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

        # Load COCO annotations
        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build index: image_id -> image info
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = sorted(list(self.images.keys()))

        # Build index: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Load categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        print(f"Loaded COCO dataset: {split} split")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Annotations: {len(self.coco_data['annotations'])}")
        print(f"  Categories: {self.categories}")

        self.resolution = cfg["image_size"]
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.split_dir / img_info['file_name']
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get bbox - format is [x, y, width, height] in COCO format
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # Get class name from category_id
            category_id = ann.get("category_id", 0)
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)

            # Convert from COCO [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = bbox_coco
            box_tensor = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)

            # Scale box to resolution
            box_tensor[0] *= scale_w
            box_tensor[2] *= scale_w
            box_tensor[1] *= scale_h
            box_tensor[3] *= scale_h

            # IMPORTANT: Normalize boxes to [0, 1] range (required by SAM3 loss functions)
            box_tensor /= self.resolution

            # Handle segmentation mask (polygon or RLE format)
            segment = None
            segmentation = ann.get("segmentation", None)

            if segmentation:
                try:
                    # Check if it's RLE format (dict) or polygon format (list)
                    if isinstance(segmentation, dict):
                        # RLE format: {"counts": "...", "size": [h, w]}
                        mask_np = mask_utils.decode(segmentation)
                    elif isinstance(segmentation, list):
                        # Polygon format: [[x1, y1, x2, y2, ...], ...]
                        # Convert polygon to RLE, then decode
                        rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                        rle = mask_utils.merge(rles)
                        mask_np = mask_utils.decode(rle)
                    else:
                        print(f"Warning: Unknown segmentation format: {type(segmentation)}")
                        segment = None
                        continue

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor

                except Exception as e:
                    print(f"Warning: Error processing mask for image {img_id}, ann {i}: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2]-box_tensor[0])*(box_tensor[3]-box_tensor[1]),
                object_id=i,
                segment=segment
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct Queries - one per unique category
        # Each query maps to only the objects of that category
        from collections import defaultdict

        # Group object IDs by their class name
        class_to_object_ids = defaultdict(list)
        for obj, class_name in zip(objects, object_class_names):
            class_to_object_ids[class_name.lower()].append(obj.object_id)

        # Create one query per category
        queries = []
        if len(class_to_object_ids) > 0:
            for query_text, obj_ids in class_to_object_ids.items():
                query = FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=0,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1
                    )
                )
                queries.append(query)
        else:
            # No annotations: create a single generic query
            query = FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.15):
    """
    Merge overlapping masks that likely represent the same object (e.g., crack segments).

    This is more aggressive than NMS - it MERGES masks instead of suppressing them.
    Useful for cracks where model splits one crack into many segments.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute IoU
            intersection = (current_mask & binary_masks[j]).sum().item()
            union = (current_mask | binary_masks[j]).sum().item()
            iou = intersection / union if union > 0 else 0

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def apply_sam3_nms(pred_logits, pred_masks, pred_boxes, prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100):
    """
    Apply SAM3's standard NMS pipeline to filter predictions.

    Args:
        pred_logits: [N, 1] logits
        pred_masks: [N, H, W] mask logits
        pred_boxes: [N, 4] boxes in normalized format
        prob_threshold: Score threshold for filtering (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: IoU threshold for NMS (default: 0.7, SAM3 uses 0.5-0.7)
        max_detections: Maximum detections to keep (default: 100)

    Returns:
        Tuple of (filtered_masks, filtered_scores, filtered_boxes)
    """
    if len(pred_logits) == 0:
        return pred_masks[:0], pred_logits[:0].squeeze(-1), pred_boxes[:0]

    # Convert logits to probabilities
    pred_probs = torch.sigmoid(pred_logits).squeeze(-1)  # [N]

    # Convert mask logits to binary masks (sigmoid + threshold)
    pred_masks_sigmoid = torch.sigmoid(pred_masks)  # [N, H, W]
    pred_masks_binary = pred_masks_sigmoid > 0.5  # [N, H, W]

    # Apply SAM3's NMS
    # nms_masks expects: pred_probs [N], pred_masks [N, H, W], prob_threshold, iou_threshold
    # Returns: keep mask [N] of booleans
    keep_mask = nms_masks(
        pred_probs=pred_probs,
        pred_masks=pred_masks_binary.float(),  # NMS expects float masks
        prob_threshold=prob_threshold,
        iou_threshold=nms_iou_threshold
    )

    # Filter predictions
    filtered_masks = pred_masks_sigmoid[keep_mask]  # Keep sigmoid masks for later
    filtered_scores = pred_probs[keep_mask]
    filtered_boxes = pred_boxes[keep_mask]

    # Top-K selection by score
    if max_detections > 0 and len(filtered_scores) > max_detections:
        top_k_scores, top_k_indices = torch.topk(filtered_scores, k=max_detections, largest=True)
        filtered_masks = filtered_masks[top_k_indices]
        filtered_scores = top_k_scores
        filtered_boxes = filtered_boxes[top_k_indices]

    return filtered_masks, filtered_scores, filtered_boxes


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288,
                                       prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100,
                                       merge_cracks=False, merge_iou_threshold=0.15):
    """
    Convert model predictions to COCO format using SAM3's NMS pipeline.

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs
        resolution: Resolution for box scaling (default: 288)
        prob_threshold: Score threshold (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: NMS IoU threshold (default: 0.7)
        max_detections: Max detections per image (default: 100)
        merge_cracks: If True, merge overlapping segments instead of NMS suppression (default: False)
        merge_iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)
    """
    coco_predictions = []
    pred_id = 0

    if merge_cracks:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using CRACK MERGING mode: prob_threshold={prob_threshold}, merge_iou={merge_iou_threshold}, max_dets={max_detections}")
        print(f"[INFO] This will MERGE overlapping crack segments instead of suppressing them")
    else:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using SAM3 NMS: prob_threshold={prob_threshold}, nms_iou={nms_iou_threshold}, max_dets={max_detections}")

    for img_id, preds in tqdm(zip(image_ids, predictions_list), total=len(predictions_list), desc="Converting predictions"):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        logits = preds['pred_logits']  # [N, 1]
        boxes = preds['pred_boxes']    # [N, 4]
        masks = preds['pred_masks']    # [N, H, W]
        # print(logits.shape, boxes.shape, masks.shape)

        if merge_cracks:
            # Step 1: Filter by score threshold
            pred_probs = torch.sigmoid(logits).squeeze(-1)  # [N]
            valid_mask = pred_probs > prob_threshold
            # print(torch.sum(valid_mask))

            filtered_masks = masks[valid_mask]
            filtered_scores = pred_probs[valid_mask]
            filtered_boxes = boxes[valid_mask]

            if len(filtered_masks) > 0:
                # Step 2: Convert masks to binary
                pred_masks_sigmoid = torch.sigmoid(filtered_masks)
                pred_masks_binary = (pred_masks_sigmoid > 0.5)

                # Step 3: MERGE overlapping crack segments
                merged_masks, merged_scores, merged_boxes = merge_overlapping_masks(
                    pred_masks_binary.cpu(),
                    filtered_scores.cpu(),
                    filtered_boxes.cpu(),
                    iou_threshold=merge_iou_threshold
                )

                # Step 4: Top-K selection by score
                if max_detections > 0 and len(merged_scores) > max_detections:
                    top_k_scores, top_k_indices = torch.topk(merged_scores, k=max_detections, largest=True)
                    merged_masks = merged_masks[top_k_indices]
                    merged_scores = top_k_scores
                    merged_boxes = merged_boxes[top_k_indices]

                # Return merged results (already binary)
                filtered_masks = merged_masks.float()  # Already binary, just convert to float
                filtered_scores = merged_scores
                filtered_boxes = merged_boxes
            else:
                filtered_masks = torch.tensor([])
                filtered_scores = torch.tensor([])
                filtered_boxes = torch.tensor([])
        else:
            # Apply SAM3's NMS pipeline (standard suppression)
            filtered_masks, filtered_scores, filtered_boxes = apply_sam3_nms(
                pred_logits=logits,
                pred_masks=masks,
                pred_boxes=boxes,
                prob_threshold=prob_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections
            )
        # print(len(filtered_masks))
        if len(filtered_masks) > 0:
            # Convert filtered masks to binary for RLE encoding
            binary_masks = (filtered_masks > 0.5).cpu()
            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, filtered_scores.cpu().tolist(), filtered_boxes.cpu().tolist())):
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution
                # print(box)

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from dataset.

    OPTIMIZATION: Downsample GT masks to 288×288 to match prediction resolution.
    """
    print(f"\n[INFO] Creating COCO ground truth (downsampling to {mask_resolution}×{mask_resolution})...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in tqdm(list(indices), desc="Creating GT"):
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True
        })

        datapoint = dataset[idx]

        for obj in datapoint.images[0].objects:
            # Scale boxes to mask_resolution
            box = obj.bbox * mask_resolution
            x1, y1, x2, y2 = box.tolist()
            x, y, w, h = x1, y1, x2-x1, y2-y1

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    print(f"[INFO] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")

    return coco_gt


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset, model_resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # Scale boxes from normalized [0,1] to original size
            box = obj.bbox  # Already in [0,1] normalized coordinates
            x1, y1, x2, y2 = box.tolist()

            # Convert to original image coordinates
            x1_orig = x1 * orig_w
            y1_orig = y1 * orig_h
            x2_orig = x2 * orig_w
            y2_orig = y2 * orig_h

            # Convert to COCO format [x, y, w, h]
            x, y = x1_orig, y1_orig
            w = x2_orig - x1_orig
            h = y2_orig - y1_orig

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


def move_to_device(obj, device):
    """Recursively move objects to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            val = getattr(obj, field)
            setattr(obj, field, move_to_device(val, device))
        return obj
    return obj


def _aggregate_gt_masks(find_target, image_size):
    """Merge all GT object masks into one foreground mask per image."""
    device = find_target.segments.device
    batch_size = int(find_target.num_boxes.shape[0])
    gt_masks = torch.zeros((batch_size, image_size, image_size), dtype=torch.bool, device=device)

    cursor = 0
    for batch_idx, num_boxes in enumerate(find_target.num_boxes.tolist()):
        count = int(num_boxes)
        if count <= 0:
            continue

        segments = find_target.segments[cursor:cursor + count]
        valid = find_target.is_valid_segment[cursor:cursor + count]
        cursor += count

        if segments.numel() == 0:
            continue
        if valid.numel() > 0:
            segments = segments[valid]
        if segments.numel() == 0:
            continue

        if segments.shape[-2:] != (image_size, image_size):
            segments = torch.nn.functional.interpolate(
                segments.float().unsqueeze(1),
                size=(image_size, image_size),
                mode="nearest",
            ).squeeze(1) > 0.5

        gt_masks[batch_idx] = segments.bool().any(dim=0)

    return gt_masks


def _aggregate_pred_masks(outputs, image_size, prob_threshold=0.3, mask_threshold=0.5):
    """Merge selected predicted object masks into one foreground mask per image."""
    logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]

    scores = torch.sigmoid(logits).squeeze(-1)
    mask_probs = torch.sigmoid(pred_masks)

    if mask_probs.shape[-2:] != (image_size, image_size):
        mask_probs = torch.nn.functional.interpolate(
            mask_probs,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    keep = scores > prob_threshold
    binary_masks = mask_probs > mask_threshold
    return (binary_masks & keep[:, :, None, None]).any(dim=1)


def compute_iou_dice_from_outputs(outputs, find_target, image_size, prob_threshold=0.3, mask_threshold=0.5, eps=1e-6):
    """Compute per-image foreground IoU/Dice from SAM3 final outputs and GT masks."""
    pred_foreground = _aggregate_pred_masks(
        outputs,
        image_size=image_size,
        prob_threshold=prob_threshold,
        mask_threshold=mask_threshold,
    )
    gt_foreground = _aggregate_gt_masks(find_target, image_size)

    intersection = (pred_foreground & gt_foreground).flatten(1).sum(dim=1).float()
    pred_area = pred_foreground.flatten(1).sum(dim=1).float()
    gt_area = gt_foreground.flatten(1).sum(dim=1).float()
    union = pred_area + gt_area - intersection

    both_empty = union == 0
    iou = torch.where(both_empty, torch.ones_like(union), (intersection + eps) / (union + eps))
    dice_den = pred_area + gt_area
    dice = torch.where(both_empty, torch.ones_like(dice_den), (2 * intersection + eps) / (dice_den + eps))

    return iou.detach(), dice.detach()


def _overlay_mask_on_rgb(rgb_image, mask, color, alpha=0.45):
    """Overlay a boolean mask on a PIL RGB image."""
    image = rgb_image.convert("RGB")
    mask_np = mask.detach().cpu().numpy().astype(bool)
    base = np.asarray(image).astype(np.float32).copy()
    overlay_color = np.asarray(color, dtype=np.float32)
    base[mask_np] = base[mask_np] * (1.0 - alpha) + overlay_color * alpha
    return PILImage.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _draw_label(image, text):
    """Add a compact label bar to the top of an image."""
    image = image.convert("RGB")
    width, height = image.size
    bar_height = 34
    out = PILImage.new("RGB", (width, height + bar_height), (24, 24, 24))
    out.paste(image, (0, bar_height))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((8, 10), text, fill=(255, 255, 255), font=font)
    return out


def save_prediction_panel(rgb_image, gt_mask, pred_mask, iou, dice, output_path):
    """Save RGB | GT | Pred panel with per-image IoU/Dice."""
    rgb_panel = _draw_label(rgb_image, "RGB")
    gt_panel = _draw_label(_overlay_mask_on_rgb(rgb_image, gt_mask, color=(0, 220, 80)), "GT")
    pred_label = f"Pred  IoU={iou:.4f}  Dice={dice:.4f}"
    pred_panel = _draw_label(_overlay_mask_on_rgb(rgb_image, pred_mask, color=(255, 80, 40)), pred_label)

    width, height = rgb_panel.size
    canvas = PILImage.new("RGB", (width * 3, height), (0, 0, 0))
    canvas.paste(rgb_panel, (0, 0))
    canvas.paste(gt_panel, (width, 0))
    canvas.paste(pred_panel, (width * 2, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def validate(config_path, weights_path, val_data_dir, num_samples=None,
             prob_threshold=0.3, nms_iou=0.7, merge_cracks=False, merge_iou=0.15,
             use_base_model=False, mask_threshold=0.5,
             save_predictions=False, prediction_output_dir=None):
    """Run validation with foreground-mask IoU and Dice.

    Args:
        config_path: Path to config file. Optional for use_base_model=True.
        weights_path: Path to LoRA weights. Not required if use_base_model=True.
        val_data_dir: Direct path to validation data directory containing _annotations.coco.json
                      (e.g., /workspace/data2/valid)
        num_samples: Optional limit for number of samples (for debugging)
        use_base_model: If True, use original SAM3 model without LoRA (default: False)
        prob_threshold: Minimum object probability used when selecting predicted masks.
        mask_threshold: Pixel probability threshold used to binarize selected masks.
        save_predictions: If True, save RGB|GT|Pred visualization panels.
        prediction_output_dir: Output directory for saved panels.

    Example (with LoRA):
        validate(
            config_path="configs/full_lora_config.yaml",
            weights_path="outputs/sam3_lora_full/best_lora_weights.pt",
            val_data_dir="/workspace/data2/valid"
        )

    Example (base SAM3 model):
        validate(
            config_path=None,
            weights_path=None,
            val_data_dir="/workspace/data2/valid",
            use_base_model=True
        )
    """
    if not use_base_model and (config_path is None or weights_path is None):
        raise ValueError("config_path and weights_path are required when use_base_model=False")

    if config_path is None:
        config = {
            "data": {"image_size": 1008},
            "training": {"batch_size": 1},
        }
        print("No config provided for base SAM3 validation; using image_size=1008, batch_size=1.")
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    print("\nBuilding SAM3 model...")
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
        img_size=config["data"]["image_size"]
    )

    # Load config for batch_size and other settings
    if use_base_model:
        # Use original SAM3 model without LoRA
        print("Using original SAM3 model (no LoRA)")
        stats = count_parameters(model)
        print(f"Total params: {stats['total_parameters']:,}")
        # Use default batch_size for base model
        batch_size = 1
    else:
         # Apply LoRA and load weights
        print("Applying LoRA configuration...")
        lora_cfg = config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        model = apply_lora_to_model(model, lora_config)

        # Load weights
        print(f"\nLoading LoRA weights from {weights_path}...")
        load_lora_weights(model, weights_path)

        stats = count_parameters(model)
        print(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        batch_size = config["training"].get("batch_size", 1)

    model.to(device)
    model.eval()

    # Load validation data directly from the specified directory
    print(f"\nLoading validation data from {val_data_dir}...")

    # Load COCO annotations directly
    from pathlib import Path
    ann_file = Path(val_data_dir) / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    # Create a simple dataset class that loads from the directory directly
    class DirectCOCODataset(COCOSegmentDataset):
        def __init__(self, cfg, data_dir):
            self.data_dir = Path(data_dir)
            self.split_dir = self.data_dir

            # Load COCO annotations
            ann_file = self.split_dir / "_annotations.coco.json"
            if not ann_file.exists():
                raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

            with open(ann_file, 'r') as f:
                self.coco_data = json.load(f)

            # Build index: image_id -> image info
            self.images = {img['id']: img for img in self.coco_data['images']}
            self.image_ids = sorted(list(self.images.keys()))

            # Build index: image_id -> list of annotations
            self.img_to_anns = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)

            # Load categories
            self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
            print(f"Loaded COCO dataset from {data_dir}")
            print(f"  Images: {len(self.image_ids)}")
            print(f"  Annotations: {len(self.coco_data['annotations'])}")
            print(f"  Categories: {self.categories}")

            self.resolution = cfg["image_size"]
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    val_ds = DirectCOCODataset(config["data"], val_data_dir)

    if num_samples:
        print(f"\n[INFO] Limiting validation to {num_samples} samples for debugging")

    prediction_output_path = None
    if save_predictions:
        prediction_output_path = Path(
            prediction_output_dir
            or Path(val_data_dir).parent / "iou_dice_predictions"
        )
        prediction_output_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving prediction panels to: {prediction_output_path}")

    def collate_fn(batch):
        return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,  # Enable parallel data loading
        pin_memory=True  # Faster GPU transfer
    )

    # Run validation
    print("\n" + "="*80)
    print("RUNNING IOU/DICE VALIDATION")
    print("="*80)

    val_ious = []
    val_dices = []

    # Use automatic mixed precision for faster inference
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(val_loader, desc="Validation")):
            if num_samples and batch_idx * batch_size >= num_samples:
                break

            input_batch = batch_dict["input"]
            input_batch = move_to_device(input_batch, device)

            # Forward pass with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs_list = model(input_batch)
            else:
                outputs_list = model(input_batch)
            # torch.Size([4, 200, 1]) torch.Size([4, 200, 4]) torch.Size([4, 200, 160, 160])
            # print(outputs_list[-1]["pred_logits"].shape, outputs_list[-1]["pred_boxes"].shape, outputs_list[-1]["pred_masks"].shape)
            # print(len(outputs_list)) # 1

            # Extract final predictions and compute foreground metrics.
            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_stage = list(outputs_iter)[-1]
                final_outputs = final_stage[-1]

                batch_iou, batch_dice = compute_iou_dice_from_outputs(
                    final_outputs,
                    input_batch.find_targets[0],
                    image_size=config["data"]["image_size"],
                    prob_threshold=prob_threshold,
                    mask_threshold=mask_threshold,
                )
                val_ious.extend(batch_iou.cpu().tolist())
                val_dices.extend(batch_dice.cpu().tolist())

                if save_predictions:
                    pred_masks = _aggregate_pred_masks(
                        final_outputs,
                        image_size=config["data"]["image_size"],
                        prob_threshold=prob_threshold,
                        mask_threshold=mask_threshold,
                    )
                    gt_masks = _aggregate_gt_masks(
                        input_batch.find_targets[0],
                        image_size=config["data"]["image_size"],
                    )

                    actual_batch_size = pred_masks.shape[0]
                    for sample_idx in range(actual_batch_size):
                        global_idx = batch_idx * batch_size + sample_idx
                        if num_samples and global_idx >= num_samples:
                            continue

                        image_id = val_ds.image_ids[global_idx]
                        file_name = val_ds.images[image_id]["file_name"]
                        stem = Path(file_name).stem
                        out_file = prediction_output_path / f"{global_idx:06d}_{stem}_iou_dice.png"

                        save_prediction_panel(
                            rgb_image=input_batch.raw_images[sample_idx],
                            gt_mask=gt_masks[sample_idx],
                            pred_mask=pred_masks[sample_idx],
                            iou=float(batch_iou[sample_idx].item()),
                            dice=float(batch_dice[sample_idx].item()),
                            output_path=out_file,
                        )

    mean_iou = float(np.mean(val_ious)) if val_ious else 0.0
    mean_dice = float(np.mean(val_dices)) if val_dices else 0.0

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Images evaluated: {len(val_ious)}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Prediction prob threshold: {prob_threshold:.3f}")
    print(f"Mask threshold: {mask_threshold:.3f}")
    if save_predictions:
        print(f"Prediction panels: {prediction_output_path}")
    print("="*80)

    return {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "num_images": len(val_ious),
        "prediction_output_dir": str(prediction_output_path) if prediction_output_path else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone validation script for SAM3 LoRA model with foreground-mask IoU/Dice"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (for LoRA settings). Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights file. Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="Direct path to validation data directory containing _annotations.coco.json (e.g., /workspace/data2/valid)"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use original SAM3 model without LoRA (for baseline comparison)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit validation to N samples (for debugging)"
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for filtering predictions (default: 0.3)"
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask probability threshold for binarizing predictions (default: 0.5)"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable aggressive merging of overlapping segments (recommended for crack detection)"
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.15,
        help="IoU threshold for merging overlapping predictions (default: 0.15, lower = more aggressive)"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save RGB | GT | Pred visualization panels with per-image IoU/Dice"
    )
    parser.add_argument(
        "--prediction-output-dir",
        type=str,
        default=None,
        help="Directory for saved prediction panels. Defaults to <data_parent>/iou_dice_predictions"
    )
    args = parser.parse_args()

    # Validate argument combinations
    if not args.use_base_model:
        if args.config is None or args.weights is None:
            parser.error("--config and --weights are required when not using --use-base-model")

    validate(
        config_path=args.config,
        weights_path=args.weights,
        val_data_dir=args.val_data_dir,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        merge_cracks=args.merge,
        merge_iou=args.merge_iou,
        use_base_model=args.use_base_model,
        mask_threshold=args.mask_threshold,
        save_predictions=args.save_predictions,
        prediction_output_dir=args.prediction_output_dir,
    )
