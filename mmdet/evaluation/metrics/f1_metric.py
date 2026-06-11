# Copyright (c) OpenMMLab. All rights reserved.
# Custom metrics for F1, Precision, Recall evaluation
# Uses COCO API to compute metrics at multiple IoU thresholds

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import get_local_path, dump, load
from mmengine.logging import MMLogger
import tempfile
import os.path as osp
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.registry import METRICS


@METRICS.register_module()
class F1PrecisionRecallMetric(BaseMetric):
    """Custom metric for computing F1 Score, Precision, Recall at multiple IoU thresholds.

    This metric uses COCO API to compute metrics, ensuring alignment
    with COCO Metric results.

    Args:
        ann_file (str): Path to the coco format annotation file.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
    """

    default_prefix: Optional[str] = 'f1_pr'

    def __init__(
        self,
        ann_file: str,
        backend_args: dict = None,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.backend_args = backend_args

        with get_local_path(ann_file, backend_args=backend_args) as local_path:
            self._coco_api = COCO(local_path)

        self.results = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions."""
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['pred_bboxes'] = pred['bboxes'].cpu().numpy()
            result['pred_scores'] = pred['scores'].cpu().numpy()
            result['pred_labels'] = pred['labels'].cpu().numpy()
            self.results.append(result)

    def results2json(self, outfile_prefix: str) -> str:
        """Convert predictions to COCO format json file."""
        bbox_json_results = []
        for result in self.results:
            image_id = result['img_id']
            labels = result['pred_labels']
            bboxes = result['pred_bboxes']
            scores = result['pred_scores']

            for i, label in enumerate(labels):
                x1, y1, x2, y2 = bboxes[i]
                data = {
                    'image_id': image_id,
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(scores[i]),
                    'category_id': int(label)
                }
                bbox_json_results.append(data)

        result_file = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_file)
        return result_file

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute F1, Precision, Recall metrics at multiple IoU thresholds."""
        logger: MMLogger = MMLogger.get_current_instance()

        num_classes = len(self.dataset_meta['classes'])
        classes = self.dataset_meta['classes']
        cat_ids = self._coco_api.get_cat_ids()

        # Create temp file
        tmp_dir = tempfile.TemporaryDirectory()
        outfile_prefix = osp.join(tmp_dir.name, 'results')

        # Dump predictions
        self.results2json(outfile_prefix)
        predictions = load(f'{outfile_prefix}.bbox.json')
        coco_dt = self._coco_api.loadRes(predictions)

        # COCOeval with multiple IoU thresholds (like COCO standard)
        coco_eval = COCOeval(self._coco_api, coco_dt, 'bbox')
        coco_eval.params.catIds = cat_ids
        coco_eval.params.imgIds = self._coco_api.get_img_ids()
        # Use standard COCO IoU thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        coco_eval.params.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        coco_eval.params.maxDets = np.array([100, 300])

        coco_eval.evaluate()
        coco_eval.accumulate()

        # Get precision and recall matrices
        # precisions shape: (nIoU, nRecall, nClass, nArea, nMaxDet)
        # recalls shape: (nIoU, nClass, nArea, nMaxDet)
        precisions = coco_eval.eval['precision']
        recalls = coco_eval.eval['recall']

        logger.info(f'[F1Metric] Processing {num_classes} classes')

        # IoU indices
        iou_50_idx = 0
        iou_75_idx = 5  # IoU=0.75
        iou_avg_idx = slice(None)  # All IoU thresholds

        # Compute per-class metrics
        eval_results = OrderedDict()

        # Compute metrics for each IoU threshold
        for iou_name, iou_idx in [('IoU=0.50', iou_50_idx), ('IoU=0.75', iou_75_idx)]:
            ap50_list = []
            recall_list = []
            recall_300_list = []

            for idx, cat_id in enumerate(cat_ids):
                cls_name = classes[idx] if idx < len(classes) else f'class_{idx}'

                # Get precision for this IoU, area=all, maxDets=100
                prec = precisions[iou_idx, :, idx, 0, 0]
                valid_prec = prec[prec > -1]

                if len(valid_prec) > 0:
                    ap = float(np.mean(valid_prec))
                else:
                    ap = 0.0

                ap50_list.append(ap)

                # Get recall
                rec_100 = recalls[iou_idx, idx, 0, 0] if recalls is not None else 0.0
                rec_300 = recalls[iou_idx, idx, 0, 1] if recalls is not None else 0.0
                if rec_100 > -1:
                    recall_list.append(float(rec_100))
                if rec_300 > -1:
                    recall_300_list.append(float(rec_300))

            # Overall metrics for this IoU
            mAP = float(np.mean(ap50_list))
            AR_100 = float(np.mean(recall_list)) if len(recall_list) > 0 else 0.0
            AR_300 = float(np.mean(recall_300_list)) if len(recall_300_list) > 0 else 0.0

            eval_results[f'mAP@{iou_name.split("=")[1]}'] = round(mAP, 4)
            eval_results[f'AR@100_{iou_name.split("=")[1]}'] = round(AR_100, 4)
            eval_results[f'AR@300_{iou_name.split("=")[1]}'] = round(AR_300, 4)

        # Compute mAP@0.5:0.95 (average across all IoU thresholds)
        # Shape: (10 IoU thresholds, 101 recalls, nClass, 4 areas, 2 maxDets)
        # Average over IoU dimension (axis 0)
        mAP_all_iou = np.mean(precisions[:, :, :, 0, 0], axis=0)  # (101, nClass)
        mAP_avg = float(np.mean(mAP_all_iou[mAP_all_iou > -1])) if np.any(mAP_all_iou > -1) else 0.0

        # Average recall across all IoU thresholds
        AR_100_all_iou = np.mean(recalls[:, :, 0, 0], axis=0)  # (nClass,)
        AR_300_all_iou = np.mean(recalls[:, :, 0, 1], axis=0)  # (nClass,)
        AR_100_avg = float(np.mean(AR_100_all_iou[AR_100_all_iou > -1])) if np.any(AR_100_all_iou > -1) else 0.0
        AR_300_avg = float(np.mean(AR_300_all_iou[AR_300_all_iou > -1])) if np.any(AR_300_all_iou > -1) else 0.0

        eval_results['mAP@0.5:0.95'] = round(mAP_avg, 4)
        eval_results['AR@100@0.5:0.95'] = round(AR_100_avg, 4)
        eval_results['AR@300@0.5:0.95'] = round(AR_300_avg, 4)

        # F1 calculation at IoU=0.5 (use AR@300 to match max_per_img=300)
        mAP_50 = eval_results.get('mAP@0.50', 0.0)
        AR_300_at_50 = eval_results.get('AR@300_0.50', 0.0)

        precision = mAP_50
        recall = AR_300_at_50
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # ACC = (TP + TN) / (TP + TN + FN + FP) = TP / GT = Recall
        acc = recall

        eval_results['precision'] = round(precision, 4)
        eval_results['recall'] = round(recall, 4)
        eval_results['acc'] = round(acc, 4)
        eval_results['f1'] = round(f1, 4)

        # Print comprehensive table
        table_data = [
            ['IoU', 'mAP', 'AR@100', 'AR@300'],
            ['-' * 25, '-' * 10, '-' * 10, '-' * 10],
            ['0.50', f"{eval_results.get('mAP@0.50', 0):.4f}", f"{eval_results.get('AR@100_0.50', 0):.4f}", f"{eval_results.get('AR@300_0.50', 0):.4f}"],
            ['0.75', f"{eval_results.get('mAP@0.75', 0):.4f}", f"{eval_results.get('AR@100_0.75', 0):.4f}", f"{eval_results.get('AR@300_0.75', 0):.4f}"],
            ['0.50:0.95', f"{eval_results.get('mAP@0.5:0.95', 0):.4f}", f"{eval_results.get('AR@100@0.5:0.95', 0):.4f}", f"{eval_results.get('AR@300@0.5:0.95', 0):.4f}"],
            ['-' * 25, '-' * 10, '-' * 10, '-' * 10],
            ['P (≈mAP@50)', f'{precision:.4f}', '', ''],
            ['R (≈AR@300)', f'{recall:.4f}', '', ''],
            ['ACC', f'{acc:.4f}', '', ''],
            ['F1@50,300', f'{f1:.4f}', '', ''],
        ]

        table = AsciiTable(table_data)
        logger.info(f'\n[Custom Metrics - Multi IoU Thresholds]\n{table.table}')
        logger.info(f'\nSummary: P={precision:.4f}, R={recall:.4f}, ACC={acc:.4f}, F1={f1:.4f}')

        tmp_dir.cleanup()

        return eval_results
