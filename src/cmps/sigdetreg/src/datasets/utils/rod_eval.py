import torch
import copy
import numpy as np
from .rod_api import RODEval


class RODEvaluator(object):
    def __init__(self, iou_types):
        assert isinstance(iou_types, (list, tuple))
        self.iou_types = iou_types
        self.eval = {}
        for iou_type in iou_types:
            self.eval[iou_type] = RODEval(iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions, targets):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, targets, iou_type)

            eval_instance = self.eval[iou_type]
            eval_instance.dt = results['dt']
            eval_instance.gt = results['gt']
            eval_instance.params.imgIds = list(img_ids)
            eval_instance.params.catIds = list(range(0, 11))
            eval_instance.evaluate()

            self.eval_imgs[iou_type].append(eval_instance.evalImgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            self.create_common_eval(self.eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for eval_instance in self.eval.values():
            eval_instance.accumulate()

    def summarize(self):
        for iou_type, eval_instance in self.eval.items():
            print("IoU metric: {}".format(iou_type))
            eval_instance.summarize()

    def prepare(self, predictions, targets, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_custom_detection(predictions, targets)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_custom_detection(self, predictions, targets):
        dt_results = []
        gt_results = []

        for img_id in predictions.keys():
            prediction = predictions[img_id]
            target = targets[img_id]

            if len(prediction) == 0:
                continue

            pred_boxes = prediction["boxes"].tolist()
            pred_scores = prediction["scores"].tolist()
            pred_labels = prediction["labels"].tolist()

            gt_boxes = target["boxes"].tolist()
            gt_labels = target["labels"].tolist()

            dt_results.extend(
                [
                    {
                        "image_id": img_id,
                        "category_id": pred_labels[k],
                        "bbox": box,
                        "score": pred_scores[k],
                        "area": box[1] - box[0],
                    }
                    for k, box in enumerate(pred_boxes)
                ]
            )

            gt_results.extend(
                [
                    {
                        "image_id": img_id,
                        "category_id": gt_labels[k],
                        "bbox": box,
                        "area": box[1] - box[0],
                        'ignore': False,
                        'iscrowd': 0,
                    }
                    for k, box in enumerate(gt_boxes)
                ]
            )

        return {"dt": dt_results, "gt": gt_results}

    def create_common_eval(self, eval_instance, img_ids, eval_imgs):
        eval_instance.imgIds = img_ids
        eval_instance.evalImgs = eval_imgs
