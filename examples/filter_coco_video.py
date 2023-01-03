from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)
from motpy import Detection, ModelPreset, MultiObjectTracker, NpImage
from pycocotools.coco import COCO
import json
import numpy as np
from os.path import dirname, basename
import pathlib

def process_anns(anns, th=0.5) -> Sequence[Detection]:

    # convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]
    out_detections = []
    for ann in anns:
        confidence = ann['score']
        if confidence > th:
            xmin = int(ann['bbox'][0])
            ymin = int(ann['bbox'][1])
            xmax = int(ann['bbox'][0] + ann['bbox'][2])
            ymax = int(ann['bbox'][1] + ann['bbox'][3])
            out_detections.append(
                Detection(box=[xmin, ymin, xmax, ymax], \
                          class_id=ann['category_id'], \
                          score=confidence)
            )

    return out_detections



def run(fpath_coco: str, \
        cap_fps=30.0, \
        tracker_min_iou: float = 0.25):

    coco = COCO(fpath_coco)

    with open(fpath_coco, 'rb') as f:
        data = json.load(f)


    tracker = MultiObjectTracker(
        dt=1 / cap_fps,
        tracker_kwargs={'max_staleness': 5},
        model_spec={'order_pos': 1, 'dim_pos': 2,
                    'order_size': 0, 'dim_size': 2,
                    'q_var_pos': 5000., 'r_var_pos': 0.1},
        matching_fn_kwargs={'min_iou': tracker_min_iou,
                            'multi_match_min_iou': 0.93})

    id_anns = 0
    coco_annotations = list()
    for coco_image in data['images']:
    # while True:

        ids_anns = coco.getAnnIds(imgIds=[coco_image['id']])
        anns = coco.loadAnns(ids=ids_anns)
        detections = process_anns(anns)

        # track detected objects
        _ = tracker.step(detections=detections)
        active_tracks = tracker.active_tracks(min_steps_alive=3)


        for track in active_tracks:
            
            bbox = track.box
            bbox[2:4] += - bbox[0:2]
            coco_annotations.append(
                dict(
                id = id_anns, \
                category_id = track.class_id,\
                image_id = coco_image["id"],\
                area = int(bbox[2]*bbox[3]),\
                bbox = np.round(bbox, decimals=1).tolist(),\
                iscrowd = 0,\
                score = round(track.score, 2), \
                id_track = track.id
                )
            )
            # print(track)
            # print(coco_annotations[-1])
            id_anns += 1

    data['annotations'] = coco_annotations
    
    fpath_export = f"{dirname(fpath_coco)}/{pathlib.Path(fpath_coco).stem}-filtered.json"
    with open(fpath_export, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':

    # https://github.com/wmuron/motpy/blob/c77f85d27e371c0a298e9a88ca99292d9b9cbe6b/motpy/tracker.py

    fpath_coco = './assets/response_1671265846410.json'
    run(fpath_coco)
