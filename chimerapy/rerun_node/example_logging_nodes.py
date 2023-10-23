import json
from typing import Dict

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr
from transformers import DetrFeatureExtractor, DetrForSegmentation

from chimerapy.engine import DataChunk, Node
from chimerapy.orchestrator import step_node


@step_node(name="ChimeraPyRerunNode_ExampleDetector")
class Detector(Node):
    """Mock detector node for ChimeraPyRerunNode example."""

    def __init__(self, coco_categories_loc, name="Detector"):
        super().__init__(name=name)
        self.coco_categories_loc = coco_categories_loc
        self.coco_categories = None
        self.feature_extractor = None
        self.model = None

    def setup(self):
        rr.init("rerun_example_detection", spawn=True)
        with open(self.coco_categories_loc) as f:
            self.coco_categories = json.load(f)

        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50-panoptic",
        )
        self.model = DetrForSegmentation.from_pretrained(
            "facebook/detr-resnet-50-panoptic",
        )
        self.frames_index = 0
        self.is_thing_from_id: dict[int, bool] = {
            cat["id"]: bool(cat["isthing"]) for cat in self.coco_categories
        }
        class_descriptions = [
            rr.AnnotationInfo(
                id=cat["id"], color=cat["color"], label=cat["name"]
            )
            for cat in self.coco_categories
        ]
        rr.log("/", rr.AnnotationContext(class_descriptions), timeless=True)

    def step(self, data_chunks: Dict[str, DataChunk]) -> None:
        self.frames_index += 1
        frame_idx = self.frames_index
        for name, data_chunk in data_chunks.items():
            rr.set_time_sequence("frame", frame_idx)
            image = data_chunk.get("image")["value"]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rr.log(
                f"{name}_image/rgb",
                rr.Image(image_rgb).compress(jpeg_quality=85),
            )

            inputs = self.feature_extractor(images=image, return_tensors="pt")
            _, _, scaled_height, scaled_width = inputs["pixel_values"].shape
            scaled_size = (scaled_width, scaled_height)
            rgb_scaled = cv2.resize(image_rgb, scaled_size)

            rr.log(
                f"{name}_image_scaled/rgb",
                rr.Image(rgb_scaled).compress(jpeg_quality=85),
            )
            outputs = self.model(**inputs)

            self.logger.debug(
                "Extracting detections and segmentations from network output"
            )
            processed_sizes = [(scaled_height, scaled_width)]
            segmentation_mask = (
                self.feature_extractor.post_process_semantic_segmentation(
                    outputs, processed_sizes
                )[0]
            )
            detections = self.feature_extractor.post_process_object_detection(
                outputs, threshold=0.8, target_sizes=processed_sizes
            )[0]

            mask = segmentation_mask.detach().cpu().numpy().astype(np.uint8)
            rr.log(
                f"{name}_image_scaled/segmentation", rr.SegmentationImage(mask)
            )

            boxes = detections["boxes"].detach().cpu().numpy()
            class_ids = detections["labels"].detach().cpu().numpy()
            things = [self.is_thing_from_id[id] for id in class_ids]

            self.log_detections(name, boxes, class_ids, things)

    def log_detections(
        self,
        name,
        boxes: npt.NDArray[np.float32],
        class_ids: list[int],
        things: list[bool],
    ) -> None:
        things_np = np.array(things)
        class_ids_np = np.array(class_ids, dtype=np.uint16)

        thing_boxes = boxes[things_np, :]
        thing_class_ids = class_ids_np[things_np]
        rr.log(
            f"{name}_image_scaled/detections/things",
            rr.Boxes2D(
                array=thing_boxes,
                array_format=rr.Box2DFormat.XYXY,
                class_ids=thing_class_ids,
            ),
        )

        background_boxes = boxes[~things_np, :]
        background_class_ids = class_ids[~things_np]
        rr.log(
            f"{name}/image_scaled/detections/background",
            rr.Boxes2D(
                array=background_boxes,
                array_format=rr.Box2DFormat.XYXY,
                class_ids=background_class_ids,
            ),
        )

    def teardown(self):
        rr.disconnect()
