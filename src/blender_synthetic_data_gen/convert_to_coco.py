from collections import namedtuple
from typing import *
import json
from pathlib import Path
import logging
import traceback
import numpy as np
import typer


KeypointSpec = namedtuple("KeypointSpec", ["coco_name", "blender_name", "enabled"])


KEYPOINT_SPECS = [
    KeypointSpec("boundary_top_left", "LeftCornerTop", True),
    KeypointSpec("boundary_top_right", "RightCornerTop", True),
    KeypointSpec("boundary_bottom_left", "LeftCornerBottom", True),
    KeypointSpec("boundary_bottom_right", "RightCornerBottom", True),
    KeypointSpec("left_penalty_top_left", "LeftPenaltyTopLeft", True),
    KeypointSpec("left_penalty_top_right", "LeftPenaltyTopRight", True),
    KeypointSpec("left_penalty_bottom_left", "LeftPenaltyBottomLeft", True),
    KeypointSpec("left_penalty_bottom_right", "LeftPenaltyBottomRight", True),
    KeypointSpec("right_penalty_top_left", "RightPenaltyTopLeft", True),
    KeypointSpec("right_penalty_top_right", "RightPenaltyTopRight", True),
    KeypointSpec("right_penalty_bottom_left", "RightPenaltyBottomLeft", True),
    KeypointSpec("right_penalty_bottom_right", "RightPenaltyBottomRight", True),
    KeypointSpec("left_goal_top_left", "LeftGoalTopLeft", True),
    KeypointSpec("left_goal_top_right", "LeftGoalTopRight", True),
    KeypointSpec("left_goal_bottom_left", "LeftGoalBottomLeft", True),
    KeypointSpec("left_goal_bottom_right", "LeftGoalBottomRight", True),
    KeypointSpec("right_goal_top_left", "RightGoalTopLeft", True),
    KeypointSpec("right_goal_top_right", "RightGoalTopRight", True),
    KeypointSpec("right_goal_bottom_left", "RightGoalBottomLeft", True),
    KeypointSpec("right_goal_bottom_right", "RightGoalBottomRight", True),
    KeypointSpec("centre_top", "CenterTop", True),
    KeypointSpec("centre_bottom", "CenterBottom", True),
    KeypointSpec("circle_top", "CircleTop", True),
    KeypointSpec("circle_bottom", "CircleBottom", True),
    KeypointSpec("left_penalty_spot", "LeftPenaltyCircle", False),
    KeypointSpec("right_penalty_spot", "RightPenaltyCircle", False),
    KeypointSpec("centre_spot", "CenterSpot", False),
    KeypointSpec("circle_left", "CircleLeft", True),
    KeypointSpec("circle_right", "CircleRight", True),
    KeypointSpec("left_penalty_d_top", "LeftPenaltyArcTop", True),
    KeypointSpec("left_penalty_d_bottom", "LeftPenaltyArcBottom", True),
    KeypointSpec("right_penalty_d_top", "RightPenaltyArcTop", True),
    KeypointSpec("right_penalty_d_bottom", "RightPenaltyArcBottom", True),
    KeypointSpec("left_goal_top", "LeftGoalTop", True),
    KeypointSpec("left_goal_bottom", "LeftGoalBottom", True),
    KeypointSpec("right_goal_top", "RightGoalTop", True),
    KeypointSpec("right_goal_bottom", "RightGoalBottom", True),
    KeypointSpec("centre_top_left", "CircleTopLeft", True),
    KeypointSpec("centre_top_right", "CircleTopRight", True),
    KeypointSpec("centre_bottom_left", "CircleBottomLeft", True),
    KeypointSpec("centre_bottom_right", "CircleBottomRight", True),
    KeypointSpec("left_penalty_d", "LeftPenaltyArc", True),
    KeypointSpec("right_penalty_d", "RightPenaltyArc", True),
]


def build_keypoint_names():
    names = []
    for s in KEYPOINT_SPECS:
        if not s.enabled:
            continue
        if s.coco_name in names:
            continue
        names.append(s.coco_name)
    return names


KEYPOINT_NAMES = build_keypoint_names()
BLENDER_TO_COCO_NAMES = {
    s.blender_name: s.coco_name
    for s in KEYPOINT_SPECS
    if s.enabled
}


class DarwinFile:
    def __init__(
            self,
            img_id: int,
            darwin_path: Path,
            image_root: Path,
            keypoint_names: List[str],
            img_filename_filter: str = ""
    ):
        self.img_id = img_id
        self.darwin_path = darwin_path
        self.image_root = image_root
        self.num_keypoints = len(keypoint_names)
        self.img_filename_filter = img_filename_filter
        self.mapped_names_to_idx = {v: i for i, v in enumerate(keypoint_names)}


def get_duplicates(items):
    items_set = set()
    duplicates = []
    for i in items:
        if i in items_set:
            duplicates.append(i)
        else:
            items_set.add(i)
    return duplicates


def darwin_to_coco(item: DarwinFile) -> Tuple[bool, Optional[Dict], Optional[List[Dict]], Set[str]]:
    # return image section & annotations sections
    darwin_ann = json.loads(Path(item.darwin_path).read_text())
    if (item.img_filename_filter is not None) and (item.img_filename_filter not in darwin_ann["image"]["filename"]):
        return False, None, None
    image = {
        "id": item.img_id,
        "file_name": str(item.image_root / darwin_ann["image"]["filename"]),
        # "width": darwin_ann["image"]["width"],
        # "height": darwin_ann["image"]["height"],
        "width": darwin_ann["image"]["height"],  # TODO: this is actually wrong, but it's flipped to accommodate h/w bug form blender
        "height": darwin_ann["image"]["width"],  # TODO: this is actually wrong, but it's flipped to accommodate h/w bug form blender
    }
    # 0th item always corresponds to the football field
    coco_anns = [{
        "id": 0,  # will be cleaned up later
        "image_id": item.img_id,
        "segmentation": [0, 0, 0, 0],
        "iscrowd": 0,
        "category_id": 0,
        "area": image["width"] * image["height"],
        "bbox": [0, 0, image["width"], image["height"]],
        "num_keypoints": item.num_keypoints,
        "keypoints": [],
    }]
    keypoints = [[0, 0, 0] for _ in range(item.num_keypoints)]  # 0=not labelled, 1=occluded, 2=visible]
    seen_names = []
    ignored_names = set()
    for a in darwin_ann["annotations"]:
        ann_is_bbox = "bounding_box" in a
        item_name = a["name"]
        box_is_corner = item_name in BLENDER_TO_COCO_NAMES
        if not (ann_is_bbox and box_is_corner):
            ignored_names.add(item_name)
            continue

        corner_name = BLENDER_TO_COCO_NAMES[item_name]
        seen_names.append(item_name)
        corner_idx = item.mapped_names_to_idx[corner_name]

        kpt_x = np.clip(0.5 * a["bounding_box"]["w"] + a["bounding_box"]["x"], 0.0, image["width"])
        kpt_y = np.clip(0.5 * a["bounding_box"]["h"] + a["bounding_box"]["y"], 0.0, image["height"])
        keypoints[corner_idx] = [kpt_x, kpt_y, 2]
    coco_anns[0]["keypoints"] = [i for ii in keypoints for i in ii]

    duplicates = get_duplicates(seen_names)
    assert len(duplicates) == 0, f"duplicated names seen in image: {duplicates}"
    return True, image, coco_anns, ignored_names


def aggregate_img_and_ann(imgs_and_anns: List[Tuple[Dict, List[Dict]]]) -> Tuple[List[Dict], List[Dict]]:
    imgs, anns = zip(*imgs_and_anns)
    anns = [a for aa in anns for a in aa]
    for i, a in enumerate(anns):
        a["id"] = i
    return imgs, anns


def main(
        input_dir: Path,
        image_root: Path,
        output_dir: Path = ".",
        img_filter: str = "",
        skeleton_set_name: str = "football-field",
):
    coco_json = {
        'info': {
            'description': 'converted darwin to keypoints',
            'version': '1.0'
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {
                'name': skeleton_set_name,
                'supercategory': 'none',
                'id': 0,
                'keypoints': KEYPOINT_NAMES,
                'skeleton': [],  # TODO: it's not really required but we can add this later for completeness
            }
        ],
    }
    annotation_files = list(input_dir.glob("*.json"))
    darwin_files = [DarwinFile(
        img_id=i,
        darwin_path=p,
        image_root=image_root,
        img_filename_filter=img_filter,
        keypoint_names=[k.coco_name for k in KEYPOINT_SPECS if k.enabled]
    ) for i, p in enumerate(annotation_files)]
    logging.info(f"found {len(annotation_files)} annotations under {str(input_dir)}")
    imgs_and_anns = []
    n_added_imgs = 0
    has_error = False
    ignored_names = set()
    for df in darwin_files:
        try:
            should_add, img, ann, img_ignored_names = darwin_to_coco(df)
            if should_add:
                imgs_and_anns.append((img, ann))
                n_added_imgs += 1
                ignored_names = ignored_names.union(img_ignored_names)
        except Exception as e:
            print(f"issues processing file: {df.darwin_path}: {repr(e)}")
            print(traceback.format_exc())
            has_error = True
    if has_error:
        raise RuntimeError("error occurred during file preprocessing")
    imgs, anns = aggregate_img_and_ann(imgs_and_anns)
    coco_json["images"] = imgs
    coco_json["annotations"] = anns

    output_dir = Path(output_dir) / "coco.json"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.write_text(json.dumps(coco_json, indent=4))
    print(f"the following names were ignored: {sorted(list(ignored_names))}")
    print(f"added {n_added_imgs} / {len(darwin_files)} wrt the given filter: {img_filter}")
    print(output_dir)


if __name__ == "__main__":
    typer.run(main)
