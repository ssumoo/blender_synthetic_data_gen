from pathlib import Path
from tqdm import tqdm
import typer
import json
import cv2


colors = {
    "human": (0, 255, 0),
    "ball": (255, 255, 0),
}


def main(input_path: Path, output_path: Path = "./rendered"):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    found_images = list(input_path.glob("*.png"))
    print(f"found {len(found_images)} images")
    for p in tqdm(found_images):
        ann_p = p.parent / f"{p.stem}.json"
        ann = json.loads(ann_p.read_text())
        img = cv2.imread(str(p))

        for a in ann["annotations"]:
            if "bounding_box" in a:
                color = colors.get(a["name"], (255, 0, 255))
                bbox = a["bounding_box"]
                img = cv2.rectangle(
                    img,
                    (int(bbox["x"]), int(bbox["y"])),
                    (int(bbox["x"]+bbox["w"]), int(bbox["y"]+bbox["h"])),
                    color,
                    1
                )
                if a["name"] in colors:
                    img = cv2.putText(
                        img,
                        a["name"],
                        (int(bbox["x"]), int(bbox["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )
        out_path = output_path / p.name
        cv2.imwrite(str(out_path), img)


if __name__ == "__main__":
    typer.run(main)
