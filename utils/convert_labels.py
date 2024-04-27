import json
import argparse
from helper import fix_bounding_box


def convert_json_to_txt(json_file, txt_file):
    """Converts JSON annotations to a TXT file in YOLO format.

    Args:
        json_file (str): Path to the JSON annotation file.
        txt_file (str): Path to the output TXT file.
    """

    with open(json_file) as f:
        data = json.load(f)

    with open(txt_file, "w") as f:
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]
            xmin, ymin = points[0]
            xmax, ymax = points[1]
            # fix condition where xmin > xmax, and ymin > ymax
            xmin, ymin, xmax, ymax = fix_bounding_box([xmin, ymin, xmax, ymax])
            f.write(f"{xmin},{ymin},{xmax},{ymax},{label}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str, help="Path to the JSON annotation file.")
    parser.add_argument("--txt-file", type=str, help="Path to the output TXT file.")
    args = parser.parse_args()
    convert_json_to_txt(args.json_file, args.txt_file)
    print("Done.")