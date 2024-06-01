import argparse
import glob
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--output", type=str)
    return parser.parse_args()


def main(args):
    data_root = args.root

    data_list = []
    img_files = glob.glob(os.path.join(data_root, "img/*.jpg"))
    for img_file in img_files:
        data_list.append(
            [
                img_file,
                img_file.replace("img", "label_img").replace("jpg", "png"),
                f"A natural bird-eye view image of a {'road' if 'RO' in img_file else 'river'}",
            ]
        )

    pd.DataFrame(data_list, columns=["image", "conditional", "prompt"]).to_csv(
        args.output, index=False
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
