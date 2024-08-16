import glob
import shutil
import argparse


def train_val_keypoints_split(path_to_dir: str, split_ratio: float):
    if path_to_dir[-1] != '/':
        path_to_dir += '/'
    pts_train_files = glob.glob(path_to_dir + "train/*.pts")
    for i in range(int(len(pts_train_files) * split_ratio)):
        shutil.move(pts_train_files[i], path_to_dir + "val")
    try:
        shutil.move(pts_train_files[i][:-4] + '.jpg', path_to_dir + "val")
    except FileNotFoundError:
        shutil.move(pts_train_files[i][:-4] + '.png', path_to_dir + "val")


def main():
    parser = argparse.ArgumentParser(description='split train to tain and val sets',
                                     add_help=True)
    parser.add_argument('--train_path',
                        action='store', type=str, help='path to folder with train folder')
    parser.add_argument('--split_ratio', action='store',
                        type=float, help='validation/total files ratio')

    args = parser.parse_args()

    train_val_keypoints_split(args.train_path, args.split_ratio)


if __name__ == '__main__':
    main()
