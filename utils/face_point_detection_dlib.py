
import os
import dlib
import numpy as np
import argparse

from skimage import io


def face_point_detection(face_path: str, detector_path: str) -> dict:

    # Read path to face folder and detector file

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(detector_path)
    i = 0
    not_reck_faces = []
    faces = {}
    print('Reading directory {}'.format(face_path))

    for idx, fname in enumerate(os.listdir(face_path)):
        cur_path = os.path.join(face_path, fname)

        if idx % 100 == 0:

            print(f'Predicted {idx} files, currently predicting {fname}')

        if cur_path.endswith('.jpg'):
            img = io.imread(cur_path)
            dets, _, _ = detector.run(img, 1, -1)

            try:
                shape = predictor(img, dets[0])
                faces[fname] = np.array([[p.x, p.y] for p in shape.parts()])
            except IndexError:
                i += 1
                not_reck_faces.append(fname)

    print(f"Cound't recognize face for {i} images. List of images saved in missing_faces.txt")\

    with open("missing_faces.txt", "w") as mf:
        mf.write(str(not_reck_faces))

    return faces


def main():
    parser = argparse.ArgumentParser(description='dlib face 68p detector',
                                     add_help=True)
    parser.add_argument('--predictions_path',
                        action='store', type=str, help='folder')
    parser.add_argument('--output_path', action='store',
                        type=str, help='folder')
    parser.add_argument('--predictor_path', action='store', type=str,
                        help='path to .dat file', default='data/shape_predictor_68_face_landmarks.dat')

    args = parser.parse_args()

    detected_faces = face_point_detection(
        args.predictions_path, args.predictor_path)

    # saving prections to .pts file

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    for fname, value in detected_faces.items():
        np.savetxt(args.output_path +
                   fname[:-4] + '.pts', value, fmt='%.3f')
    print(f'.pts files were saved in {args.output_path}')


if __name__ == '__main__':
    main()
