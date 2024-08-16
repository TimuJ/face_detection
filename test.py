import os
import cv2
import json
import torch

import torchvision.transforms as transforms

import numpy as np
from src.model import FaceKeypointResNet34


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    with open('test_config.json', 'r') as f:
        config = json.load(f)

    path_to_weights = str(config['path_to_weights'])
    test_dir = str(config['test_dir'])
    output_dir = str(config['output_dir'])

    model = FaceKeypointResNet34(requires_grad=True).to(device)
    model.load_state_dict(torch.load(path_to_weights))

    print(f'Model from {path_to_weights} loaded')

    transform = transforms.Compose([transforms.ToTensor()])

    model = model.to(device).eval()
    for idx, fname in enumerate(os.listdir(test_dir)):
        if idx % 100 == 0:
            print(f"Predicting {idx} image from {fname}")
        cur_path = os.path.join(test_dir, fname)
        if cur_path.endswith('.jpg') or cur_path.endswith('.png'):
            img = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
            orig_h, orig_w, c = img.shape
            img = cv2.resize(img, (224, 224))
            img = img / 255.
            img = transform(img)
            img = img.to(torch.float)
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img)
            outputs = outputs.cpu().detach().numpy()
            outputs = outputs.reshape(-1, 2)
            keypoints = outputs * [orig_w / 224, orig_h / 224]
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            np.savetxt(output_dir + fname[:-4] + '.pts',
                       keypoints, fmt='%.3f', delimiter=' ')


if __name__ == '__main__':
    main()
