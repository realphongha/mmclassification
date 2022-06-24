# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from black import out

import mmcv
import os
import shutil
from tqdm import tqdm

from mmcls.apis import inference_model, init_model


def main():
    parser = ArgumentParser()
    parser.add_argument('dir', help='Directory of images')
    parser.add_argument('cfg', help='Config file')
    parser.add_argument('ckpt', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--output', help='Output directory')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.cfg, args.ckpt, device=args.device)
    # run infer
    for fn in tqdm(os.listdir(args.dir)):
        img = os.path.join(args.dir, fn)
        result = inference_model(model, img)
        cls = result["pred_class"]
        output_dir = os.path.join(args.output, cls)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(img, output_dir)


if __name__ == '__main__':
    main()
