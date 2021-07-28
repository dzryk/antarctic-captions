import os
import torch
import numpy as np
import pytorch_lightning as pl

from CLIP import clip
from argparse import ArgumentParser

import dataset
import utils


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--textfile', type=str)
    parser.add_argument('--embfile', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--num_return_sequences', type=int, default=3)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--ncaptions', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # Load cache
    cache = []
    with open(args.textfile) as f:
        for line in f:
            cache.append(line.strip())
    cache_emb = np.load(args.embfile)

    # Load ckpt
    net = utils.load_ckpt(args)
    net.cache = cache
    net.cache_emb = torch.tensor(cache_emb).to(args.device)

    # Load image preprocessor
    preprocess = clip.load(args.clip_model, jit=False)[1]

    # Load development set
    datamodule = dataset.DataModule(
        batch_size=args.batch_size,
        preprocess=preprocess,
        datadir=args.datadir)
    datamodule.setup()
    dev = datamodule.val_dataloader()

    # Generate captions
    preds, refs = utils.generate(args, net, dev)
    print(f'Number of captions generated: {len(preds)}')

    # Save predictions and references
    predfile = os.path.join(args.savedir, 'preds.txt')
    with open(predfile, 'w') as f:
        for line in preds:
            f.write(line.strip() + '\n')
    for idx in range(args.ncaptions):
        ref_file = os.path.join(args.savedir, f'ref{idx}.txt')
        with open(ref_file, 'w') as f:
            for group in refs:
                f.write(group[idx].strip() + '\n')

    
if __name__ == '__main__':
    main()