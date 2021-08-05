import sys
sys.path.append('./clip-grams')

import json
import os
import torch
import numpy as np
import pytorch_lightning as pl

from CLIP import clip
from argparse import ArgumentParser

import dataset
import utils
import clipgrams


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--textfile', type=str)
    parser.add_argument('--embfile', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--index_dir', type=str, default=None)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--num_return_sequences', type=int, default=3)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # Load cache
    if args.index_dir:
        fname = os.path.join(args.index_dir, 'args.txt')
        with open(fname, 'r') as f:
            index_args = json.load(f)
            for key in list(index_args.keys()):
                if key not in args.__dict__.keys():
                    args.__dict__[key] = index_args[key]
        cache = clipgrams.TextDataset(folder=args.text_dir, args=args).data
        cache_emb = clipgrams.load_index(args)
    else:
        cache = []
        with open(args.textfile) as f:
            for line in f:
                cache.append(line.strip())
        cache_emb = np.load(args.embfile)
        cache_emb = torch.tensor(cache_emb).to(args.device)

    # Load ckpt
    net = utils.load_ckpt(args)
    net.cache = cache
    net.cache_emb = cache_emb

    # Load image preprocessor
    preprocess = clip.load(args.clip_model, jit=False)[1]

    # Load development set
    datamodule = dataset.DataModule(
        train_datadir=None,
        dev_datadir=args.datadir,
        batch_size=args.batch_size,
        preprocess=preprocess,
        all_captions=True)
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
    for idx in range(len(refs[0])):
        ref_file = os.path.join(args.savedir, f'ref{idx}.txt')
        with open(ref_file, 'w') as f:
            for group in refs:
                f.write(group[idx].strip() + '\n')

    
if __name__ == '__main__':
    main()