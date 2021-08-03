import os
import torch
import numpy as np
import pytorch_lightning as pl

from CLIP import clip
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

import dataset
import model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--train_datadir', type=str)
    parser.add_argument('--dev_datadir', type=str)
    parser.add_argument('--textfile', type=str)
    parser.add_argument('--embfile', type=str)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--maxlen_enc', type=int, default=128)
    parser.add_argument('--maxlen_dec', type=int, default=32)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--val_after_n_epochs', type=int, default=1)
    parser.add_argument('--tmax', type=int, default=1e5)
    parser.add_argument('--save_top_k', type=int, default=10)
    parser.add_argument('--lrate', type=float, default=3e-4)
    args = parser.parse_args()

    # Load cache
    cache = []
    with open(args.textfile) as f:
        for line in f:
            cache.append(line.strip())
    cache_emb = np.load(args.embfile)

    # Load image preprocessor
    preprocess = clip.load(args.clip_model, jit=False)[1]

    # Train model
    ckpt_callback = ModelCheckpoint(
        monitor='vloss',
        mode='min',
        filename='-{epoch:02d}-{vloss:.3f}',
        save_top_k=args.save_top_k)
    datamodule = dataset.DataModule(
        train_datadir=args.train_datadir,
        dev_datadir=args.dev_datadir,
        batch_size=args.batch_size,
        nworkers=args.nworkers,
        preprocess=preprocess)
    net = model.Model(args, cache=cache, cache_emb=cache_emb)
    trainer = pl.Trainer(
        default_root_dir=args.datadir,
        gpus=args.gpus,
        tpu_cores=args.tpu_cores,
        max_steps=args.tmax,
        callbacks=[ckpt_callback],
        check_val_every_n_epoch=args.val_after_n_epochs)
    trainer.fit(net, datamodule)


if __name__ == '__main__':
    main()

