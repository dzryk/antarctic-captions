import numpy as np

from argparse import ArgumentParser
from CLIP import clip

import dataset
import utils


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--textfile', type=str)
    parser.add_argument('--embfile', type=str)
    parser.add_argument('--topk', type=int, default=25000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # Compute cache entries
    grams = dataset.get_cache(args)
    print(f'Number of entries: {len(grams)}')

    # Encode using CLIP
    perceiver = clip.load(args.clip_model, jit=False)[0]
    perceiver = perceiver.eval().requires_grad_(False).to(args.device)
    emb = utils.encode_cache(args, grams, perceiver).cpu().numpy()
    
    # Save results
    np.save(args.embfile, emb)
    with open(args.textfile, 'w') as f:
        for line in grams:
            f.write(line + '\n')


if __name__ == '__main__':
    main()

