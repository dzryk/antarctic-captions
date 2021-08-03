import json

from collections import defaultdict
from argparse import ArgumentParser


def save_text(datadir, split):
    """
    Loads the captions json, extracts the image ids and captions and
    stores them so they can be used by TextImageDataset format.
    """
    path = f'{datadir}/annotations/captions_{split}.json'
    with open(path, 'r') as f:
        captions = json.load(f)
    x = captions['annotations']
    ids = defaultdict(list)
    for item in x:
        ids[item['image_id']].append(item['caption'])
    keys = list(ids.keys())
    for k in keys:
        fname = f'{str(k).zfill(12)}.txt'
        caps = ids[k][:5]
        path = f'{datadir}/{split}/{fname}'
        with open(path, 'w') as f:
            for line in caps:
                f.write(line + '\n')


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str)
    args = parser.parse_args()

    save_text(args.datadir, 'train2017')
    save_text(args.datadir, 'val2017')


if __name__ == '__main__':
    main()

