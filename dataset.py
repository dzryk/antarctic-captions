import collections
import nltk
import torch
import pytorch_lightning as pl
import numpy as np
import torchvision.transforms.functional as F

from sklearn.feature_extraction import image
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions

import utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, preprocess, ncaptions):
        self.data = data
        self.preprocess = preprocess
        self.ncaptions = ncaptions

    def __getitem__(self, idx):
        items = []
        img, target = self.data[idx]
        img = self.preprocess(img)
        items.append(img)
        items.append(target[:self.ncaptions])
        return items
    
    def __len__(self):
        return len(self.data)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, ncaptions):
        self.data = data
        self.ncaptions = ncaptions

    def __getitem__(self, idx):
        _, target = self.data[idx]
        return target[:self.ncaptions]
    
    def __len__(self):
        return len(self.data)


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=64,
                 nworkers=0,
                 preprocess=None,
                 ncaptions=5,
                 datadir='.'):
        super().__init__()
        self.batch_size = batch_size
        self.nworkers = nworkers
        self.preprocess = preprocess
        self.datadir = datadir
        self.ncaptions = ncaptions
    
    def setup(self, stage=None):
        datadir=self.datadir
        train_datatype='train2017'
        train_annfile='{}/annotations/captions_{}.json'.format(datadir, train_datatype)
        train_loc='{}/{}'.format(datadir, train_datatype)
        dev_datatype='val2017'
        dev_annfile='{}/annotations/captions_{}.json'.format(datadir, dev_datatype)
        dev_loc='{}/{}'.format(datadir, dev_datatype)

        if stage == 'fit' or stage is None:
            train_data = CocoCaptions(root=train_loc, annFile=train_annfile)
            valid_data = CocoCaptions(root=dev_loc, annFile=dev_annfile)
            self.train = Dataset(train_data, self.preprocess, self.ncaptions)
            self.valid = Dataset(valid_data, self.preprocess, self.ncaptions)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nworkers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.nworkers,
            pin_memory=True)


def get_cache(args):
    """
    Computes each entries from a DataLoader. By default uses COCO training
    captions. Bigrams are selected based on PMI and the topk are selected.
    """
    datatype = args.split
    annfile = '{}/annotations/captions_{}.json'.format(args.datadir, datatype)
    loc='{}/{}'.format(args.datadir, datatype)
    loader = CocoCaptions(root=loc, annFile=annfile)
    loader = TextDataset(loader, args.ncaptions)
    loader = DataLoader(loader, batch_size=args.batch_size, shuffle=True)
    cache = []
    for batch in loader:
        for x in batch:
            for sent in x:
                wordlist = nltk.wordpunct_tokenize(sent)
                wordlist = [w.lower() for w in wordlist]
                cache.extend(wordlist)
    
    # Unigrams
    unigrams = collections.Counter(cache).most_common(args.topk)
    unigrams = [t[0] for t in unigrams]
    
    # Bigrams
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(cache)
    finder.apply_freq_filter(3)
    bigrams = finder.nbest(bigram_measures.pmi, args.topk)
    bigrams = [' '.join(g) for g in bigrams]

    return unigrams + bigrams