import torch
import torch.nn.functional as F

from CLIP import clip
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions

import model


def load_ckpt(args):
    """Loads a trained checkpoint."""
    net = model.Model.load_from_checkpoint(args.ckpt)
    net = net.eval().requires_grad_(False).to(args.device)
    return net


def encode_cache(args, cache, model):
    """Compute CLIP embeddings for each entry in cache."""
    pcache = [args.prefix + x for x in cache]
    clip_dim = model.ln_final.normalized_shape[0]
    cache_emb = torch.zeros((len(cache), clip_dim)).to(args.device)
    for idx in range(0, len(cache), args.batch_size):
        batch = pcache[idx:idx+args.batch_size]
        cache_emb[idx:idx+args.batch_size] = model.encode_text(
            clip.tokenize(batch).to(args.device)).float()
    cache_emb /= cache_emb.norm(dim=-1, keepdim=True)
    return cache_emb


def build_table(x, perceiver, cache, cache_emb, topk, return_images=False):
    """
    Maps each image to a linearized row in a table. Each entry in a row
    is delimited by "|". Each entry comes from the topk results in the cache
    as determined by cosine similarity in CLIP space.
    """
    table = []
    x = perceiver.encode_image(x).float()
    x /= x.norm(dim=-1, keepdim=True)
    similarity = (100.0 * x @ cache_emb.T).softmax(dim=-1)
    for idx in range(len(x)):
        row = ''
        values, indices = similarity[idx].topk(topk)
        for _, index in zip(values, indices):
            row += cache[index] + ' | '
        table.append(row)
    if return_images:
        return table, x
    return table


def clip_rescoring(args, net, candidates, x):
    """
    Rescores candidate captions using CLIP. The caption with the highest
    score determined by cosine similarity is returned.
    """
    textemb = net.perceiver.encode_text(
        clip.tokenize(candidates).to(args.device)).float()
    textemb /= textemb.norm(dim=-1, keepdim=True)
    similarity = (100.0 * x @ textemb.T).softmax(dim=-1)
    _, indices = similarity[0].topk(1)
    return candidates[indices[0]]


def generate(args, net, loader):
    """
    Generates a single caption per image in the DataLoader. For each image:
    1. Map to a linearized table entry
    2. Generate a candidate list of captions via sampling
    3. Rescore each candidate using CLIP.
    Generated captions as well as references are stored for external scoring.
    """
    pred, ref = [],[]
    for batch in loader:
        src, tgt = batch
        src, imgs = build_table(src.to(args.device), 
                                perceiver=net.perceiver,
                                cache=net.cache,
                                cache_emb=net.cache_emb,
                                topk=args.topk,
                                return_images=True)
        for idx, x in enumerate(src):
            caps = [tgt[i][idx] for i in range(args.ncaptions)]
            inputs = net.tokenizer.encode(x, return_tensors='pt').to(args.device)
            out = net.model.generate(inputs,
                                     do_sample=False,
                                     num_beams=args.num_return_sequences,
                                     temperature=args.temperature,
                                     top_p=args.top_p,
                                     num_return_sequences=args.num_return_sequences)
            candidates = []
            for seq in out:
                candidates.append(net.tokenizer.decode(seq, skip_special_tokens=True))
            out = clip_rescoring(args, net, candidates, imgs[idx][None,:])
            print(out)
            pred.append(out)
            ref.append(caps)
    return pred, ref