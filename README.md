# antarctic-captions

antarctic-captions is a small project for generating image descriptions and stories with frozen vision and language models. It combines CLIP, a retrieval-based cache and a pre-trained encoder-decoder model.

A key goal for this project is to be able to generate reasonable captions on a wide distribution of images well beyond what is available in standard captioning datasets, such as COCO.

## Quickstart

A [colab](https://colab.research.google.com/drive/1FwGEVKXvmpeMvAYqGr4z7Nt3llaZz-F8) is available where you can download pre-trained models and generate captions for images.

## Method

At a high level, an image is mapped in CLIP space where it is scored against a large cache of n-grams. The top-k results are passed to BART as a linearized table, which is used to generate candidate captions. These candidates are then rescored by CLIP.

To get BART to generate captions, we use COCO to fine-tune the layernorm parameters of the encoder. This acts as a kind of soft-prompt and tells BART to denoise the input text into a caption.

Out of ~555M parameters from CLIP+BART, only 74K are fine-tuned. Thus the vast majority of parameters are held frozen. The model is thus able to use elements of the cache as part of its captions even if they are atypical or out of distribution of COCO. For example, if "Darth Vader" exists in the cache, the model can utilize this entry to describe an image of Darth Vader.

## Model training

Install requirements:

```
pip install -r requirements.txt
```

Clone CLIP into this project's repository:

```
git clone https://github.com/openai/CLIP
```

Download COCO dataset. Inside `download.sh`, set download_dir to a location where you want to store the data. Then run:

```
./download.sh
```

If you want to use a pre-existing cache, you can download them:

```
wget -m -np -c -U "eye02" -w 2 -R "index.html*" "https://the-eye.eu/public/AI/models/antarctic-captions/postcache.txt"
wget -m -np -c -U "eye02" -w 2 -R "index.html*" "https://the-eye.eu/public/AI/models/antarctic-captions/postcache.npy"
```

To compute a new cache using COCO, run the following. A cache consists of two files: a txt file containing one entry on each line as well as a .npy file containing CLIP embeddings for each line. Full list of settings can be found in `compute_cache.py`.

```
python3 compute_cache.py --datadir=[path to COCO dir] --textfile=[path to save text] --embfile=[path to save numpy file]
```

To train a model

```
python3 trainer.py --datadir=[path to COCO dir] --textfile=[path to cache text] --embfile=[path to cache numpy file]
```

Full list of settings are found in `trainer.py`. The pre-trained model available for download was trained for 6 epochs. You can already start to obtain caption-like outputs after a few thousand steps.

## Generation

To generate captions on the development set, run:

```
python3 sampler.py --datadir=[path to COCO dir] --textfile=[path to cache text] --embfile=[path to cache numpy file] --savedir=[where to save outputs] --ckpt=[path to model checkpoint]
```

Full list of settings are found in `sampler.py`. NOTE: this function differs than the generation done in the notebook. By default, captions here are generated via beam search, whereas sampling and re-scoring is done in the notebook. 

## Custom cache

A custom cache simply requires a text file of entries and a numpy array of their CLIP embeddings. You can use helper functions in `utils.py` to help with this, or do this separately. Once they are available they can be passed into training, sampling or the colab.

You can also augment new entries to the existing cache. For example, by adding "Geoff Hinton", the model can recognize some images of Geoff. This is a direct result of CLIP's capacity to align images and text. See [here](https://twitter.com/dzryk/status/1420432987481591819) for a sample output.

## Generating little stories for images

By using the outputs of the system as a prompt for large LMs, such as GPT-neo or GPT-J, we can generate little stories about the image. A prompt that typically works well is `Once upon a time there was {caption}`. See [here](https://twitter.com/dzryk/status/1418566309923667972) for a few results.

## Qualitative observations of outputs

- Sampling a large number of captions (e.g. 1000) and re-ranking tends to result in more diverse captions that often contain relevant details, at the expense of speed. A beam search is much faster and gives "crisper" captions, but will be less likely to include particular details as these outputs tend to be much more coco-like. For example, a beam search is unlikely to output "Geoff Hinton" and say instead "A man is ..." or "A professor is ..."

- If CLIP is happy to match a particular detail, like a person or city name, the quality of the rest of the caption may degrade.

- Consider lowering the temperature and top_p settings if the model goes off the rails on a specific image.

- The cache used for training the model does not need to be the same cache for inference. The model seems perfectly fine to mix and match. Furthermore, the top-k setting can be modified at inference time to give BART more or less information. Given BART was trained as a denoising autoencoder, it does not seem to be too picky about how long and/or messy its inputs are (within reason).

## Intended use

The code and models provided are intended for research purposes only. Due to the high reliance of CLIP, the model inherits most of its limitations and biases, as described in their model card. COCO captions used for fine-tuning often contain labelling biases, such as annotators attempting to infer unknown attributes from the image context.

## Acknowledgements

Thanks to The-Eye and EleutherAI for hosting our initial models and to EleutherAI members for many discussions.

### What's behind the name?

antarctic-captions is a reference to the original "Show, Attend and Tell" library called [arctic-captions](https://github.com/kelvinxu/arctic-captions). Also, since the majority of parameters are held frozen, it is a very cold model!
