# This is the official implementation of the paper [Efficient Shapley Values Calculation for Transformer Explainability] 
Our code is based on [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838)


## Introduction: please refer to
[Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838).

We introduce a novel method which allows to visualize classifications made by a Transformer based model for both vision and NLP tasks.
Our method also allows to visualize explanations per class.

Method consists of 3 phases:

1. Run inference once in auxiliary model to record essential values.

2. Run the 2nd inference in masked model to get visualization.


## Credits
ViT implementation is based on:
- https://github.com/rwightman/pytorch-image-models
- https://github.com/lucidrains/vit-pytorch
- pretrained weights from: https://github.com/google-research/vision_transformer

BERT implementation is taken from the huggingface Transformers library:
https://huggingface.co/transformers/

ERASER benchmark code adapted from the ERASER GitHub implementation: https://github.com/jayded/eraserbenchmark

Text visualizations in supplementary were created using TAHV heatmap generator for text: https://github.com/jiesutd/Text-Attention-Heatmap-Visualization

## Reproducing results on ViT

### Section A. Segmentation Results

Example:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/imagenet_seg_eval.py --method perturb_fast --imagenet-seg-path /path/to/gtsegs_ijcv.mat

```
[Link to download dataset](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat).

In the exmaple above we run a segmentation test with our method. Notice you can choose which method you wish to run using the `--method` argument. 
You must provide a path to imagenet segmentation data in `--imagenet-seg-path`.

### Section B. Perturbation Results

Example:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/generate_visualizations.py --method perturb_fast --imagenet-validation-path /path/to/imagenet_validation_directory
```

Notice that you can choose to visualize by target or top class by using the `--vis-cls` argument.

Now to run the perturbation test run the following command:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/pertubation_eval_from_hdf5.py --method perturb_fast
```

Notice that you can use the `--neg` argument to run either positive or negative perturbation.
