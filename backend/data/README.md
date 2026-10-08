# Concrete Crack Detection Dataset

## Recommended dataset

Use **Concrete Crack Images for Classification** by Çağlar Fırat Özgenel, hosted on Mendeley Data:

- Dataset page: https://data.mendeley.com/datasets/5y9wdsg2zt/2
- Official download: https://data.mendeley.com/public-api/zip/5y9wdsg2zt/download/2
- Version: 2; DOI: `10.17632/5y9wdsg2zt.2`
- License: CC BY 4.0 (cite the authors and dataset when using it)
- Size: 40,000 RGB images, 227 × 227 pixels
- Labels: 20,000 `Positive` (crack) and 20,000 `Negative` (no crack)

The archive is approximately 230 MB. The extracted dataset is already available beside the repository at `../Dataset/Concrete Crack Images for Classification/`; the trainer uses this location by default. The dataset is not copied into the Git repository because of its size.

## Expected local layout

The current dataset follows the layout expected by the trainer:

```text
Dataset/Concrete Crack Images for Classification/
├── Positive/   # cracked concrete; target label 1
└── Negative/   # non-cracked concrete; target label 0
```

The trainer also recognizes `crack` / `no crack` folder names. To use a different dataset location, pass it with the trainer's `--data-dir` option. Keep downloaded data out of source control.

## Model task

This dataset supports **binary image classification**:

- Input: RGB concrete image
- Output: `crack` / `no_crack`
- Recommended initial split: stratified train/validation/test split, for example 70%/15%/15%.
- Avoid augmenting validation and test images. Apply augmentation only to training data.

The dataset was produced by extracting patches from 458 high-resolution source images. If source-image identity is available in the downloaded metadata, split by original source image rather than randomly splitting patches; this helps avoid near-duplicate patches leaking between train and test sets. Report this limitation in the project evaluation if source identities are unavailable.

## Citation

Özgenel, Ç. F. (2019). *Concrete Crack Images for Classification* (Version 2). Mendeley Data. https://doi.org/10.17632/5y9wdsg2zt.2

Also cite the associated paper when appropriate: Özgenel, Ç. F., & Gönenç Sorguç, A. (2018). *Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings*. ISARC 2018.
