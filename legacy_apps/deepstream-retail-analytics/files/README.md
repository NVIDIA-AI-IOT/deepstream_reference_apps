# Description of files

* [0001_compressed.h264](./0001_compressed.h264) - Input file for running the pipeline
* [basketClassifier.etlt](./basketClassifier.etlt) - Model file for classifying people with and without baskets
* [basket_classifier_labels.txt](./basket_classifier_labels.txt) - Label file for the above model

# RECOMMENDED: Verify checksum after cloning files using git LFS

LFS files are not cloned by default when you clone the repository

Clone model and input files by running

```bash
git lfs pull
```

Verify the checksum of files after cloning

```bash
sha512sum -c checksum.txt
```