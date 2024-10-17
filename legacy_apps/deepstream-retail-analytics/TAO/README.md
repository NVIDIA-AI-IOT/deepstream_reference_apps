# Training a Classification Model using TAO Toolkit

# Dataset Structure

TAO Toolkit requirements

* One folder for training, validation and testing
* Each of the above folders should contain one folder for each class that the model should learn ("hasBasket" and "noBasket" in this case)

```
|--dataset_root:
    |--train
        |--hasBasket:
            |--1.jpg
            |--2.jpg
        |--noBasket:
            |--01.jpg
            |--02.jpg
    |--val
        |--hasBasket:
            |--3.jpg
            |--4.jpg
        |--noBasket:
            |--03.jpg
            |--04.jpg
    |--test
        |--hasBasket:
            |--5.jpg
            |--6.jpg
        |--noBasket:
            |--05.jpg
            |--06.jpg
```

If your dataset is in KITTI format (object detection) and you would like to convert it to a classification dataset, you can use the [`kitti_to_classification.py`](kitti_to_classification.py) file provided in this directory.


# Training a classification model

## Prerequisites

* Installation of TAO toolkit
* Download a PTM (pre-trained model) from NGC. We will use [resnet34](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_detectnet_v2/files) for this example

**Note:** You could use NGC command line tool or curl to download the model

## Steps to train a classification model

Refer to [TAO documentation](https://docs.nvidia.com/tao/tao-toolkit/text/image_classification.html) for more information on how to modify the spec file for training.

We need to set a key to encode the model. The same key is to be used when saving/loading the model. In this example the key is set as `nvidia_tlt`

* Create a directory to store checkpoints

```bash
mkdir classification_model
```

* Train the model

```bash
tao classification train -e SPECS_classification_train.txt -k nvidia_tlt -r classification_model
```

* Evaluate the model

The path for the evaluation dataset is specified in the `eval_config` in the spec file

```bash
tao classification evaluate -e SPECS_classification_train.txt -k nvidia_tlt
```

* OPTIONAL: Running inference on an image/directory using the model

```bash
tao classification inference -m <model> -i <image> -k nvidia_tlt -cm <classmap> -e SPECS_classification_train.txt
```

* Exporting the model

Checkpoints for the model trained are located in the `classification_model/weights` directory as `.tlt` files(The directory to store output is set during the training step).

Training logs are located in the same folder as JSON and CSV files.

Pick a model to export depending on the loss/accuracy values

```bash
tao classification export -m <model> -k nvidia_tlt -o basketClassifier.etlt
```

* Deploying the model using DeepStream

There are two ways to use the above exported model with DeepStream

**Option 1:** Use the above exported `.etlt` model directly with DeepStream.

**Option 2:** Use the `tao-converter` and generate a device specific engine file. The generated engine file should also be specified in the config file for the inference engine. Refer to [basket_classifier.yml](../configs/basket_classifier.yml) for an example config file.

**Generating an engine using tao-converter**

*Output Nodes:* Since this is a classification model, there is only only one classification node "predictions/Softmax"

*Dimensions:* Dimensions for the below command should be the same as it was specified in the [SPECS_classification_train.txt](./SPECS_classification_train.txt)

*input_file:* The input for the below command is the model exported in the previous command

```bash
tao-converter -k nvidia_tlt -d 3,224,224 -o predictions/Softmax basketClassifier.etlt
```