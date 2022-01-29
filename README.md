# `ffcv` ImageNet Training
A minimal, single-file PyTorch ImageNet training script designed for hackability. Run `train_imagenet.py` to get...
- ...high accuracies on ImageNet
- ...with as many lines of code as the PyTorch ImageNet example
- ...in 1/10th the time.

## Results
Train models more efficiently, either with 8 GPUs in parallel or by training 8 ResNet-18's at once.
<img src="assets/perf_scatterplot.svg" width='830px'/>

See benchmark setup here: [https://docs.ffcv.io/benchmarks.html](https://docs.ffcv.io/benchmarks.html).

## Citation
If you use this setup in your research, cite:

```
@misc{leclerc2022ffcv,
    author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
    title = {ffcv},
    year = {2022},
    howpublished = {\url{https://github.com/libffcv/ffcv/}},
    note = {commit xxxxxxx}
}
```
(Make sure to replace ``xxxxxxx`` above with the hash of the commit used!)

## Configurations
The configuration files corresponding to the above results are:

| Link to Config                                                                                                                         |   top_1 |   top_5 |   # Epochs |   Time (mins) | Architecture   | Setup    |
|:---------------------------------------------------------------------------------------------------------------------------------------|--------:|--------:|-----------:|--------------:|:---------------|:---------|
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_88_epochs.yaml'>Link</a> | 0.784 | 0.941  |         88 |       77.2 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_56_epochs.yaml'>Link</a> | 0.780 | 0.937 |         56 |       49.4 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_40_epochs.yaml'>Link</a> | 0.772 | 0.932 |         40 |       35.6 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_32_epochs.yaml'>Link</a> | 0.766 | 0.927 |         32 |       28.7 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_24_epochs.yaml'>Link</a> | 0.756 | 0.921 |         24 |       21.7  | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_16_epochs.yaml'>Link</a> | 0.738 | 0.908 |         16 |       14.9 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_88_epochs.yaml'>Link</a> | 0.724 | 0.903   |         88 |      187.3  | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_56_epochs.yaml'>Link</a> | 0.713  | 0.899 |         56 |      119.4   | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_40_epochs.yaml'>Link</a> | 0.706 | 0.894 |         40 |       85.5 | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_32_epochs.yaml'>Link</a> | 0.700 | 0.889 |         32 |       68.9   | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_24_epochs.yaml'>Link</a> | 0.688  | 0.881 |         24 |       51.6 | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_16_epochs.yaml'>Link</a> | 0.669 | 0.868 |         16 |       35.0 | ResNet-18      | 1 x A100 |

## Training Models

First pip install the requirements file in this directory:
```
pip install -r requirements.txt
```
Then, generate an ImageNet dataset; make the dataset used for the results above with the following command (`IMAGENET_DIR` should point to a PyTorch style [ImageNet dataset](https://github.com/MadryLab/pytorch-imagenet-dataset):

```bash
# Required environmental variables for the script:
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/

# Starting in the root of the Git repo:
cd examples;

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90
```
Then, choose a configuration from the [configuration table](#configurations). With the config file path in hand, train as follows:
```bash
# 8 GPU training (use only 1 for ResNet-18 training)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
python train_imagenet.py --config-file rn50_configs/<your config file>.yaml \
    --data.train_dataset=/path/to/train/dataset.ffcv \
    --data.val_dataset=/path/to/val/dataset.ffcv \
    --data.num_workers=12 --data.in_memory=1 \
    --logging.folder=/your/path/here
```
Adjust the configuration by either changing the passed YAML file or by specifying arguments via [fastargs](https://github.com/GuillaumeLeclerc/fastargs) (i.e. how the dataset paths were passed above).

## Training Details
<p><b>System setup.</b> We trained on p4.24xlarge ec2 instances (8 A100s).
</p>

<p><b>Dataset setup. Generally larger side length will aid in accuracy but decrease
throughput:</b>

 - ResNet-50 training: 50% JPEG 500px side length
 - ResNet-18 training: 10% JPEG 400px side length

</p>


<p><b>Algorithmic details.</b> We use a standard ImageNet training pipeline (à la the PyTorch ImageNet example) with only the following differences/highlights:

- SGD optimizer with momentum and weight decay on all non-batchnorm parameters
- Test-time augmentation over left/right flips
- Progressive resizing from 160px to 192px: 160px training until 75% of the way through training (by epochs), then 192px until the end of training.
- Validation set sizing according to ["Fixing the train-test resolution discrepancy"](https://arxiv.org/abs/1906.06423): 224px at test time.
- Label smoothing
- Cyclic learning rate schedule
</p>

Refer to the code and configuration files for a more exact specification.
To obtain configurations we first gridded for hyperparameters at a 30 epoch schedule. Fixing these parameters, we then varied only the number of epochs (stretching the learning rate schedule across the number of epochs as motivated by [Budgeted Training](https://arxiv.org/abs/1905.04753)) and plotted the results above.

## FAQ
### Why is the first epoch slow?
The first epoch can be slow for the first epoch if the dataset hasn't been cached in memory yet.

### What if I can't fit my dataset in memory?
See this [guide here](https://docs.ffcv.io/parameter_tuning.html#scenario-large-scale-datasets).

### Other questions
Please open up a [GitHub discussion](https://github.com/MadryLab/ffcv/discussions) for non-bug related questions; if you find a bug please report it on [GitHub issues](https://github.com/MadryLab/ffcv/issues).
