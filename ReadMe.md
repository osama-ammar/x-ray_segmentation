# Project : ROI detection
>
> in this trial we converted the problem of roi detection into segmentation which get better results

## Description

- This project is designed to train a model for jaw roi detection, where:  
  - input ---> is a MIP projections of a dicom (Sagittal and coronal)
  - output ---> is a mask on the jaws roi
- The used model is UNET
- We used accuracy, precision, iou as metrics.
- loss function : cross entropy

## Usage

1. Navigate to the project directory.
2. edit the configuration file if you need .
    - if you need to reproduce an exact trail , get the parameters of the saved `config.yaml` of this trial and put these parameters in
    `model_config.yaml`
3. Run the following command to execute the project:
   `python main.py`

## Configuration

The project can be configured by editing `model_config.yaml` file . The available configuration options are as follows:

```
- `input_size`: The size of input images (default: 400).
- `in_channels`: The number of input channels (default: 1).
- `out_channels`: The number of output channels (default: 2).
- `epochs`: The number of training epochs (default: 100).
- `batch_size`: The batch size for training (default: 8).
- `learning_rate`: The learning rate for optimization (default: 1e-3).
- `log_period`: The interval for printing training logs (default: 1).
- `criterion` : The Loss function used (default:crossentropy)
- `optimizer`: The optimizer (default:AdamW).
- `mode`: The mode of operation (train/test/export).
- `weights_path`: The path to load the checkpoint for testing or exporting .
- `dataset_path`: The path to the dataset directory .
- `results_path`: The path to save the result files .
```

## Dataset

The dataset used for training and testing should be placed in the specified `dataset_path` directory.

## Training

To train the model, set the `mode` argument to `train`. The training progress, including loss and accuracy, will be displayed during training.

## Testing

To test the trained model, set the `mode` argument to `test`. The model will be loaded from the `checkpoint_path`, and the test results will be displayed.

## Exporting

To export the trained model, set the `mode` argument to `export`. The model will be loaded from the `checkpoint_path`, and the exported model will be saved in the `model_path`.

## Results

The results, including trained models and logs, will be saved in the specified `model_path` and `result_path` directories, respectively. Each run of the program will create a subdirectory in the `result_path` directory named with the model type and the current date and time.

--------------------------------------------------------

### Model performance

- Generally , accuracy > 98%
- IOU = ....

--------------------------------------------------------

## TODO

- [x] editing hyperparams in only ony file
- [x] integrating DVC
- [ ] automated statistical analysis when dataset changes
- [ ] experimenting weight and biases
- [ ] implement early stopping
- [x] use MLFlow
  - [x] log params
  - [x] log models
  - [x] log dataset
- [\] template for augmentations - separating it
- [ ] include training time and computer resources in trial summary
- [ ] apply some performance optimization mentioned in [https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html]
