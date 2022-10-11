# CNN model in A2

![image-20221011155116550](Readme.assets/image-20221011155116550.png)

## How to use?

Same as A2_Baseline, There should be a folder called A2_CNN after unzip the file, please put the folder **in the same path of the data folder.**

```
---A2/
 |---A2_Baseline/
 | |---main.py
 | |...
 |---A2_CNN/
 | |---main.py
 | |...
 |---data/
 | |---data.tsv
 | |---train.tsv
 | |...
 |---models/
 |---results/
```

* Make sure data folder contains all the `.tsv` files used for training since the preprocessing code is not included.

* You should run all the python scripts below under the **parent directory** of `A2_Baseline/`, which is the folder `A2/` or anything similar to that.

## Files

* `A2_starter.py` - Starter code given in assignment zip.

* `Classifier.py` - Defines the CNN model.

* `ext_meaning.py` - Extract the meanings from trained model's linear layer. Which is used in P5.3. 

  Usage: `python ./A2_Baseline/ext_meaning.py`

  **Notes: you should specify which model to extract meaning. Change model path in the script before running this script.**

* `main.py` - Main entrance. Used to train and test the model. Arguments are listed in the following "Parser" section.

* `train.py` - Functions for training and testing the CNN Model, Performing Grid Search.

* `utils.py` - Functions for plotting curves, saving results, saving models.

* `gs_args.json` - Hyperparameter matrix for grid search. When running `main.py` using `-g 1`， other arguments will be blocked and the training function will use arguments from this json file instead.

## Parsers

in `main.py`:

```python
parser.add_argument("-bs", "--batch_size", type=int, default=16)
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-ml", "--max_len", type=int, default=0)
parser.add_argument("-s", "--save_model", type=bool, default=True)
parser.add_argument("-o", "--overfit_debug", type=bool, default=False)

parser.add_argument("-k1", "--k1", type=int, default=2)
parser.add_argument("-n1", "--n1", type=int, default=10)
parser.add_argument("-k2", "--k2", type=int, default=4)
parser.add_argument("-n2", "--n2", type=int, default=10)
parser.add_argument("-f", "--freeze_embedding", type=bool, default=True)
parser.add_argument("-g", "--grid_search", type=bool, default=False)
parser.add_argument("-v", "--val_epoch", type=int, default=2)
```

* `-bs`: batch size
* `-e`: training epochs
* `-lr`: learning rate for Adam optimizer
* `-ml`: maximum length for padding indexed sentences in lambda function.
* `-s`: whether to save the model into the `./model` directory. Models won't overlap since timestamps are included in the file name.
* `-o`: whether to use overfit dataset to train the model. 

* `-k1`, `-n1`, `-k2`, `-n2`: CNN structure parameters.
* `-f` whether to freeze the embedding layer of the model.
* `-g` whether to use grid search or train the model only once.
* `-v` determines how much epochs per 1 validation.

## Issues

If there's any issues regarding running the codes, please contact haocheng.wei@mail.utoronto.ca