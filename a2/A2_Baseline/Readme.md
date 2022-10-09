# Baseline model in A2

![image-20221008223458024](Readme.assets/image-20221008223458024.png)

## How to use?

There should be a folder called A2_Baseline after unzip the file, please put the folder in the same path of the data folder. e.g.

```
---./
 |---A2_Baseline/
 | |---main.py
 | |...
 |---data/
   |---data.tsv
   |---train.tsv
   |...
```

* Make sure data folder contains all the `.tsv` files used for training since the preprocessing code is not included.

* You should run all the python scripts below under the **parent directory** of `A2_Baseline/`

## Files

* `A2_starter.py` - Starter code given in assignment zip. Function `main(args)` is moved to `main.py`, other functions are slightly modified.

* `Baseline.py` - Defines the Baseline model.

* `ext_meaning.py` - Extract the meanings from trained model's linear layer. Which is used in P4.6. Usage: `python ./A2_Baseline/ext_meaning.py`

* `main.py` - Function `main(args)` included. Used to train and test the model. Arguments are listed in the following "Parser" section.

  For example, a normal output of `main.py` will be like this:

  ![image-20221008225249795](Readme.assets/image-20221008225249795.png)

* `plot.py` - Functions for plotting the curves.

## Parsers

in `main.py`:

```
parser.add_argument("-bs", "--batch_size", type=int, default=16)
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-ml", "--max_len", type=int, default=0)
parser.add_argument("-s", "--save_model", type=bool, default=True)
parser.add_argument("-o", "--overfit_debug", type=bool, default=False)
```

* `-bs`: batch size
* `-e`: training epochs
* `-lr`: learning rate for Adam optimizer
* `-ml`: maximum length for padding indexed sentences in lambda function.
* `-s`: whether to save the model into the `./model` directory. Models won't overlap since timestamps are included in the file name.
* `-o`: whether to use overfit dataset to train the model. 