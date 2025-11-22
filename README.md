# gtfread

A microscopic library to read GTF (Gene Transfer Format) files.

```sh
pip install gtfread
```

<!-- 
Takes some inspiration from [pycli](https://github.com/garenchan/pycli) although I didn't know 
this existed before starting. This lib is even smaller than `pycli` and does not require any setup 
and has no dependencies.

The good thing about this is, you don't need to write or maintain any code for the CLI, all your types,
variables, descriptions are introspected from the code itself, so all you do is throw on a decorator.

## Usage

```bash
ucli path.to.your.function --help
```

For this to work, all you need to do is have a function in the module with the right type-annotations
and all optional values should have a default value.

```python
# Here model_name is required and data_path is optional
def train(
        model_name: str, 
        data_path: str = "./path/to/data"):
    ...
```

Alternatively, the command decorator can be used to directly turn the function into a CLI command

```python
# main.py
from upycli.decorator import command

@command
def train(model_name: str, data_path: str = "./path/to/data"):
    """
    A function to train a model
    """
    ...
```
```bash
python main.py train --help
>>> usage: main.py [-h] --model_name MODEL_NAME [--data_path DATA_PATH]

A function to train a model

options:
  -h, --help               show this help message and exit
  --model_name MODEL_NAME  str
  --data_path DATA_PATH    str (default ./path/to/data)
```

Your variables, types, required values and description are introspected from the code itself, 
so all you do is throw on the decorator. -->
