# gt5 Grammatical Error Correction Model

These are the instructions for loading and running the grammatical error correction model on the datasets CoNLL-2014 and cLang-8.

## Requirements

- Python 3.x
- datasets
- pandas
- transformers
- Errant
- subprocess
- os
- random

## Installation

1. Navigate to the project folder:
    ```
    cd gt5_files
    ```
3. Install the required packages:
    ```
    pip install datasets
    ```
    ```
    pip install transformers
    ```
    ```
    pip install pandas
    ```
    ```
    pip install subprocess
    ```
    ```
    pip install Errant
    ```
    ```
    pip install os
    ```
    ```
    pip install random
    ```

## Usage

To run the main file or the model, use the following command:
 ```
    python3 main.py
 ```

After the completion of above file, to run the evaluation file on Errant, use the following command:
 ```
    python3 Gecmodel_errEval.py
 ```