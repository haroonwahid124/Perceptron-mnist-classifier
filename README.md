# perceptron-mnist-classifier

A perceptron classifier built from scratch and trained on the MNIST handwritten digit dataset. Developed across three iterative parts, each refining the model's training and evaluation, alongside an original baseline implementation for comparison.

## Contents

| File | Description |
|---|---|
| `Perceptron-orig.py` | Original baseline perceptron implementation. |
| `Perceptron_part1.py` | Part 1 perceptron implementation. |
| `main-part1` | Entry point / runner for part 1. |
| `Perceptron_part2.py` | Part 2 perceptron implementation. |
| `main-part2.py` | Entry point / runner for part 2. |
| `Perceptron_part3.py` | Part 3 perceptron implementation. |
| `main-part3.py` | Entry point / runner for part 3. |

## Dataset

This project trains on the MNIST handwritten digit dataset. The dataset zip is **not included** in this repo (excluded via `.gitignore` to keep the repo lightweight). To run the code, download MNIST separately, e.g.:

```bash
python -c "from tensorflow.keras.datasets import mnist; mnist.load_data()"
```

or source it from [the original MNIST site](http://yann.lecun.com/exdb/mnist/) / [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset), and place it where the scripts expect it.

## Usage

```bash
python main-part1.py
python main-part2.py
python main-part3.py
```

## Requirements

Python 3.10 with `numpy` (check individual scripts for any additional imports).
