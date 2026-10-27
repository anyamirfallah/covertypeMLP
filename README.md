# Forest Cover Type Classification with PyTorch

A deep learning project that uses a **Multi-Layer Perceptron (MLP)** to classify forest cover types from cartographic and environmental features. Built with PyTorch and Scikit-learn, the project includes data preprocessing, model training, early stopping, evaluation metrics, and interactive visualizations.

## Overview

The goal of this project is to predict forest cover types using the [Covertype dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html).

The model learns from 54 numerical and binary cartographic features to classify each observation into one of seven forest cover categories.

The project focuses not only on model training but also on understanding its behavior through detailed evaluation and visualization.

## Dataset

The project uses the Covertype dataset provided by Scikit-learn.

- **Input features:** 54
- **Target classes:** 7 forest cover types
- **Task:** Multiclass classification
- **Preprocessing:** Min-max normalization
- **Data splitting:** Stratified training, validation, and test sets

The dataset is divided into:

| Split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Stratified splitting preserves the approximate distribution of the target classes across the three sets.

The `MinMaxScaler` is fitted exclusively on the training data and then applied to the validation and test sets to prevent data leakage.

## Model Architecture

The classifier is a fully connected neural network implemented using `torch.nn.Module`.

```text
Input (54 features)
        |
        v
Linear (54 → 512)
        |
Batch Normalization
        |
ReLU + Dropout (0.2)
        |
        v
Linear (512 → 256)
        |
Batch Normalization
        |
ReLU + Dropout (0.2)
        |
        v
Linear (256 → 128)
        |
Batch Normalization
        |
ReLU + Dropout (0.2)
        |
        v
Linear (128 → 7)
        |
        v
Class Logits
```

### Architectural choices

- **Batch Normalization:** Stabilizes intermediate activations during training.
- **ReLU:** Introduces non-linearity.
- **Dropout:** Reduces overfitting by randomly disabling a proportion of activations during training.
- **Kaiming initialization:** Initializes linear-layer weights for networks using ReLU activations.
- **Seven output units:** Produce logits for the seven target classes.

The model returns logits rather than applying softmax explicitly, since `CrossEntropyLoss` handles the required log-softmax computation internally.

## Training Configuration

| Parameter | Value |
|---|---|
| Hidden dimensions | 512, 256, 128 |
| Dropout | 0.2 |
| Batch size | 1,024 |
| Evaluation batch size | 2,048 |
| Maximum epochs | 50 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Weight decay | 0.00001 |
| Loss function | Cross-Entropy Loss |
| Learning-rate scheduler | Cosine Annealing |
| Early-stopping patience | 7 epochs |
| Minimum improvement | 0.0001 |
| Random seed | 42 |

### Training strategy

The model is trained using mini-batch gradient descent with Adam optimization.

A cosine annealing scheduler adjusts the learning rate throughout training. Early stopping monitors validation macro-F1, and the best checkpoint is saved when the validation score improves by more than the configured minimum delta.

After training, the best saved model weights are restored before the final test evaluation.

The implementation also supports GPU acceleration when CUDA is available.

## Evaluation

The model is evaluated using several complementary metrics:

- **Accuracy:** Overall proportion of correctly classified samples.
- **Balanced accuracy:** Accounts for differences in class frequencies.
- **Macro-F1:** Gives equal weight to each class's F1 score.
- **Weighted-F1:** Weights each class according to its support.
- **Precision and recall:** Measure class-specific prediction quality.
- **Cross-entropy loss:** Measures the discrepancy between predicted class probabilities and true labels.

Macro-F1 is used as the primary model-selection metric to account for class imbalance.

## Interactive Visualizations

The notebook generates interactive visualizations using Plotly.

| Visualization | Purpose |
|---|---|
| Training dashboard | Tracks training and validation loss, accuracy, and macro-F1 across epochs |
| Confusion matrix | Examines class-by-class prediction errors |
| Class distribution | Compares true and predicted class frequencies |
| Per-class metrics | Compares precision, recall, and F1 score for each class |
| Feature distributions | Compares raw and scaled distributions of selected features |
| Per-class test loss | Identifies classes with higher average cross-entropy loss |

These visualizations help analyze model convergence, class imbalance, generalization, and potential weaknesses in individual classes.

The notebook exports several visualizations as standalone HTML files for convenient viewing and sharing.

## Technologies Used

- **Python** — Core programming language
- **PyTorch** — Neural network implementation and training
- **NumPy** — Numerical computation
- **Scikit-learn** — Dataset loading, preprocessing, splitting, and evaluation
- **Plotly** — Interactive data visualization
- **Jupyter Notebook** — Experimentation and analysis

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Install dependencies

```bash
pip install numpy torch scikit-learn plotly jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook covertypemlp.ipynb
```

Alternatively, open the notebook in Visual Studio Code or Google Colab.

### 4. Run the experiment

Execute the notebook cells in order.

The Covertype dataset is downloaded through Scikit-learn when necessary. The notebook uses a local `scikit_learn_data` directory for dataset storage.

Depending on your hardware, downloading the dataset and training the model may take some time.

## Generated Artifacts

The notebook is configured to generate the following files:

```text
best_mlp.pth
training_dashboard.html
interactive_confusion_matrix.html
class_distribution.html
per_class_metrics.html
feature_distributions.html
per_class_test_loss.html
```

- `best_mlp.pth` contains the best model checkpoint and its training configuration.
- The HTML files contain interactive evaluation visualizations.

These files are generated when their corresponding notebook cells execute successfully.

## Future Improvements

Potential extensions include:

- Comparing the MLP against Random Forest, XGBoost, and other classification models.
- Performing systematic hyperparameter tuning.
- Adding permutation-based feature importance analysis.
- Investigating errors in underperforming classes.
- Comparing macro-F1, balanced accuracy, and per-class recall across models.
- Tracking experiments and comparing different architectures.

## Reproducibility

The notebook sets the random seed to `42` for Python, NumPy, and PyTorch and uses stratified data splitting.

Exact reproducibility is not guaranteed across different hardware, software versions, or nondeterministic GPU operations.

## License

No license has been specified for this project yet. Add a `LICENSE` file if you intend to distribute the code for reuse.
