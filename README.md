# CNN Hyperparameter Optimization with TPE and Hyperband

This project implements hyperparameter optimization for CNN models using Tree-structured Parzen Estimators (TPE) combined with Hyperband pruning for efficient multi-class classification.

## Project Structure

- `src/`
  - `config/`: Configuration files for model and search space
  - `dataset/`: Dataset handling and processing
  - `ethics/`: Functions used to address ethical concerns
  - `models/`: CNN model implementations
  - `search/`: Hyperparameter optimization implementation
  - `trainer/`: Model training logic
  - `utils/`: Utility functions for losses, optimizers, etc.
- `data/`: Place your dataset here
- `pruned_models/`: Storage for pruned model trials
- `completed_models/`: Storage for completed model trials
- `model_output/`: Output directory for single model training

## Setup

1. Create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# For Windows:
.\.venv\Scripts\activate
# For Linux/Mac:
source .venv/bin/activate
```

2. Install dependencies:

```bash
# For MPS / CPU-only
pip install torch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt

# For CUDA-enabled GPU
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

3. Create a `data` folder in the project root and place your dataset files there.

4. (Optional) For parallel processing:
   - Set up MySQL locally (user must have root access)
   - Create `.env` file with:
     ```
     STORAGE_CONNECTION_STRING=mysql+pymysql://root:<your_mysql_password>@localhost:3306/optuna_study
     MYSQL_PASSWORD=<your_mysql_password>
     ```

## Usage

### Single Model Training

To train a single model with predefined configuration:

```bash
python -m src.train_from_config
```

This uses `src/config/model_config.json` and saves results to `saved_outputs/completed/manual_{idx}`.

### Hyperparameter Optimization

Two options are available:

1. Direct execution:

```bash
python -m src.main
```

2. Parallel processing using batch script (Windows only):

```bash
run_scripts.bat <num_parallel_processes> // 4 usually works the best
```

Models will be saved to:

- `saved_outputs/pruned/trial_{idx}`: Trials terminated by Hyperband
- `saved_outputs/completed/trial_{idx}`: Successfully completed trials

## Configuration

- `search_config.yaml`: Defines hyperparameter search space including:

  - Architecture parameters (conv layers, kernel sizes, normalizations)
  - Training parameters (epochs, learning rates, momentum)
  - Augmentation settings
  - Loss function configurations
  - Regularization parameters
  - And more

- `model_config.json`: Configuration for single model training with fixed parameters

  - Can contain multiple configurations to train sequentially
  - Each configuration specifies a complete set of hyperparameters

- `static_config.yaml`: Contains static configuration parameters:
  - Training budget parameters:
    ```yaml
    min_budget: 1 # Minimum number of epochs for Hyperband
    max_budget: 25 # Maximum number of epochs for training
    eta: 3 # Reduction factor for Hyperband
    total_trials: 50 # Total number of trials to run
    ```
  - Model parameters:
    ```yaml
    batch_size: 256 # Batch size for training
    input_size: 128 # Input image size
    num_classes: 6 # Number of output classes
    direction: "minimize" # Optimization direction
    ```
  - Data paths:
    ```yaml
    train_data: "data/X_train.npy"
    train_labels: "data/Y_train.npy"
    test_data: "data/X_test.npy"
    test_labels: "data/Y_test.npy"
    ```

## Parallel Processing

For distributed optimization (works only on Windows):

1. Ensure MySQL is running locally
2. Configure storage connection in `.env`
3. Use `run_scripts.bat` with desired parallel process count

The system uses Optuna with optional database backend for parallel trial processing.

## Best Model Retrieval

To identify the best-performing models after a hyperparameter search, use:

```bash
python -m src.get_best_model
```

This script will:

- Scan all completed trials in saved_outputs/completed/

- Sort them based on final validation loss or accuracy

- Print a ranked list of top-performing models

This helps you easily choose the most promising models for deployment or further analysis.

## CAM & Saliency Visualization

After training or optimizing models, you can visualize Class Activation Maps (CAM), Grad-CAM++, and Saliency Maps for model interpretability:

```bash
python -m src.run_cam --trial <trial_number>
```
This script will:

- Load the model from saved_outputs/completed/trial_<trial_number>/model.pt

- Visualize and save CAM, Grad-CAM++, and Saliency Maps for one image per class

- Store the outputs in the cam_outputs/ folder

Ensure this directory exists or will be automatically created during execution.

