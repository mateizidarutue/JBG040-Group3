# CNN Hyperparameter Optimization with TPE and Hyperband

This project implements hyperparameter optimization for CNN models using Tree-structured Parzen Estimators (TPE) combined with Hyperband pruning for efficient multi-class classification.

## Project Structure

- `src/`
  - `config/`: Configuration files for model and search space
  - `dataset/`: Dataset handling and processing
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
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `data` folder in the project root and place your dataset files there.

4. (Optional) For parallel processing:
   - Set up MySQL locally (user must have root access)
   - Create `.env` file with:
     ```
     STORAGE_CONNECTION_STRING=your_optuna_storage_string
     MYSQL_PASSWORD=your_mysql_password
     ```

## Usage

### Single Model Training

To train a single model with predefined configuration:

```bash
python -m src.train_with_config
```

This uses `src/config/model_config.json` and saves results to `model_output/`.

### Hyperparameter Optimization

Two options are available:

1. Direct execution:

```bash
python -m src.main
```

2. Parallel processing using batch script:

```bash
run_scripts.bat <num_parallel_processes>
```

Models will be saved to:

- `pruned_models/`: Trials terminated by Hyperband
- `completed_models/`: Successfully completed trials

## Configuration

- `search_config.yaml`: Defines hyperparameter search space including:

  - Architecture parameters (conv layers, kernel sizes, normalizations)
  - Training parameters (epochs, learning rates, momentum)
  - Augmentation settings
  - Loss function configurations
  - Regularization parameters

- `model_config.json`: Configuration for single model training with fixed parameters

## Parallel Processing

For distributed optimization:

1. Ensure MySQL is running locally
2. Configure storage connection in `.env`
3. Use `run_scripts.bat` with desired parallel process count

The system uses Optuna with optional database backend for parallel trial processing.
