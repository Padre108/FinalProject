# Final Project

## Overview
This project is a machine learning pipeline for image classification, organized into data preprocessing, model training, evaluation, and deployment. It uses Keras/TensorFlow for deep learning and is structured for reproducibility and scalability.

## Project Structure
```
final_project_specs.md         # Project specifications and requirements
data/                         # Dataset directory
    train/                    # Training data (normal/opacity)
    val/                      # Validation data (normal/opacity)
    test/                     # Test data (normal/opacity)
models/                       # Saved models (.h5, .keras)
src/                          # Source code
    app.py                   # Main application (possibly for inference or API)
    data_preprocessing.py    # Data preprocessing scripts
    evaluate_model.py        # Model evaluation scripts
    train_model.py           # Model training scripts
    utils.py                 # Utility functions
```

## Setup
1. Clone the repository.
2. (Recommended) Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
- To train the model:
  ```
  python src/train_model.py
  ```
- To evaluate the model:
  ```
  python src/evaluate_model.py
  ```
- To preprocess data:
  ```
  python src/data_preprocessing.py
  ```
- To run the app (if applicable):
  ```
  python src/app.py
  ```

## Models
Trained models are saved in the `models/` directory as `.h5` and `.keras` files.

## Data
Organize your data in the `data/` directory as follows:
- `train/normal/`, `train/opacity/`
- `val/normal/`, `val/opacity/`
- `test/normal/`, `test/opacity/`

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Specify your license here (e.g., MIT, Apache 2.0, etc.).
