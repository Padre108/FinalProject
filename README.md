# Chest X-ray Opacity Classifier

A ResNet50-based deep learning mini app to classify chest X-rays as **Normal** or **Opacity** (pneumonia/fluid).

## Quick Start

1. **Setup:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run src/app.py
   ```
   The app opens at `http://localhost:8501`

## How to Use the App

1. **Upload an X-ray:**
   - Click "Upload a chest X-ray image" on the left panel
   - Select a JPG, JPEG, or PNG file
   - The app validates it's a real chest X-ray

2. **View Results:**
   - **Prediction:** Normal or Opacity detected (with confidence %)
   - **Grad-CAM Heatmap:** Colored overlay showing which regions influenced the prediction
   - **Raw Probability:** Expand "More Details" for technical metrics

3. **Interpret the Heatmap:**
   - Bright red/yellow = model focused on these regions
   - Blue/cool colors = less attention
   - Use alongside radiologist review—not a diagnosis!

## Training & Evaluation

3. **Or train the model:**
   ```bash
   python src/train_model.py
   ```

4. **Evaluate the model:**
   ```bash
   python src/evaluate_model.py
   ```

## Project Structure
```
data/              # Train/val/test splits (normal/opacity folders)
models/            # Saved .keras models
src/
  ├─ app.py        # Streamlit web interface
  ├─ train_model.py
  ├─ evaluate_model.py
  ├─ data_preprocessing.py
  └─ utils.py
```

## Model Architecture

- **Backbone:** ResNet50 (ImageNet pre-trained)
- **Head:** Classification layer with dropout regularization
- **Loss:** Binary Crossentropy
- **Training:** 2-stage (freeze backbone → fine-tune)
- **Threshold:** Optimized for high recall (medical priority)

## Key Features

✅ Grad-CAM heatmaps show which regions influenced predictions  
✅ Privacy-aware audit logging (file hash only, no raw images stored)  
✅ Image validation to reject non-X-ray uploads  
✅ Optimized decision threshold for medical triage  

## Data Organization
```
data/train/normal/    data/train/opacity/
data/val/normal/      data/val/opacity/
data/test/normal/     data/test/opacity/
```

---

**Disclaimer:** Triage aid only. Not a clinical diagnosis. Always pair with radiologist review.


