# Brain Tumor MRI Image Classification

**Project**: Brain Tumor MRI Image Classification  
**Description**: Train a CNN/Transfer-learning model to classify brain MRI images (e.g., glioma, meningioma, pituitary, no tumor) and deploy a Streamlit app to make predictions.

## Repo contents
- `train.py` - training script (TensorFlow / Keras) that expects a folder dataset structured as `data/train/<class>/*` and `data/val/<class>/*`
- `app.py` - Streamlit web app for uploading an MRI image and getting a predicted class and confidence
- `utils.py` - helper functions for preprocessing and loading the model
- `requirements.txt` - Python dependencies
- `model_stub.md` - guidance for saving/loading your trained .h5 file
- `README.md` - this file

## Quick start (local)

1. Create a Python virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. Prepare dataset folder:
   ```
   data/
     train/
       glioma/
       meningioma/
       pituitary/
       no_tumor/
     val/
       glioma/
       meningioma/
       pituitary/
       no_tumor/
   ```
   (Use Kaggle Brain Tumor MRI dataset or your preferred dataset.)

3. Train:
   ```bash
   python train.py --epochs 10 --batch_size 32 --img_size 224
   ```
   This will save `best_model.h5` in the `models/` folder.

4. Run Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How to push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: brain tumor MRI classification"
gh repo create YOUR_GITHUB_USERNAME/brain-tumor-mri-classification --public --source=. --remote=origin
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username. If you don't have GitHub CLI, create an empty repo on GitHub and follow the `git remote add` and push steps.

## Notes
- The training script uses TensorFlow/Keras. For large datasets, use GPU-enabled environment (Colab or local GPU).
- The Streamlit app expects `models/best_model.h5` to exist. See `model_stub.md` for details.
