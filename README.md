# Artificial-Intelligence

# Project NN – Artificial Neural Networks for Amazon Climate Prediction

---

## 🎯 Objective

Build and evaluate **neural network models** to analyze climate data from the Amazon rainforest (1982–2022) for two tasks:

- **Task A (Classification)**: Predict whether a **hot event** occurs in a given month.  
- **Task B (Regression)**: Predict the **actual temperature** for a given month.

Climate drivers used as input features: **ENSO**, **TSA**, **TNA**, and **NAO** indices.

---

## 📊 Data

- **Input features**: Monthly values of ENSO, TSA, TNA, NAO (1982–2022).  
- **Target**:
  - For Task A: Binary `Hot` label (`1` if temperature > monthly threshold, else `0`).
  - For Task B: Raw monthly temperature (°C).
- **Optional feature**: Month (must be **cyclically encoded** if used):  
  ```python
  month_norm = 2 * π * (month - 1) / 12

## 🧠 Tasks & Requirements

### ✅ Task A: Classification (Hot Event Detection)
- **Label creation**: Define `Hot = 1` if temperature > monthly threshold; otherwise `0`.
- **Visualization**: Bar plot showing the number of hot months per year.
- **Data split**: Random partition into training, validation, and test sets.
- **Preprocessing**: Apply and record all transformations (e.g., feature scaling) **only to input features**—never to the target.
- **Model**:
  - Neural network classifier
  - Trainable parameters < `N_samples / 10`
  - Appropriate loss (e.g., binary cross-entropy), optimizer, learning rate, batch size, and number of epochs
- **Training plot**: Accuracy (y-axis) vs. epochs (x-axis) for both training and validation sets.
- **Evaluation on test set**:
  - Confusion matrix
  - Balanced Accuracy
  - Sensitivity (True Positive Rate, TPR)
  - Specificity (True Negative Rate, TNR)

### ✅ Task B: Regression (Temperature Prediction)

#### **Part 1: Random Split**
- **Data split**: Random partition using the same proportions as Task A.
- **Preprocessing**: Transform **features only**; **do not scale or normalize the target** (temperature).
- **Model**:
  - Neural network regressor
  - Trainable parameters < `N_samples / 10`
  - Suitable loss function (e.g., MAE or MSE)
- **Training plot**: Loss (y-axis) vs. epochs (x-axis) for training and validation sets.
- **Evaluation metrics**:
  - Pearson correlation coefficient (`r`)
  - Mean Absolute Error (MAE)

#### **Part 2: Year-wise Split**
- **Split strategy**: Partition by **entire calendar years** (non-consecutive years allowed), maintaining same train/val/test proportions.
- **Target scaling**: Fit a **separate scaler on training-set targets only**, then apply to validation and test targets.
- **Feature preprocessing**: Reuse **exact same feature transformations** from the random split.
- **Model**: Re-train the **same architecture and hyperparameters** from Part 1 on the year-wise split.
- **Evaluation**:
  - Same metrics: Pearson `r` and MAE
  - Training plot: Loss vs. epochs (train + validation)
