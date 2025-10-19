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


# Project RL – Reinforcement Learning: Q-learning vs SARSA with Teacher Guidance

---

## 🎯 Objective

Implement and compare **Q-learning** and **SARSA** in an 11×11 grid world, then extend both with a **teacher-student interactive reinforcement learning (IntRL)** framework where pre-trained agents provide probabilistic advice to new learners.

---

## 🌍 Environment (`env.py`)

- **Grid size**: 11 × 11  
- **Goal**: Fixed at `(10, 10)` → **+25 reward**  
- **Obstacles** (10 cells):
  - L-shape: `(2,2)`, `(2,3)`, `(2,4)`, `(3,2)`, `(4,2)`  
  - Cross: `(5,4)`, `(5,5)`, `(5,6)`, `(4,5)`, `(6,5)` → **–10 penalty**  
- **Step cost**: –1 per move  
- Agent starts at random non-obstacle, non-goal position.

---

## ⚙️ Shared Hyperparameters (All Tasks)

Use **identical settings** across all tasks for fair comparison:
- **Learning rate (α)**: 0.1 – 0.5  
- **Discount factor (γ)**: 0.9 – 0.99  
- **Epsilon**:  
  - With decay: start 0.8–1.0 → end 0.01–0.1  
  - Or fixed: 0.1–0.3  
- **Episodes**: 300–1000 (fewer allowed in teacher tasks if justified)  
- **Max steps/episode**: 50–100  
- **Random seed**: Set for reproducibility

---

## 📋 Tasks

### ✅ Task 1: Q-learning
- Implement **Q-learning** with ε-greedy action selection.
- Track per episode: total reward, steps, success (reached goal).
- **Outputs**:
  - Plot: episode rewards + 50-episode moving average + y=0 line
  - Metrics: Success Rate, Avg Reward, Avg Learning Speed
  - Save trained Q-table (for Task 3)

### ✅ Task 2: SARSA
- Implement **SARSA** using **same hyperparameters** as Task 1.
- Same metrics and plotting requirements.
- Save trained Q-table (for Task 4)

### ✅ Task 3: Q-learning with Teacher
- Load Q-table from Task 1 as **teacher**.
- New Q-learning **student** receives advice via:
  - **Availability** ∈ `[0.1, 0.3, 0.5, 0.7, 1.0]`
  - **Accuracy** ∈ `[0.1, 0.3, 0.5, 0.7, 1.0]`
- Advice logic:
  - If advice given **and correct** → follow teacher’s best action
  - If advice given **but incorrect** → random action ≠ teacher’s best
  - Else → ε-greedy as usual
- **Output**: Heatmap of **Avg Reward** vs (Availability, Accuracy)

### ✅ Task 4: SARSA with Teacher
- Same as Task 3, but using SARSA teacher (from Task 2) and SARSA student.
- Same parameter grid and heatmap output.

---

## 🔍 Analysis

### Baseline Comparison
- Side-by-side plots: Q-learning vs SARSA
  - Episode rewards (smoothed)
  - Rolling success rates (50-episode window)

### Teacher Impact Analysis
- **For Q-learning (8 marks)**: Analyze how availability/accuracy affect learning curves, convergence, and robustness.
- **For SARSA (8 marks)**: Same analysis.
- Include plots comparing baseline vs teacher-guided runs at key configurations (e.g., availability = 0.1, 0.5, 1.0).

### Teacher Effectiveness Summary
- Comparative visualisations showing:
  - How both algorithms respond to teacher guidance
  - Optimal teacher settings
  - Which algorithm benefits more from advice

---

## 🧮 Evaluation Metrics

- **Success Rate** = (Successful Episodes / Total) × 100%  
- **Average Reward** = Σ(Rewardᵢ) / N  
- **Average Learning Speed** = 1 / (Σ(Stepsᵢ) / N)

---
