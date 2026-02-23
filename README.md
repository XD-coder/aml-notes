# Applied Machine Learning (AML) — Complete Study Report
**MADE BY** - KARTIKEY MISHRA
*~thank me later*

---

## TABLE OF CONTENTS
1. [Lecture 1 — Introduction to Machine Learning](#lecture-1)
2. [Lecture 2 — Hyperparameters & Tuning](#lecture-2)
3. [Lecture 3 — Loss Functions (Theory + Implementation)](#lecture-3)
4. [Lecture 4 — Regression Loss Functions with Numericals](#lecture-4)
5. [Lecture 5 — Classification Loss Functions](#lecture-5)
6. [Lecture 6 — Sparse Categorical Loss & Triplet Loss](#lecture-6)
7. [Lecture 14 — Data Cleaning: Missing Data & Outliers](#lecture-14)
8. [Lecture 15 — Feature Scaling & Feature Encoding](#lecture-15)
9. [Lecture 16 — Dimensionality Reduction](#lecture-16)
10. [Lecture 17 — PCA (Deep Dive)](#lecture-17)
11. [Lecture 18 — Cross-Validation](#lecture-18)
12. [Lecture 19 — Handling Imbalanced Data](#lecture-19)
13. [⭐ Important Topics for Test](#important-topics)

---

## Lecture 1 — Introduction to Machine Learning {#lecture-1}

### What is Machine Learning?
ML is a branch of AI where machines **learn from data** to make predictions or decisions without being explicitly programmed. It differs from traditional programming in that rules are **learned** from examples, not written by hand.

### AI vs ML vs DL vs GenAI
| Term | Definition |
|------|-----------|
| **AI** | Broadest field — creating intelligent systems (reasoning, perception, language) |
| **ML** | Subset of AI — learning patterns from data |
| **DL** | Subset of ML — uses deep neural networks (CNNs, RNNs, Transformers) |
| **GenAI** | Specific application — generating new content (text, images, audio) |

**Hierarchy:** AI ⊃ ML ⊃ DL ⊃ GenAI

### Types of AI
- **Narrow AI** — task-specific (e.g., virtual assistants, recommendation systems)
- **General AI** — theoretical; performs any intellectual task a human can
- **Superintelligent AI** — speculative; surpasses human intelligence

### Evolution of ML
- **1950s:** Alan Turing's concept of machine intelligence; Arthur Samuel's self-learning algorithms
- **Modern era:** caused by exponential data growth + GPU/TPU computing power

### Emerging Trends
| Trend | Description |
|-------|-------------|
| **Federated Learning** | Train models across decentralized devices while preserving privacy |
| **Reinforcement Learning** | Agents learn via sequential decisions; used in robotics, gaming |
| **Explainable AI (XAI)** | Making ML decisions interpretable; addresses bias and fairness |
| **Quantum ML** | Quantum computing dramatically accelerates ML computation |
| **Generative AI** | Expansion into creative, educational, and virtual domains |

### Ethical Challenges
- Bias & Fairness
- Privacy Concerns
- Job Displacement
- Security Risks (adversarial attacks)

---

## Lecture 2 — Hyperparameters & Tuning {#lecture-2}

### What Are Hyperparameters?
Configuration settings **set before training** that govern the learning process. Unlike model parameters (learned during training), hyperparameters are fixed throughout training.

**Examples:** learning rate, number of hidden layers, max depth of decision tree, number of clusters in K-Means.

### Two Categories of Hyperparameters
1. **Model-specific** — define model architecture (e.g., number of layers, kernel type)
2. **Optimization** — influence training (e.g., optimizer, batch size, epochs)

### Hyperparameter Tuning Techniques

| Technique | Description | Pros | Cons |
|-----------|-------------|------|------|
| **Grid Search** | Tests every combination in a predefined grid | Exhaustive, guaranteed | Computationally expensive |
| **Random Search** | Randomly samples hyperparameter combinations | More efficient than grid search | May miss optimal combination |
| **Bayesian Optimization** | Uses probabilistic surrogate model; focuses on promising regions | Very efficient; intelligent exploration | Complex to implement |
| **Gradient-Based** | Applies gradient descent directly to hyperparameters | Effective for continuous params | Requires differentiable objective |
| **Evolutionary Algorithms** | Mutation, crossover, selection on a population of candidates | Good for complex spaces | Computationally intensive |
| **Manual Tuning** | Based on intuition/domain knowledge | Fast for simple models | Poor for complex models |
| **Automated (AutoML, Optuna)** | Combines multiple methods automatically | Minimal manual effort | Black-box |

### Key Hyperparameters Reference Table

| Hyperparameter | Description |
|----------------|-------------|
| **Learning Rate** | Step size per iteration toward loss minimum |
| **Batch Size** | Number of examples per forward/backward pass |
| **Number of Epochs** | How many times the full dataset is processed |
| **Hidden Layers** | Depth of neural network |
| **Neurons per Layer** | Width of neural network |
| **Activation Function** | Non-linearity per neuron (ReLU, Sigmoid, Tanh) |
| **Dropout Rate** | Fraction of neurons randomly disabled during training |
| **Regularization (L1/L2)** | Penalizes large weights to reduce overfitting |
| **Kernel Type (SVM)** | Maps data to higher-dimensional space |
| **Max Depth (Decision Tree)** | Controls tree complexity |
| **Number of Estimators** | Number of trees in Random Forest / Gradient Boosting |
| **Momentum** | Accelerates SGD convergence |

---

## Lecture 3 — Loss Functions (Theory + Implementation) {#lecture-3}

### What Are Loss Functions?
Mathematical formulas that measure the **error between predictions and actual values**. The optimization process **minimizes** the loss function to improve model performance.

### Regression Loss Functions

#### 1. Mean Squared Error (MSE)
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
- Penalizes large errors heavily (squaring)
- Sensitive to outliers
- Assumes normally distributed errors
- Differentiable — works well with gradient descent

#### 2. Mean Absolute Error (MAE)
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$
- Robust to outliers (no squaring)
- Assumes Laplace-distributed errors
- Not differentiable at zero (needs sub-gradient methods)

#### 3. Huber Loss
$$
L_\delta(y_i, \hat{y}_i) = \begin{cases}
\frac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| \leq \delta \\
\delta|y_i - \hat{y}_i| - \frac{\delta^2}{2} & \text{if } |y_i - \hat{y}_i| > \delta
\end{cases}
$$
- Combines MSE (small errors) + MAE (large errors)
- δ controls transition point
- Robust but still sensitive to small errors

### Classification Loss Functions

#### 4. Binary Cross-Entropy (Log Loss)
$$
BCE = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$
- Used for binary classification (y ∈ {0, 1})
- Penalizes confident but wrong predictions heavily

#### 5. Categorical Cross-Entropy
$$
CCE = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})
$$
- Used for multi-class classification with one-hot encoded labels
- K = number of classes

#### 6. Sparse Categorical Cross-Entropy
$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i, y_i})
$$
- Same as categorical CE but uses **integer labels** instead of one-hot encoding
- More memory efficient for many classes

#### 7. Hinge Loss (SVM Loss)
$$
L(y_i, \hat{y}_i) = \max(0, 1 - y_i \hat{y}_i)
$$
- Used in SVMs; y ∈ {-1, +1}
- Optimizes maximum-margin classification

#### 8. Triplet Loss (Metric Learning)
$$
L = \frac{1}{n} \sum_{i=1}^{n} \max(0, d(a,p) - d(a,n) + \alpha)
$$
- Used in face verification, similarity search
- Anchor (A), Positive (P, same class), Negative (N, different class)
- α = margin enforcing minimum distance between positive/negative pairs

---

## Lecture 4 — Regression Loss Functions with Sample Numericals {#lecture-4}

### MSE Numerical Example
**Given:** y = [3, -0.5, 2, 7], ŷ = [2.5, 0.0, 2, 8]
1. Squared errors: (0.5)² = 0.25, (0.5)² = 0.25, 0² = 0, (1)² = 1
2. Sum = 1.5
3. **MSE = 1.5 / 4 = 0.375**

### MAE Numerical Example
**Using same dataset:**
1. Absolute errors: 0.5, 0.5, 0, 1
2. Sum = 2.0
3. **MAE = 2.0 / 4 = 0.5**

### Huber Loss Numerical Example (δ = 1)
1. |0.5| ≤ 1 → L = ½(0.5)² = 0.125
2. |0.5| ≤ 1 → L = ½(0.5)² = 0.125
3. |0| ≤ 1 → L = 0
4. |1| = δ → L = 1·1 - 0.5 = 0.5
5. **Huber = (0.125 + 0.125 + 0 + 0.5) / 4 = 0.1875**

### Comparison Summary
| Aspect | MSE | MAE | Huber |
|--------|-----|-----|-------|
| Outlier Sensitivity | High | Low | Medium |
| Differentiability | Everywhere | Not at zero | Everywhere |
| Best For | Clean data | Noisy/outlier data | Moderate outliers |
| Error Penalty | Quadratic | Linear | Quadratic → Linear |

---

## Lecture 5 — Classification Loss Functions {#lecture-5}

### Binary Cross-Entropy — Numerical Example
**Given:** y = [1, 0, 1], ŷ = [0.9, 0.2, 0.8]
1. Loss₁ = -log(0.9) = 0.105
2. Loss₂ = -log(1 - 0.2) = -log(0.8) = 0.223
3. Loss₃ = -log(0.8) = 0.223
4. **BCE = (0.105 + 0.223 + 0.223) / 3 ≈ 0.184**

### Categorical Cross-Entropy — Numerical Example
**Given:** 3 samples with one-hot labels, predicted probs [0.7, 0.8, 0.7] for true classes
1. Loss₁ = -log(0.7) = 0.357
2. Loss₂ = -log(0.8) = 0.223
3. Loss₃ = -log(0.7) = 0.357
4. **CCE = (0.357 + 0.223 + 0.357) / 3 ≈ 0.312**

### Why Logarithmic Penalty?
- Log penalizes **confident but wrong** predictions very heavily
- Rewards accurate high-confidence predictions
- Provides probabilistic interpretation

---

## Lecture 6 — Sparse Categorical Loss & Triplet Loss {#lecture-6}

### Sparse Categorical Cross-Entropy — Numerical Example
**Given:** 3 samples, true labels [1, 2, 0], softmax predictions available
1. Extract probabilities of true class: p₁=0.7, p₂=0.4, p₃=0.8
2. Log: -0.357, -0.916, -0.223
3. **L = (-(-0.357) + -(-0.916) + -(-0.223)) / 3 = 1.496/3 ≈ 0.499**

### Triplet Loss — Numerical Example
**Given:** α = 1.0, f(A)=[0.5,0.1], f(P)=[0.6,0.2], f(N)=[0.9,0.8]
1. d(A,P) = (0.5-0.6)² + (0.1-0.2)² = 0.01 + 0.01 = **0.02**
2. d(A,N) = (0.5-0.9)² + (0.1-0.8)² = 0.16 + 0.49 = **0.65**
3. L = max(0.02 - 0.65 + 1.0, 0) = max(0.37, 0) = **0.37**

### Sparse Categorical vs Triplet Loss
| Property | Sparse Categorical | Triplet Loss |
|----------|--------------------|--------------|
| Domain | Classification | Metric learning |
| Inputs | Softmax probs + integer labels | Embeddings (A, P, N) |
| Goal | Minimize prediction error | Learn separable embeddings |
| Applications | Image/text classification | Face verification, similarity search |

---

## Lecture 14 — Data Cleaning: Missing Data & Outliers {#lecture-14}

### Why Data Cleaning?
Raw data contains errors, missing values, and outliers. Without cleaning, ML models produce biased, unreliable results.

### Types of Missing Data
| Type | Definition | Example |
|------|-----------|---------|
| **MCAR** | Missingness unrelated to any data | Random sensor failure |
| **MAR** | Missingness related to *observed* data | Older respondents skip a question |
| **MNAR** | Missingness related to the *missing* value itself | High earners not disclosing income |

### Methods to Handle Missing Data
| Method | When to Use |
|--------|-------------|
| **Listwise Deletion** | Missing data is small % |
| **Column Deletion** | Column has very high % missing |
| **Mean/Median/Mode Imputation** | Simple, works for many cases |
| **Forward/Backward Fill** | Time series data |
| **Linear Interpolation** | Sequential or smooth data |
| **KNN Imputation** | When similar neighbors exist |
| **MICE (Iterative Imputer)** | Complex relationships between features |
| **Predictive Modeling / Deep Learning** | Advanced scenarios |

### Types of Outliers
- **Univariate:** Outlier in a single feature (e.g., age = 200)
- **Multivariate:** Outlier across multiple features (e.g., low income + high spending)

### Outlier Detection Methods
| Method | Description |
|--------|-------------|
| **Z-Score** | Flags points > 3 standard deviations from mean |
| **IQR Rule** | Flags points outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] |
| **Box Plot** | Visual; outliers shown as individual points beyond whiskers |
| **Scatter Plot** | Visual; points far from trend are outliers |
| **Isolation Forest** | ML-based; uses decision tree ensembles |
| **DBSCAN** | Density-based clustering; labels low-density points as outliers |

### Box Plot Components
- **Q1 (25th percentile)** — lower box boundary
- **Q2 (Median)** — line inside box
- **Q3 (75th percentile)** — upper box boundary
- **IQR = Q3 - Q1**
- **Whiskers** = Q1 - 1.5×IQR to Q3 + 1.5×IQR
- Points outside whiskers = **outliers**

### Methods to Handle Outliers
| Method | Description |
|--------|-------------|
| **Removal** | Delete rows with outliers (Z-score based) |
| **Capping/Trimming** | Replace with threshold value |
| **Log/Sqrt/Box-Cox Transform** | Reduce impact of extreme values |
| **Imputation** | Replace with mean/median |

---

## Lecture 15 — Feature Scaling & Feature Encoding {#lecture-15}

### Why Feature Scaling?
Algorithms using distance measures (KNN, SVM, K-Means) are dominated by features with larger scales. Scaling ensures **equal contribution** from all features.

### Feature Scaling Methods

| Method | Formula | Best For |
|--------|---------|---------|
| **Min-Max (Normalization)** | $X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$ → [0,1] | Bounded range, preserve distribution |
| **Standardization (Z-score)** | $X_{scaled} = \frac{X - \mu}{\sigma}$ → mean=0, std=1 | Gaussian-distributed data |
| **Robust Scaling** | $X_{scaled} = \frac{X - \text{Median}}{\text{IQR}}$ | Data with outliers |
| **Max Abs Scaling** | $X_{scaled} = \frac{X}{|X_{max}|}$ | Sparse data, no shifting needed |

### Feature Encoding Methods

| Method | Description | Best For |
|--------|-------------|---------|
| **Label Encoding** | Each category → unique integer | Ordinal data |
| **One-Hot Encoding** | Binary column per category | Nominal data (no order) |
| **Ordinal Encoding** | Assigns integers based on order | Ordered categories |
| **Binary Encoding** | Category → binary digits | Many categories, reduce dimensionality |
| **Frequency Encoding** | Replace with category frequency | Frequency is informative |
| **Target Encoding** | Replace with mean of target variable per category | Classification (with care for overfitting) |

### Key Insights
- **Label Encoding** can falsely imply ordinality for nominal data
- **One-Hot Encoding** increases dimensionality (bad for many categories)
- **Target Encoding** is powerful but prone to overfitting
- **Robust Scaling** is the best scaler when outliers are present

---

## Lecture 16 — Dimensionality Reduction {#lecture-16}

### The Problem: Curse of Dimensionality
As the number of features grows, the **dimension space grows exponentially** (e.g., 4 points × 3 features = 4³ = 64 space). This causes:
- Sparse data — harder to find patterns
- Increased multicollinearity
- Slower training
- Degraded model accuracy

**Hughes Phenomenon:** Predictive power increases up to a certain dimension, then decreases.

### Why Reduce Dimensions?
- Reduces **overfitting**
- Faster **training**
- Better **visualization** (2D/3D)
- Removes **redundant/irrelevant features**

### Techniques

#### Feature Selection (keep original features)
| Method | Description |
|--------|-------------|
| **Filter Methods** | Statistical scoring (correlation, mutual information) |
| **Wrapper Methods** | Evaluate feature subsets using model (e.g., RFE) |
| **Embedded Methods** | Selection built into training (e.g., LASSO) |

#### Feature Extraction (create new lower-dimensional features)
| Technique | Type | Use Case |
|-----------|------|---------|
| **PCA** | Linear, unsupervised | Image compression, visualization |
| **LDA** | Linear, supervised | Classification, maximizes class separability |
| **t-SNE** | Non-linear | Data visualization only |
| **Autoencoders** | Non-linear (Deep Learning) | Complex feature learning |
| **SVD** | Linear | Text/recommendation systems |

### When to Use Which
| Scenario | Recommended Technique |
|----------|----------------------|
| Interpretation & Speed | Feature Selection |
| Linear data | PCA or LDA |
| Non-linear data | t-SNE or Autoencoders |
| Text data | SVD |

---

## Lecture 17 — Principal Component Analysis (PCA) {#lecture-17}

### What is PCA?
A **linear, unsupervised** dimensionality reduction technique that projects data onto directions of **maximum variance**. Finds new uncorrelated axes (Principal Components).

### Key Mathematical Concepts

#### Variance
$$
\text{Var}(X) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2
$$

#### Covariance
$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
$$

#### Covariance Matrix (d features → d×d matrix)
- **Symmetric:** C = Cᵀ
- **Positive Semi-Definite:** all eigenvalues ≥ 0
- Diagonal = variances; off-diagonal = covariances

#### Eigenvalues & Eigenvectors
$$
Av = \lambda v \quad \Rightarrow \quad \det(C - \lambda I) = 0
$$
- **Larger eigenvalue** = more variance in that direction = more important principal component
- **Smaller eigenvalue** = less important direction

### PCA Steps (Complete Algorithm)
1. **Standardize** the data: $Z = \frac{X - \mu}{\sigma}$
2. **Compute Covariance Matrix:** $C = \frac{1}{n} Z^T Z$
3. **Compute Eigenvalues & Eigenvectors:** solve $\det(C - \lambda I) = 0$
4. **Sort** eigenvectors by descending eigenvalues
5. **Select top k** principal components
6. **Transform data:** $X_{new} = X \cdot W$

### Applications
- Image Compression
- Feature Extraction
- Data Visualization
- Noise Reduction

### Advantages
- Removes redundancy and correlations
- Improves computational efficiency
- Enhances interpretability

### Limitations
- Assumes **linear relationships**
- **Sensitive to outliers**
- Transformed features are hard to interpret

### PCA Numerical Example (condensed)
Given 4×3 data matrix, after mean-centering and computing covariance matrix:
- Eigenvalues: λ₁ = 3, λ₂ = 1.67, λ₃ = 0.34
- Selecting top 2 PCs (λ₁ and λ₂) for 2D projection
- Final reduced matrix preserves maximum variance

---

## Lecture 18 — Cross-Validation {#lecture-18}

### Why Cross-Validation?
A single train-test split can be misleading (lucky split). Cross-validation provides an **unbiased estimate** of model performance by training/testing on multiple splits.

### Problems Addressed
- **Overfitting** — model memorizes training data
- **Underfitting** — model fails to learn patterns
- **Data Leakage** — model sees test data during training
- **Hyperparameter Tuning** — reliable evaluation across different configurations

### Cross-Validation Techniques

#### 1. Hold-Out Validation
- Split: typically 70-80% train / 20-30% test
- **Use when:** large datasets, time-intensive training (deep learning)
- **Limitation:** results depend heavily on how the split is made

#### 2. K-Fold Cross-Validation
- Divide data into K equal folds; train on K-1, test on 1; repeat K times; **average the scores**
- **Use when:** limited data, hyperparameter tuning
- **Limitation:** computationally expensive for large K

#### 3. Stratified K-Fold
- Same as K-Fold but **maintains class proportion** in each fold
- **Use when:** imbalanced datasets (fraud detection)
- **Limitation:** extra compute for maintaining class distribution

#### 4. Leave-One-Out Cross-Validation (LOOCV)
- Each single data point is the test set; N iterations for N points
- **Use when:** very small datasets (medical/biological research)
- **Limitation:** very computationally expensive; high variance

#### 5. Leave-P-Out
- P points left out each time (generalization of LOOCV)
- **Use when:** small datasets where LOOCV is infeasible
- **Limitation:** expensive for large P

#### 6. Time Series Cross-Validation (Rolling Window)
- Training set **expands incrementally**; tested on future data
- **Use when:** stock prediction, weather forecasting, sales forecasting
- **Limitation:** cannot shuffle data; computationally expensive

---

## Lecture 19 — Handling Imbalanced Data {#lecture-19}

### The Problem
When one class dominates (e.g., 99% legitimate / 1% fraudulent), a naive classifier predicting the majority class achieves **99% accuracy** but **0% fraud recall** — useless in practice.

### Why Standard Accuracy Fails
- Biased predictions favoring majority class
- Poor recall and precision for minority class
- Need alternative metrics: **Precision, Recall, F1-Score, ROC-AUC**

### Oversampling Techniques (increase minority class)

| Technique | Description |
|-----------|-------------|
| **Random Oversampling** | Duplicates random minority class samples |
| **SMOTE** | Generates **synthetic** minority samples by interpolating between existing ones; most popular |
| **ADASYN** | Like SMOTE, but focuses on generating samples near **decision boundaries** |

### Undersampling Techniques (reduce majority class)

| Technique | Description |
|-----------|-------------|
| **Random Undersampling** | Randomly removes majority class samples |
| **NearMiss** | Selects majority samples based on distance to minority samples |
| **Tomek Links** | Removes majority class samples that are closest to minority samples |

### Hybrid Approach
**SMOTE + Tomek Links** (SMOTETomek) — balances data while reducing noise from borderline samples

### Alternative Methods
| Method | Description |
|--------|-------------|
| **Cost-Sensitive Learning** | Higher misclassification cost for minority class |
| **Anomaly Detection** | Treat minority class as anomalies |
| **Ensemble Methods** | Boosting (AdaBoost, XGBoost) focuses on hard-to-classify samples |
| **Threshold Moving** | Adjust decision threshold to improve minority recall |

### Performance Metrics for Imbalanced Data
$$
\text{Precision} = \frac{TP}{TP + FP}
$$
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
- **ROC-AUC** — how well model distinguishes between classes

### Best Practices
- **Oversampling** → use when data is limited (preserves minority info)
- **Undersampling** → use when dataset is large, removing redundant majority is safe
- **Hybrid** → best of both worlds
- Evaluate with **precision-recall curves**, not accuracy

---

## ⭐ IMPORTANT TOPICS FOR TEST {#important-topics}

> Based on depth of coverage, presence of numerical examples, and conceptual complexity.

---

### 🔴 TIER 1 — MUST KNOW (Very Likely to Appear)

#### 1. Loss Functions (Formulas + Numericals)
- **MSE, MAE, Huber Loss** — know formulas and be able to compute step-by-step
- **Binary Cross-Entropy** — formula + numerical (y=[1,0,1], ŷ=[0.9,0.2,0.8] → 0.184)
- **Categorical Cross-Entropy** — formula + numerical
- **Sparse Categorical Cross-Entropy** — formula (log of predicted probability of true class)
- **Hinge Loss** — formula max(0, 1 - y·ŷ)
- **Triplet Loss** — formula + numerical (anchor, positive, negative distances)
- **Comparison table:** MSE vs MAE vs Huber (sensitivity to outliers, differentiability)

#### 2. PCA (Full Algorithm)
- Variance and Covariance formulas
- Covariance matrix properties (symmetric, positive semi-definite)
- Eigenvalue equation: det(C - λI) = 0
- Full 5-step algorithm: Standardize → Covariance Matrix → Eigenvalues/Eigenvectors → Rank → Transform
- Know: larger eigenvalue = more important component
- Advantages and Limitations

#### 3. Feature Scaling Methods
- Min-Max formula and range [0,1]
- Standardization formula (mean=0, std=1)
- Robust Scaling formula (uses IQR)
- Max Abs Scaling formula
- **When to use which** (especially Robust Scaling for outlier-heavy data)

#### 4. Cross-Validation Techniques
- K-Fold: know how it works (train K-1, test 1, repeat K times, average)
- Stratified K-Fold: for imbalanced data
- LOOCV: for small datasets; expensive
- Time Series CV: rolling window; no shuffling
- Know when to apply each

#### 5. Handling Missing Data
- MCAR vs MAR vs MNAR definitions + examples
- Imputation methods: Mean/Median/Mode, KNN, MICE
- Forward/Backward fill for time series
- Know **when to use each method**

---

### 🟡 TIER 2 — IMPORTANT (Likely to Appear)

#### 6. Outlier Detection & Handling
- Z-Score method (threshold: |z| > 3)
- IQR Rule: lower = Q1 - 1.5×IQR, upper = Q3 + 1.5×IQR
- Box plot anatomy: Q1, Q2 (median), Q3, IQR, whiskers, outliers
- Methods: Removal, Capping, Log Transformation
- ML methods: Isolation Forest, DBSCAN

#### 7. Feature Encoding
- Label Encoding vs One-Hot Encoding vs Ordinal Encoding — differences and when to use
- Label Encoding problem: implies false order
- One-Hot Encoding problem: high dimensionality
- Target Encoding risk: overfitting

#### 8. Imbalanced Data Handling
- SMOTE algorithm concept (interpolates between minority samples)
- Random Oversampling vs SMOTE vs ADASYN
- Tomek Links concept
- Metrics to use: F1-Score, Precision, Recall, ROC-AUC (NOT accuracy)

#### 9. Dimensionality Reduction Overview
- Curse of Dimensionality explanation
- Hughes Phenomenon
- Feature Selection vs Feature Extraction
- PCA vs LDA vs t-SNE (linear/supervised/non-linear)
- Filter vs Wrapper vs Embedded selection methods

#### 10. Hyperparameter Tuning
- Grid Search vs Random Search vs Bayesian Optimization
- Know key hyperparameters: learning rate, batch size, dropout rate, regularization
- Difference between hyperparameter and model parameter

---

### 🟢 TIER 3 — GOOD TO KNOW (May Appear)

#### 11. Introduction to ML (Lecture 1)
- AI ⊃ ML ⊃ DL ⊃ GenAI hierarchy
- Types of AI: Narrow, General, Superintelligent
- Emerging trends: Federated Learning, XAI, RL

#### 12. Loss Function Implementation
- Python implementations (numpy-based) of MSE, MAE, Huber, BCE
- TensorFlow/PyTorch loss function usage

---

### 📋 FORMULA QUICK REFERENCE CARD

| Loss Function | Formula | Type |
|---------------|---------|------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Regression |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Regression |
| Huber (small err) | $\frac{1}{2}(y_i - \hat{y}_i)^2$ | Regression |
| Huber (large err) | $\delta|y_i-\hat{y}_i| - \frac{\delta^2}{2}$ | Regression |
| Binary CE | $-\frac{1}{n}\sum[y\log\hat{y}+(1-y)\log(1-\hat{y})]$ | Binary Classification |
| Categorical CE | $-\frac{1}{n}\sum_i\sum_k y_{ik}\log\hat{y}_{ik}$ | Multi-class |
| Sparse CE | $-\frac{1}{N}\sum\log(p_{i,y_i})$ | Multi-class (integer labels) |
| Hinge | $\max(0, 1 - y\hat{y})$ | SVM/Binary |
| Triplet | $\max(0, d(a,p) - d(a,n) + \alpha)$ | Metric Learning |

| Scaling Method | Formula |
|----------------|---------|
| Min-Max | $\frac{X - X_{min}}{X_{max} - X_{min}}$ |
| Standardization | $\frac{X - \mu}{\sigma}$ |
| Robust | $\frac{X - \text{Median}}{\text{IQR}}$ |
| Max Abs | $\frac{X}{|X_{max}|}$ |

| Metric | Formula |
|--------|---------|
| Precision | $\frac{TP}{TP+FP}$ |
| Recall | $\frac{TP}{TP+FN}$ |
| F1-Score | $2 \times \frac{P \times R}{P + R}$ |
| IQR Outlier Lower | $Q_1 - 1.5 \times IQR$ |
| IQR Outlier Upper | $Q_3 + 1.5 \times IQR$ |

---

*End of Report — Good luck on your test!*
