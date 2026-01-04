# 5 Simple Machine Learning Projects for Practice

Here are 5 progressively challenging projects designed to give you hands-on experience with different models and scenarios.

---

### Project 1: Predicting Medical Insurance Costs

* **Objective:** To predict the insurance cost for an individual based on their characteristics. This is a perfect entry point into machine learning.
* **Core ML Concept:** **Linear Regression** (for predicting a continuous numerical value).
* **Suggested Dataset:** [Medical Cost Personal Datasets on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance).

#### Step-by-Step Guide:

1.  **Exploratory Data Analysis (EDA) & Preprocessing:**
    * Load the dataset and use `.info()` and `.describe()` for a quick overview.
    * Visualize the distribution of the target variable, `charges`.
    * Explore relationships between features (like `age`, `bmi`, `smoker`) and `charges` using scatter plots and box plots.
    * Check for correlations between numerical features using a heatmap.
    * Handle categorical features (like `sex`, `smoker`, `region`) by converting them to a numerical format (e.g., One-Hot Encoding).

2.  **Model Building:**
    * Separate your features (X) from the target variable (y, which is `charges`).
    * Split your data into a training set and a testing set (80/20 split).
    * Initialize and train a `LinearRegression` model from Scikit-learn.

3.  **Model Evaluation:**
    * Make predictions on your testing set.
    * Evaluate model performance using regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared ($R^2$) score.

4.  **Model Improvement:**
    * Analyze feature importance. Can you get good results with fewer features?
    * Try creating new polynomial features (e.g., `age^2`) to capture non-linear patterns and see if it improves the $R^2$ score.

---

### Project 2: Titanic Survival Prediction

* **Objective:** To predict whether a passenger on the Titanic survived or not.
* **Core ML Concept:** **Logistic Regression** (for binary classification).
* **Suggested Dataset:** [Titanic - Machine Learning from Disaster on Kaggle](https://www.kaggle.com/c/titanic/data).

#### Step-by-Step Guide:

1.  **EDA & Preprocessing:**
    * Load the training data and identify columns with missing values (`Age`, `Cabin`).
    * Develop a strategy to handle missing values (e.g., fill `Age` with the median, drop `Cabin` or create a new category).
    * Convert categorical features (`Sex`, `Embarked`) into numerical format.
    * Engineer new features, like `FamilySize` from `SibSp` and `Parch`.

2.  **Model Building:**
    * Define your features (X) and target (y, which is `Survived`).
    * Split the data into training and testing sets.
    * Scale numerical features (`Age`, `Fare`) using `StandardScaler`.
    * Initialize and train a `LogisticRegression` model.

3.  **Model Evaluation:**
    * Evaluate using classification metrics: Confusion Matrix, Accuracy, Precision, Recall, and F1-Score.

4.  **Model Improvement:**
    * Analyze the confusion matrix. Is your model biased?
    * Try different strategies for handling missing data.
    * Adjust the regularization parameter `C` in the `LogisticRegression` model to see if performance improves.

---

### Project 3: Classifying Penguin Species

* **Objective:** To classify penguins into one of three species based on their physical measurements.
* **Core ML Concept:** **Random Forest Classifier** (an ensemble model for multi-class classification).
* **Suggested Dataset:** [Palmer Penguins Dataset on Kaggle](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data).

#### Step-by-Step Guide:

1.  **EDA & Preprocessing:**
    * Load the data and handle any missing values.
    * Use a pair plot from Seaborn to visualize the differences between the three species, color-coded by species.
    * Identify which features seem best at separating the species.

2.  **Model Building:**
    * Define your features (X) and target (y, which is `species`). Handle categorical columns.
    * Split your data into training and testing sets.
    * Initialize and train a `RandomForestClassifier` model.

3.  **Model Evaluation:**
    * Generate a classification report to see precision, recall, and F1-score for each species.
    * Visualize the confusion matrix. Which species is the model most confused about?

4.  **Model Improvement:**
    * Check the `feature_importances_` attribute of your trained model. Which features were most important?
    * Tune hyperparameters like `n_estimators` (number of trees) and `max_depth` (depth of trees).

---

### Project 4: Customer Segmentation for a Retail Store

* **Objective:** To group customers into distinct segments based on their spending habits.
* **Core ML Concept:** **K-Means Clustering** (an unsupervised learning algorithm).
* **Suggested Dataset:** [Mall Customer Segmentation Data on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

#### Step-by-Step Guide:

1.  **EDA & Preprocessing:**
    * Focus on the `Annual Income (k$)` and `Spending Score (1-100)` columns.
    * Create a scatter plot of `Spending Score` vs. `Annual Income`. Can you visually identify any clusters?

2.  **Model Building:**
    * Use the "Elbow Method" to find the optimal number of clusters (`k`).
        * Run K-Means for a range of `k` values (e.g., 1 to 10).
        * Plot `k` vs. the Within-Cluster Sum of Squares (WCSS or `inertia_`). The "elbow" point is your optimal `k`.
    * Train the `KMeans` model with your chosen `k`.

3.  **Model Evaluation & Interpretation:**
    * Get the cluster labels for each data point.
    * Create a scatter plot of the data, but color the points according to their assigned cluster label.
    * Describe each cluster (e.g., "High Income, Low Spending").

4.  **Model Improvement:**
    * Try including the `Age` feature (remember to scale all features first).
    * Does adding another dimension create more meaningful clusters?

---

### Project 5: Heart Disease Prediction

* **Objective:** To predict the presence of heart disease in a patient.
* **Core ML Concept:** **Gradient Boosting (XGBoost/LightGBM)** or **Support Vector Machines (SVM)**.
* **Suggested Dataset:** [Heart Disease UCI on Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

#### Step-by-Step Guide:

1.  **EDA & Preprocessing:**
    * Understand the data dictionary to know what each column means.
    * Visualize how each feature relates to the `target` (presence of heart disease).

2.  **Model Building:**
    * Prepare data (handle categoricals, scale numericals).
    * Split into training and testing sets.
    * Train a powerful model like `XGBClassifier`.

3.  **Model Evaluation:**
    * In a medical context, a False Negative is very costly. Focus on **Recall** to minimize missed cases.
    * Also, evaluate the **ROC Curve** and the **Area Under the Curve (AUC)** score.

4.  **Model Improvement:**
    * Use `GridSearchCV` or `RandomizedSearchCV` to tune the model's hyperparameters to maximize your Recall or AUC score.
    * Compare the performance of XGBoost with a simpler model like Logistic Regression.

---

### Your Capstone Project Idea

* **Project:** **Customer Churn Prediction**
* **Dataset:** [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **The Challenge:**
    1.  Perform a deep EDA to understand what factors lead to customer churn.
    2.  Preprocess the data thoroughly.
    3.  Build and evaluate at least three different classification models (e.g., Logistic Regression, Random Forest, XGBoost).
    4.  Compare their performance using multiple metrics (Accuracy, Precision, Recall, F1-Score, AUC).
    5.  Choose your "best" model and justify your choice based on the business problem.
    6.  Use your best model to identify the key factors that predict churn.