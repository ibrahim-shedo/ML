# Machine Learning Models Project

This project explores implementing and analyzing four different machine learning models on a selected dataset. Each model serves a specific type of machine learning task, ranging from regression to classification, and utilizes various algorithms to derive insights and predictions.

## Project Structure

The project covers the following machine learning models:

1. **Linear Regression using Ordinary Least Squares (OLS)**
   - Implements linear regression by minimizing the sum of squared residuals between observed and predicted values.
   - Aims to provide an interpretable linear model by fitting a line to the dataset.

2. **Decision Tree Classifier**
   - A classification model that uses a tree-like structure of decisions and outcomes.
   - Splits data based on feature values to predict the target class efficiently.

3. **Logistic Regression using Maximum Likelihood Estimation (MLE)**
   - A probabilistic classification model used for binary classification.
   - Implements logistic regression by maximizing the likelihood of the observed data.

4. **Bayesian Linear Regression**
   - An extension of linear regression that incorporates Bayesian inference, allowing for uncertainty quantification in the model.
   - Useful for predicting distributions rather than point estimates, making it robust for small datasets.

---

## Project Setup

To run the project, you'll need to install the following dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
Usage
Load and Explore the Dataset: Use exploratory data analysis to understand the data, handle missing values, and preprocess features.
Model Training: Run each model in sequence by executing the respective scripts. Each script includes model training and evaluation steps.
Evaluate Models: After training, each model outputs performance metrics:
Linear Regression: Mean Squared Error (MSE), R-squared
Decision Tree Classifier: Accuracy, Precision, Recall
Logistic Regression: Classification Accuracy, Confusion Matrix, Precision-Recall Curve
Bayesian Linear Regression: Posterior distributions of the model parameters and predictive intervals
Results
The results from each model are presented in the results/ directory. Each model’s effectiveness is evaluated based on the selected metrics. Visualizations are included for easy interpretation.

File Structure
data/: Contains the dataset used for the models.
src/: Includes scripts for each model:
linear_regression_ols.py
decision_tree_classifier.py
logistic_regression_mle.py
bayesian_linear_regression.py
results/: Stores output results and evaluation metrics for each model.
Contributing
Contributions are welcome! Feel free to submit a pull request if you have suggestions or improvements.

License
This project is licensed under the MIT License.
