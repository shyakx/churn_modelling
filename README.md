# Credit Risk Prediction - Machine Learning Models  

## **Project Overview**  
This project aims to build a **Credit Risk Prediction** model that classifies loan applicants as **low risk or high risk** based on their financial data.  
The dataset used contains various attributes, including **credit history, income, debt, and loan details**.  
The goal is to optimize **machine learning models using different techniques** to improve accuracy, precision, and recall while reducing overfitting.  

---

## **Dataset Description**  
The dataset includes the following key features:  

- **Person attributes** (Age, Income, Home Ownership)  
- **Credit attributes** (Debt-to-Income Ratio, Credit Bureau Data)  
- **Loan attributes** (Loan Purpose, Loan Grade, Interest Rate)  
- **Target Variable:** `loan_status` (Indicates if the applicant defaulted or not)  

---

## **Project Structure**  
Credit_Risk_Prediction/ │── notebook.ipynb # Jupyter Notebook with ML & NN models │── README.md # Project Documentation │── saved_models/ # Folder for trained models │ ├── basic_neural_network.h5 │ ├── optimized_neural_network.h5 │ ├── best_model.h5 │ ├── logistic_regression.pkl │ ├── xgboost_model.pkl


---

## **Model Implementations**  
This project explores multiple **machine learning and deep learning** approaches:  

### **Classical ML Models**  
✅ **Logistic Regression** (Hyperparameter-tuned)  
✅ **Support Vector Machine (SVM)** (Hyperparameter-tuned)  
✅ **XGBoost Classifier**  

### **Neural Network Models**  
✅ **Basic Neural Network** (No optimizations)  
✅ **Optimized Neural Network** (Various optimization techniques)  

### **Optimization & Regularization**  
🔹 **Different Optimizers:** Adam, RMSprop, SGD  
🔹 **Regularization Techniques:** L1, L2, Dropout  
🔹 **Early Stopping** to prevent overfitting  

---

## **Performance Comparison Table**  
The table below summarizes the results of the **Optimized Neural Network** with different optimization techniques:  

| Training Instance | Optimizer Used | Regularizer Used | Epochs | Early Stopping | Number of Layers | Learning Rate | Accuracy | F1-score | Recall | Precision |
|------------------|---------------|------------------|--------|---------------|--------------|--------------|---------|---------|--------|----------|
| 1 | Default | None | 10 | No | 3 | Default | 0.85 | 0.83 | 0.80 | 0.86 |
| 2 | Adam | L2 | 50 | Yes | 4 | 0.001 | 0.89 | 0.87 | 0.85 | 0.88 |
| 3 | RMSprop | L1 | 50 | Yes | 4 | 0.001 | 0.88 | 0.86 | 0.84 | 0.87 |
| 4 | **Adam** | **L2** | **50** | **Yes** | **5** | **0.0005** | **0.90** | **0.89** | **0.88** | **0.90** |
| 5 | SGD | Dropout | 30 | No | 6 | 0.01 | 0.86 | 0.84 | 0.82 | 0.85 |

---

## **Findings & Summary**  
✅ **Instance 4 (Adam + L2 Regularization, 50 epochs, 5 layers, LR = 0.0005) had the highest accuracy (90%)**  
✅ **Instance 1 (Default settings) performed the worst**, showing the importance of optimizations  
✅ **Instance 5 (SGD + Dropout) showed that a higher learning rate (0.01) and dropout might not be as effective as L2 regularization**  

---

## **Model Comparison: ML vs Neural Networks**  

| Model | Accuracy | F1-Score | Precision | Recall |
|--------|------------|------------|------------|------------|
| **Logistic Regression** | 0.81 | 0.79 | 0.80 | 0.78 |
| **SVM (RBF Kernel)** | 0.83 | 0.81 | 0.82 | 0.80 |
| **XGBoost** | 0.88 | 0.86 | 0.85 | 0.87 |
| **Optimized Neural Network (Best Model)** | **0.90** | **0.89** | **0.90** | **0.88** |

✅ **Neural Networks performed better than classical ML models**  
✅ **XGBoost was the best-performing classical ML model**  
✅ **The best neural network (Adam + L2 Regularization) achieved the highest accuracy**  

---

## **Confusion Matrix for Best Model**  
A **confusion matrix** was generated to analyze misclassifications in the best-performing model:  

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

# Show the plot
plt.title("Confusion Matrix for Best Model")
plt.show()
```
How to Run This Project

1️⃣ Clone the Repository:
```
git clone https://github.com/yourusername/Credit_Risk_Prediction.git
cd Credit_Risk_Prediction
```
2️⃣ Install Dependencies:
```
pip install -r requirements.txt
```
3️⃣ Run the Jupyter Notebook:
```
jupyter notebook
```
4️⃣ Load the Best Model for Predictions:
```
from tensorflow.keras.models import load_model
```
# Load best saved model
```
model = load_model("saved_models/best_model.h5")
```
# Make predictions
```
predictions = model.predict(X_test)
```

🎥 Video Submission
Here is the link:

📌 Conclusion
This project successfully applied Machine Learning and Neural Networks to predict credit risk.

🔹 Neural Networks with Adam & L2 regularization performed the best.
🔹 XGBoost was the strongest traditional ML model.
🔹 Optimization techniques like dropout, L1/L2 regularization, and different optimizers significantly impacted performance.

📌 Next Steps:
🚀 Test model performance on a real-world dataset
🚀 Deploy the best model as an API for credit risk assessment

📌 GitHub Repository Link
🔗 GitHub Repository
