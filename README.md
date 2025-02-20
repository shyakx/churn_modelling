# churn_modelling
This is an ML model that predicts the churn rate.

| Training Instance | Optimizer Used (Adam, RMSProp) | Regularizer Used (L1, L2) | Epochs | Early Stopping (Yes/No) | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------------|--------------------------------|--------------------------|--------|------------------------|-----------------|--------------|----------|---------|--------|-----------|
| Logistic Regression | N/A                            | None                     | N/A    | No                     | N/A             | N/A          | 0.5584   | FOR 0-0.89 1-0.23     | FOR 0-0.97 1-0.14    | FOR 0-0.82 1-0.59       |
| Basic Neural Network | Adam                           | None                     | 10     | No                     | 2               | 0.001        | 0.7023   | FOR 0-0.91 1-0.56     | FOR 0-0.96 1-0.44    | FOR 0-0.87 1-0.75       |
| Optimized Neural Network | Adam                     | Dropout (0.3)            | 50     | Yes                    | 3               | 0.001        | 0.6856   | FOR 0-0.92 1-0.53     | FOR 0-0.98 1-0.39    | FOR 0-0.86 1-0.82       |
