# churn_modelling
This is an ML model that predicts the churn rate.

| Training Instance | Optimizer Used (Adam, RMSProp) | Regularizer Used (L1, L2) | Epochs | Early Stopping (Yes/No) | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------------|--------------------------------|--------------------------|--------|------------------------|-----------------|--------------|----------|---------|--------|-----------|
| Logistic Regression | N/A                            | None                     | N/A    | No                     | N/A             | N/A          | 0.5584   | TBD     | TBD    | TBD       |
| Basic Neural Network | Adam                           | None                     | 10     | No                     | 2               | 0.001        | 0.7023   | TBD     | TBD    | TBD       |
| Optimized Neural Network | Adam                     | Dropout (0.3)            | 50     | Yes                    | 3               | 0.001        | 0.6856   | TBD     | TBD    | TBD       |
