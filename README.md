# churn_modelling
This is an ML model that predicts the churn rate


| Training Instance | Optimizer Used (Adam, RMSProp) | Regularizer Used (L1, L2) | Epochs | Early Stopping (Yes/No) | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------------|--------------------------------|--------------------------|--------|------------------------|-----------------|--------------|----------|---------|--------|-----------|
| Instance 1      | Adam                           | None                     | 10     | No                     | 2               | 0.001        | TBD      | TBD     | TBD    | TBD       |
| Instance 2      | Adam                           | L2                       | 20     | No                     | 3               | 0.001        | TBD      | TBD     | TBD    | TBD       |
| Instance 3      | RMSProp                        | L1 + L2                  | 30     | Yes                    | 3               | 0.0005       | TBD      | TBD     | TBD    | TBD       |
| Instance 4      | Adam                           | Dropout (0.3)            | 50     | Yes                    | 3               | 0.001        | TBD      | TBD     | TBD    | TBD       |
| Instance 5 (Optional) | Adam                   | L2 + Dropout (0.3)       | 50     | Yes                    | 4               | 0.0005       | TBD      | TBD     | TBD    | TBD       |
