# DST Week 08 - PyCaret & MLflow
## fix wk08.html 
[html/wk08.html](html/wk08.html)
แก้หน้าweb page ขาว: แก้ไขบรรทัดที่ 709 แก้ไขcurrentSide == 20 ตอนแรกแก้ไขเป็นcurrentSide == 28


## Dependencies

| Package | Version |
|---------|---------|
| PyCaret | 3.1.0 |
| MLflow | 3.8.1 |
| pandas | 1.5.3 |
| numpy | 1.23.5 |
| scikit-learn | 1.2.2 |
| joblib | 1.3.2 |

---

## PyCaret Model Comparison Results

### Output จากการรัน `pycaretflow.py`

```
                                    Model  Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC  TT (Sec)
lr                    Logistic Regression    0.7689  0.8047  0.5602  0.7208  0.6279  0.4641  0.4736     0.424
ridge                    Ridge Classifier    0.7670  0.0000  0.5497  0.7235  0.6221  0.4581  0.4690     0.006
lda          Linear Discriminant Analysis    0.7670  0.8055  0.5550  0.7202  0.6243  0.4594  0.4695     0.005
rf               Random Forest Classifier    0.7485  0.7911  0.5284  0.6811  0.5924  0.4150  0.4238     0.034
nb                            Naive Bayes    0.7427  0.7955  0.5702  0.6543  0.6043  0.4156  0.4215     0.005
gbc          Gradient Boosting Classifier    0.7373  0.7918  0.5550  0.6445  0.5931  0.4013  0.4059     0.020
ada                  Ada Boost Classifier    0.7372  0.7799  0.5275  0.6585  0.5796  0.3926  0.4017     0.016
et                 Extra Trees Classifier    0.7299  0.7788  0.4965  0.6516  0.5596  0.3706  0.3802     0.033
qda       Quadratic Discriminant Analysis    0.7282  0.7894  0.5281  0.6558  0.5736  0.3785  0.3910     0.005
lightgbm  Light Gradient Boosting Machine    0.7133  0.7645  0.5398  0.6036  0.5650  0.3534  0.3580     0.128
knn                K Neighbors Classifier    0.7001  0.7164  0.5020  0.5982  0.5413  0.3209  0.3271     0.213
dt               Decision Tree Classifier    0.6928  0.6512  0.5137  0.5636  0.5328  0.3070  0.3098     0.005
dummy                    Dummy Classifier    0.6518  0.5000  0.0000  0.0000  0.0000  0.0000  0.0000     0.004
svm                   SVM - Linear Kernel    0.5954  0.0000  0.3395  0.4090  0.2671  0.0720  0.0912     0.006
```

### Best Model: Logistic Regression

```python
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=123, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
```

### Top 5 Models Summary

| Rank | Model | Accuracy | AUC | F1 Score |
|:----:|-------|:--------:|:---:|:--------:|
| 1 | Logistic Regression | 76.89% | 0.8047 | 0.6279 |
| 2 | Ridge Classifier | 76.70% | - | 0.6221 |
| 3 | Linear Discriminant Analysis | 76.70% | 0.8055 | 0.6243 |
| 4 | Random Forest | 74.85% | 0.7911 | 0.5924 |
| 5 | Naive Bayes | 74.27% | 0.7955 | 0.6043 |

---

## Known Issues

### LightGBM Warning
```
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
```
สาเหตุ: ข้อมูลมีขนาดเล็ก ทำให้ model ไม่สามารถ split ได้เพิ่มเติม (ไม่ส่งผลต่อการทำงาน)

### joblib Compatibility
ต้องใช้ `joblib<1.5` เพื่อให้เข้ากันได้กับ PyCaret 3.1.0

---

## Project Structure

```
DST_week08/
├── .venv/                  # Virtual environment
├── html/
│   └── wk08.html          # Week 08 HTML file
├── .gitignore
├── .python-version        # Python 3.10
├── pycaretflow.py         # PyCaret AutoML script
├── pyproject.toml         # Project dependencies
├── uv.lock                # Lock file
└── README.md              # This file
```

---

