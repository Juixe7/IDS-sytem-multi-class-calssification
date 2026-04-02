# Classification Reports

## LR

- Accuracy: 0.6860
- Macro F1: 0.4307
- Weighted F1: 0.7087
- Macro FAR: 0.0372
- ROC-AUC OvR: 0.9194

```
                precision    recall  f1-score   support

      analysis       0.16      0.50      0.24      1594
      backdoor       0.16      0.12      0.13      1535
           dos       0.22      0.11      0.15      3806
      exploits       0.80      0.58      0.67     19844
       fuzzers       0.56      0.51      0.54     16150
       generic       0.86      0.81      0.83      4181
        normal       0.91      0.85      0.88     51890
reconnaissance       0.42      0.57      0.48      7522
     shellcode       0.21      0.79      0.33      1091
         worms       0.02      0.76      0.04       127

      accuracy                           0.69    107740
     macro avg       0.43      0.56      0.43    107740
  weighted avg       0.75      0.69      0.71    107740

```

## RF

- Accuracy: 0.7448
- Macro F1: 0.5032
- Weighted F1: 0.7171
- Macro FAR: 0.0385
- ROC-AUC OvR: 0.9467

```
                precision    recall  f1-score   support

      analysis       0.16      0.32      0.21      1594
      backdoor       0.23      0.18      0.20      1535
           dos       0.29      0.29      0.29      3806
      exploits       0.84      0.76      0.80     19844
       fuzzers       0.72      0.18      0.29     16150
       generic       0.86      0.83      0.84      4181
        normal       0.78      0.98      0.87     51890
reconnaissance       0.79      0.71      0.75      7522
     shellcode       0.50      0.74      0.59      1091
         worms       0.11      0.48      0.18       127

      accuracy                           0.74    107740
     macro avg       0.53      0.55      0.50    107740
  weighted avg       0.75      0.74      0.72    107740

```

## XGB

- Accuracy: 0.7581
- Macro F1: 0.5307
- Weighted F1: 0.7434
- Macro FAR: 0.0351
- ROC-AUC OvR: 0.9586

```
                precision    recall  f1-score   support

      analysis       0.16      0.30      0.21      1594
      backdoor       0.21      0.25      0.23      1535
           dos       0.26      0.29      0.27      3806
      exploits       0.82      0.78      0.80     19844
       fuzzers       0.72      0.28      0.40     16150
       generic       0.93      0.83      0.88      4181
        normal       0.82      0.96      0.88     51890
reconnaissance       0.83      0.74      0.78      7522
     shellcode       0.45      0.76      0.57      1091
         worms       0.18      0.63      0.28       127

      accuracy                           0.76    107740
     macro avg       0.54      0.58      0.53    107740
  weighted avg       0.77      0.76      0.74    107740

```

## LGBM

- Accuracy: 0.7534
- Macro F1: 0.5346
- Weighted F1: 0.7244
- Macro FAR: 0.0373
- ROC-AUC OvR: 0.9529

```
                precision    recall  f1-score   support

      analysis       0.16      0.24      0.19      1594
      backdoor       0.27      0.15      0.19      1535
           dos       0.25      0.34      0.29      3806
      exploits       0.82      0.81      0.82     19844
       fuzzers       0.71      0.18      0.29     16150
       generic       0.87      0.85      0.86      4181
        normal       0.79      0.97      0.87     51890
reconnaissance       0.81      0.74      0.77      7522
     shellcode       0.63      0.58      0.60      1091
         worms       0.47      0.45      0.46       127

      accuracy                           0.75    107740
     macro avg       0.58      0.53      0.53    107740
  weighted avg       0.75      0.75      0.72    107740

```
