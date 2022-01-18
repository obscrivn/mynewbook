## Evaluation Metrics

<small>Source: Kyrtis Pykes. 2020. Confusion Matrix</small>

- **Intrinsic Evaluation** — Focuses on intermediary objectives (i.e. the performance of an NLP component on a defined subtask)
- **Extrinsic Evaluation** — Focuses on the performance of the final objective (i.e. the performance of the component on the complete application)

### Confusion Matrix

**What is the Confusion Matrix?**
The confusion matrix is a useful tool used for classification tasks in machine learning with the primary objective of visualizing the performance of a machine learning model.

![](_static/cmatrix.png)

- **True Positives**: The model predicted positive and the label was actually positive.
- **True Negatives**: The model predicted negative and the label was actually negative.
- **False Positives**: The model predicted positive and the label was actually negative.
- **False Negatives**: The model predicted negative and the label was actually positive.


### Types

#### Accuracy

![](_static/accuracy.png)
Whenever the accuracy metric is used, we aim to learn the closeness of a measured value to a known value. It is therefore typically used in instances where the output variable is categorical or discrete (classification task).

#### Precision

![](_static/precision.png)
_the positive predictive value_
In instances where we are concerned with how exact the model's predictions are we would use Precision. The precision metric would inform us of the number of labels that are actually labeled as positive in correspondence to the instances that the classifier labeled as positive.



#### Recall

![](_static/recall.png)
_sensitivity or the true positive rate_
Recall measures how well the model can recall the positive class (i.e. the number of positive labels that the model identified as positive)


#### F1

![](_static/f1.png)
When we need to strike a balance between precision and recall, a better known metric is the F1-score ( also know as the F measure). F1-score is a measure of accuracy that considers the precision and recall 

#### Practice
![](_static/dogs.png)

- Q1. Identify: TP, TN, FP, FN
- Q2. Calculate accuracy
- Q3. Calculate Precision
- Q4. Calculate Recall
- Q5. Calculate F1


#### ROC Curve

![](_static/roc.png)

An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

- True Positive Rate
- False Positive Rate

An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve

AUC stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve

https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

Source: https://towardsdatascience.com/confusion-matrix-un-confused-1ba98dee0d7f
Source: https://towardsdatascience.com/confusion-matrix-un-confused-1ba98dee0d7f
