# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

class ModelEvaluator:
    def __init__(self, y_true, y_pred, y_scores=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores

    def basic_evaluation(self):
        precision = precision_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)
        roc_auc = roc_auc_score(self.y_true, self.y_pred)

        print("Basic Evaluation Metrics:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print(f"AUC-ROC: {roc_auc}")

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_scores)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="upper right")
        plt.show()

    def full_report(self, y_scores=None):
        self.basic_evaluation()
        if self.y_scores is not None:
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
        print(classification_report(self.y_true, self.y_pred))
