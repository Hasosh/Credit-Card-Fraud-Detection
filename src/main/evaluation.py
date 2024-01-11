# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    matthews_corrcoef,
    confusion_matrix
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
        mcc = matthews_corrcoef(self.y_true, self.y_pred)

        print("Basic Evaluation Metrics:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print(f"MCC: {mcc}")

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc
        }

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)

        # Reorder the confusion matrix
        cm_reordered = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]

        sns.heatmap(cm_reordered, annot=True, fmt="d", cmap='Blues',
                    xticklabels=['Positive (1)', 'Negative (0)'],
                    yticklabels=['Positive (1)', 'Negative (0)'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

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
        metrics = self.basic_evaluation()
        self.plot_confusion_matrix()
        if self.y_scores is not None:
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
        print(classification_report(self.y_true, self.y_pred))

        return metrics
