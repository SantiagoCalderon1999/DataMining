from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import metrics


def compute_metrics(model, X_test, y_test, X_train, y_train):
    y_pred = model.predict(X_test)
    score = 0
    for output in zip(y_pred, y_test):
        score = (score + 1) if output[0] == output[1] else score
    score = score / len(y_test)
    print(f"Score: {score}")

    print(f"Training score: {model.score(X_train, y_train)}")
    print(f"Test score: {model.score(X_test, y_test)}")
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(
        f"Precision per class: {metrics.precision_score(y_test, y_pred, average = None)}"
    )
    print(f"Recall per class: {metrics.recall_score(y_test, y_pred, average = None)}")
    print(f"F1-score per class: {metrics.f1_score(y_test, y_pred, average = None)}")

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()
    plt.show()
