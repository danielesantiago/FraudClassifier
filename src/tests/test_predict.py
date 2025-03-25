import os
from models.predict import calculate_metrics


def test_calculate_metrics(tmp_path):

    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    y_proba = [0.9, 0.2, 0.8, 0.3, 0.1]

    output_file = os.path.join(tmp_path, "metrics.txt")

    calculate_metrics(y_true, y_pred, y_proba, output_path=output_file)

    with open(output_file, "r") as f:
        content = f.read()

    assert "Accuracy" in content
    assert "F1 Score" in content
    assert "ROC AUC" in content
