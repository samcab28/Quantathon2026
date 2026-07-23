"""SVM-RBF clásica solicitada en el reto."""

import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metrics import calculate_metrics


def train_classical_svm(X_train, y_train, cv_folds=5, seed=20260723):
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("balance", RandomUnderSampler(random_state=seed)),
            ("svm", SVC(kernel="rbf")),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {
            "svm__C": [0.1, 1, 10],
            "svm__gamma": ["scale", "auto", 0.01],
        },
        scoring="f1",
        cv=StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=seed,
        ),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_classical_svm(model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)
    result = calculate_metrics(y_test, predictions, scores)
    result["best_parameters"] = {
        key.replace("svm__", ""): value
        for key, value in model.best_params_.items()
    }
    result["cv_f1_mean"] = float(model.best_score_)
    result["cv_f1_std"] = float(
        model.cv_results_["std_test_score"][model.best_index_]
    )
    return result, predictions


def save_confusion_matrix(matrix, output_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                matrix[row][column],
                ha="center",
                va="center",
                fontsize=14,
            )
    ax.set_xticks([0, 1], ["No potable", "Potable"])
    ax.set_yticks([0, 1], ["No potable", "Potable"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("SVM-RBF: matriz de confusión")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
