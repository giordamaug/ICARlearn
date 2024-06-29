import pandas as pd
from sklearn.base import is_classifier, clone
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import numpy as np
import random

def evaluate_fold(train_x, train_y, test_x, test_y, estimator, genes, test_genes, targets, predictions, probabilities, fold, nclasses=2):
    # Initialize classifier
    clf = clone(estimator)
    clf.fit(train_x, train_y)
    probs = clf.predict_proba(test_x)
    preds = clf.predict(test_x)
    #preds = clf.classes_[np.argmax(probs, axis=1)]
    genes = np.concatenate((genes, test_genes))
    targets = np.concatenate((targets, test_y))
    cm = confusion_matrix(test_y, preds)
    predictions = np.concatenate((predictions, preds))
    probabilities = np.concatenate((probabilities, probs[:, 0]))

    # Calculate and store evaluation metrics for each fold
    roc_auc = roc_auc_score(test_y, probs[:, 1]) if nclasses == 2 else roc_auc_score(test_y, probs, multi_class="ovr", average="macro")
    metrics = {"index": fold,
               "ROC-AUC" : roc_auc, 
               "Accuracy" : accuracy_score(test_y, preds),
               "BA" : balanced_accuracy_score(test_y, preds), 
               "Sensitivity" : cm[1, 1] / (cm[1, 0] + cm[1, 1]),
               "Specificity" : cm[0, 0] / (cm[0, 0] + cm[0, 1]), 
               "MCC" : matthews_corrcoef(test_y, preds), 
               'CM' : cm}
    return genes, targets, predictions, probabilities, metrics
    
def skfold_cv(X, Y, estimator, n_splits=10, verbose: bool = False, show_progress: bool = False, seed: int = 42):
    """
    Perform cross-validated predictions using a classifier.

    :param DataFrame X: Features DataFrame.
    :param DataFrame Y: Target variable DataFrame.
    :param int n_splits: Number of folds for cross-validation.
    :param estimator object: Classifier method (must have fit, predict, predict_proba methods)
    :param str or None outfile: File name for saving predictions.
    :param bool show_progress: Verbosity level for printing progress bar (default: False).
    :param bool verbose: Whether to print verbose information.
    :param int or None seed: Random seed for reproducibility.

    :returns: Summary statistics of the cross-validated predictions, single measures and label predictions
    :rtype: Tuple(pd.DataFrame,pd.DataFrame,pd.DataFrame)

    :example:
 
    .. code-block:: python

        # Example usage
        from lightgbm import LGBMClassifier
        X_data = pd.DataFrame(...)
        Y_data = pd.DataFrame(...)
        clf = LGBMClassifier(random_state=0)
        df_scores, scores, predictions = k_fold_cv(df_X, df_y, clf, n_splits=5, verbose=True, display=True, seed=42)
    """
    # check estimator
    assert is_classifier(estimator) and hasattr(estimator, 'fit') and callable(estimator.fit) and hasattr(estimator, 'predict_proba') and callable(estimator.predict_proba), "Bad estimator imput!"

    # get list of genes
    allgenes = Y.index
    X = X.values
    y = Y.values.ravel()

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    nclasses = len(np.unique(y))
    genes = np.array([], dtype=str)
    targets = np.array([], dtype=np.int64)
    predictions = np.array([], dtype=np.int64)
    probabilities = np.array([], dtype=np.int64)
    scores = pd.DataFrame()

    if verbose:
        print(f'Classification with {estimator.__class__.__name__}...')

    # Iterate over each fold
    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), total=kf.get_n_splits(), 
                                                   desc=f"{n_splits}-fold", disable=not show_progress)):
        X_train, y_train = X[train_idx], y[train_idx]
        genes, targets, predictions, probabilities, metrics = evaluate_fold(X_train, y_train, X[test_idx], y[test_idx], 
                                                                            estimator, genes, allgenes[test_idx], targets, 
                                                                            predictions, probabilities, fold, nclasses)
        scores = pd.concat([scores, pd.DataFrame.from_dict(metrics, orient='index').T.set_index('index')], axis=0)

    # Calculate mean and standard deviation of evaluation metrics
    df_scores = pd.DataFrame([f'{val:.4f}±{err:.4f}' for val, err in zip(scores.loc[:, scores.columns != "CM"].mean(axis=0).values,
                                                                     scores.loc[:, scores.columns != "CM"].std(axis=0))] +
                             [(scores[['CM']].sum()).values[0].tolist()],
                             columns=['measure'], index=scores.columns)

    # Create DataFrame for storing detailed predictions
    df_results = pd.DataFrame({'gene': genes, 'label': targets, 'prediction': predictions, 'probabilities': probabilities}).set_index(['gene'])

    # Save detailed predictions to a CSV file if requested
    if in_notebook():
        ConfusionMatrixDisplay(confusion_matrix=np.array(df_scores.loc['CM']['measure']), display_labels=np.unique(y)).plot()

    # Return the summary statistics of cross-validated predictions, the single measures and the prediction results
    return df_scores, scores, df_results