"""
Evaluation utilities: bootstrap CI and full metric computation.
"""

import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=2000, seed=42):
    """Compute bootstrap 95% CI for AUC."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    return lower, upper


def compute_all_metrics(y_true, y_prob):
    """Compute AUC with CI, optimal threshold (Youden), and full classification metrics."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]

    auc = roc_auc_score(y_true, y_prob)
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob)

    y_pred = (y_prob >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0

    return {
        'AUC': auc,
        'CI_low': ci_low,
        'CI_high': ci_high,
        'Threshold': optimal_threshold,
        'Accuracy': acc,
        'Sensitivity': sens,
        'Specificity': spec,
        'PPV': ppv,
        'NPV': npv,
        'F1': f1,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--progress_file', type=str, required=True,
                        help='Path to progress_<model>.json file from training.')
    args = parser.parse_args()

    with open(args.progress_file, 'r') as f:
        prog = json.load(f)

    m = compute_all_metrics(prog['all_labels'], prog['all_probs'])
    fold_aucs = prog['fold_aucs']

    print(f'\nResults for: {args.progress_file}')
    print(f'{"-"*60}')
    print(f'Fold AUC:        {[round(a, 3) for a in fold_aucs]}')
    print(f'Mean Fold AUC:   {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}')
    print(f'Aggregated AUC:  {m["AUC"]:.3f} (95% CI: {m["CI_low"]:.3f}–{m["CI_high"]:.3f})')
    print(f'Threshold:       {m["Threshold"]:.3f}')
    print(f'Accuracy:        {m["Accuracy"]:.1%}')
    print(f'Sensitivity:     {m["Sensitivity"]:.1%}')
    print(f'Specificity:     {m["Specificity"]:.1%}')
    print(f'PPV:             {m["PPV"]:.1%}')
    print(f'NPV:             {m["NPV"]:.1%}')
    print(f'F1:              {m["F1"]:.3f}')
    print(f'Confusion:       TP={m["TP"]}, TN={m["TN"]}, FP={m["FP"]}, FN={m["FN"]}')


if __name__ == '__main__':
    main()
