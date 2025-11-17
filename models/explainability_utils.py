"""
Explainability Utilities for Day 2 TP

This module provides functions for SHAP and LIME analysis.
These utilities are used in the 02_explainability_analysis.ipynb notebook.

Author: Feda Almuhisen
Course: M2 SID - Processus Data
Year: 2025-2026
"""

import numpy as np
import torch
import shap
from lime import lime_tabular


def calculate_reconstruction_error(model, X, device='cpu'):
    """
    Calculate reconstruction error for each sample.

    Args:
        model: Trained autoencoder model
        X: Input data (numpy array)
        device: 'cpu' or 'cuda'

    Returns:
        errors: Reconstruction error per sample (numpy array)
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

    return errors.cpu().numpy()


def detect_anomalies(errors, threshold_method="mean_plus_2std", percentile=95):
    """
    Detect anomalies based on reconstruction error threshold.

    Args:
        errors: Reconstruction errors (numpy array)
        threshold_method: 'mean_plus_2std', 'mean_plus_3std', or 'percentile'
        percentile: Percentile value if using 'percentile' method (default: 95)

    Returns:
        threshold: Anomaly detection threshold
        is_anomaly: Boolean array indicating anomalies
    """
    if threshold_method == "mean_plus_2std":
        threshold = errors.mean() + 2 * errors.std()
    elif threshold_method == "mean_plus_3std":
        threshold = errors.mean() + 3 * errors.std()
    elif threshold_method == "percentile":
        threshold = np.percentile(errors, percentile)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")

    is_anomaly = errors > threshold
    return threshold, is_anomaly


def create_shap_explainer(predict_fn, background_data):
    """
    Create a SHAP KernelExplainer.

    Args:
        predict_fn: Function that takes X and returns predictions
        background_data: Background dataset for SHAP approximation

    Returns:
        explainer: SHAP KernelExplainer object

    Example:
        >>> explainer = create_shap_explainer(predict_fn, X_train[:100])
    """
    # TODO 1: Students complete this
    # HINT: Use shap.KernelExplainer
    return shap.KernelExplainer(predict_fn, background_data[:100])


def calculate_shap_values(explainer, X, nsamples=50):
    """
    Calculate SHAP values for given samples.

    Args:
        explainer: SHAP explainer object
        X: Data to explain (numpy array)
        nsamples: Number of samples for SHAP approximation

    Returns:
        shap_values: SHAP values (numpy array)

    Example:
        >>> shap_values = calculate_shap_values(explainer, X_test[:20])
    """
    # TODO 2: Students complete this
    # HINT: Use explainer.shap_values(X, nsamples=...)
    return explainer.shap_values(X, nsamples = nsamples)


def create_lime_explainer(background_data, feature_names, mode='regression'):
    """
    Create a LIME TabularExplainer.

    Args:
        background_data: Training data for LIME
        feature_names: List of feature names
        mode: 'regression' or 'classification'

    Returns:
        explainer: LIME TabularExplainer object

    Example:
        >>> explainer = create_lime_explainer(X_train[:100], feature_names)
    """
    # TODO 3: Students complete this
    # HINT: Use lime_tabular.LimeTabularExplainer
    return lime_tabular.LimeTabularExplainer(
        training_data=background_data,
        feature_names=feature_names,
        mode=mode,
        verbose=False
    )

def explain_instance_lime(explainer, instance, predict_fn, num_features=14, num_samples=500):
    """
    Generate LIME explanation for a single instance.

    Args:
        explainer: LIME explainer object
        instance: Single sample to explain (1D numpy array)
        predict_fn: Function that takes X and returns predictions
        num_features: Number of features to explain
        num_samples: Number of perturbation samples

    Returns:
        explanation: LIME explanation object
        feature_weights: Dictionary of {feature_name: weight}

    Example:
        >>> exp, weights = explain_instance_lime(explainer, X_test[0], predict_fn)
    """
    
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples
    )
    feature_weights = dict(explanation.as_list())
    return explanation, feature_weights

    raise NotImplementedError("TODO 4: Generate LIME explanation")

# def str_in_key(chaine, dic):
#     keys = []
#     for key in dic.keys():
#         if chaine in key:
#             keys.append(key)
#     return keys

def compare_shap_lime(shap_values, lime_weights, feature_names):
    """
    Compare SHAP and LIME explanations by checking sign agreement.

    Args:
        shap_values: SHAP values for one sample (1D numpy array)
        lime_weights: LIME weights dictionary {feature_name: weight}
        feature_names: List of feature names in order

    Returns:
        agreement_rate: Percentage of features with same sign
        disagreeing_features: List of features where SHAP and LIME disagree

    Example:
        >>> rate, features = compare_shap_lime(shap_vals, lime_dict, feature_names)
    """
    # TODO 5: Students complete this
    # HINT: Convert lime_weights dict to array matching feature order
    # HINT: Check sign agreement: (shap * lime) >= 0

    # agreement_rate = 0
    # disagreeing_features = []
    # for i, feature in enumerate(feature_names):
    #     if shap_values[i] * lime_weights.get(str_in_key(feature, lime_weights))[0] > 0:
    #         agreement_rate += 1
    #     else:
    #         disagreeing_features.append(feature)
    # return agreement_rate/len(feature_names), disagreeing_features
    

    raise NotImplementedError("TODO 5: Compare SHAP vs LIME")


def interpret_for_engineer(shap_values, feature_names, error, threshold, is_anomaly, top_n=3):
    """
    Create engineer-friendly interpretation of ML prediction.

    Args:
        shap_values: SHAP values for one sample (1D numpy array)
        feature_names: List of feature names
        error: Reconstruction error value
        threshold: Anomaly detection threshold
        is_anomaly: Boolean indicating if sample is anomalous
        top_n: Number of top features to include

    Returns:
        interpretation: String with engineer-friendly report

    Example:
        >>> report = interpret_for_engineer(shap_vals, features, 0.42, 0.31, True)
        >>> print(report)
    """
    # TODO 6: Students complete this
    # HINT: Identify top N most important sensors
    # HINT: Explain what they mean in plain language
    # HINT: Provide actionable recommendations
    raise NotImplementedError("TODO 6: Create engineer interpretation")


# Helper functions (PROVIDED to students)

def get_top_features(shap_values, feature_names, n=5):
    """
    Get top N features by absolute SHAP value.

    Args:
        shap_values: SHAP values (1D array for single sample OR 2D for multiple)
        feature_names: List of feature names
        n: Number of top features to return

    Returns:
        top_features: List of (feature_name, shap_value) tuples
    """
    if len(shap_values.shape) == 2:
        # Multiple samples: take mean absolute SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    else:
        # Single sample
        mean_abs_shap = np.abs(shap_values)

    top_indices = np.argsort(mean_abs_shap)[::-1][:n]
    top_features = [(feature_names[i], shap_values[i] if len(shap_values.shape) == 1 else mean_abs_shap[i])
                    for i in top_indices]

    return top_features


def plot_shap_waterfall(shap_values, feature_names, title="SHAP Waterfall Plot"):
    """
    Plot SHAP waterfall for a single sample.

    Args:
        shap_values: SHAP values for one sample (1D numpy array)
        feature_names: List of feature names
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt

    # Sort by absolute SHAP value
    sorted_indices = np.argsort(np.abs(shap_values))[::-1][:10]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_shap = shap_values[sorted_indices]

    # Color: red for positive, green for negative
    colors = ['red' if v > 0 else 'green' for v in sorted_shap]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_features)), sorted_shap, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('SHAP value', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    return fig


def plot_lime_weights(lime_weights, title="LIME Feature Weights"):
    """
    Plot LIME feature weights.

    Args:
        lime_weights: Dictionary {feature_name: weight}
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Convert to DataFrame and sort
    df = pd.DataFrame(list(lime_weights.items()), columns=['Feature', 'Weight'])
    df = df.sort_values('Weight', key=abs, ascending=False).head(10)

    # Color: red for positive, green for negative
    colors = ['red' if v > 0 else 'green' for v in df['Weight']]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(df)), df['Weight'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Feature'])
    ax.set_xlabel('LIME weight', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    return fig


def plot_shap_vs_lime(shap_values, lime_weights, feature_names, title="SHAP vs LIME Comparison"):
    """
    Scatter plot comparing SHAP and LIME values.

    Args:
        shap_values: SHAP values for one sample (1D numpy array)
        lime_weights: LIME weights dictionary
        feature_names: List of feature names
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt

    # Convert LIME dict to array
    lime_array = np.array([lime_weights.get(f, 0.0) for f in feature_names])

    # Calculate agreement
    agreements = (shap_values * lime_array) >= 0
    agreement_rate = agreements.sum() / len(agreements)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(shap_values, lime_array, alpha=0.6, s=100, c='steelblue')

    # Annotate points
    for i, feat in enumerate(feature_names):
        ax.annotate(feat.replace('sensor_', 'S'),
                   (shap_values[i], lime_array[i]),
                   fontsize=9, alpha=0.7)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('SHAP value', fontweight='bold', fontsize=12)
    ax.set_ylabel('LIME value', fontweight='bold', fontsize=12)
    ax.set_title(f'{title}\nAgreement: {agreement_rate:.1%}', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig


def print_summary_report(shap_values, feature_names, errors, threshold, is_anomaly):
    """
    Print summary report of explainability analysis.

    Args:
        shap_values: SHAP values for all samples (2D numpy array)
        feature_names: List of feature names
        errors: Reconstruction errors
        threshold: Anomaly threshold
        is_anomaly: Boolean array of anomalies
    """
    print("=" * 70)
    print("EXPLAINABILITY ANALYSIS SUMMARY")
    print("=" * 70)
    print()
    print(f"Total samples analyzed: {len(errors)}")
    print(f"Anomaly threshold: {threshold:.4f}")
    print(f"Anomalies detected: {is_anomaly.sum()} ({100*is_anomaly.sum()/len(is_anomaly):.2f}%)")
    print()
    print("Top 5 Most Important Sensors (Global SHAP):")
    top_features = get_top_features(shap_values, feature_names, n=5)
    for i, (feat, val) in enumerate(top_features, 1):
        print(f"  {i}. {feat}: {val:.4f}")
    print()
    print("=" * 70)
