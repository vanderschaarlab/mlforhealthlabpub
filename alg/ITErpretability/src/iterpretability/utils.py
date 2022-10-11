# stdlib
import random
from typing import Optional

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error


abbrev_dict = {
    "shapley_value_sampling": "SVS",
    "integrated_gradients": "IG",
    "kernel_shap": "SHAP",
    "gradient_shap": "GSHAP",
    "feature_permutation": "FP",
    "feature_ablation": "FA",
    "deeplift": "DL",
    "lime": "LIME",
}

explainer_symbols = {
    "shapley_value_sampling": "D",
    "integrated_gradients": "8",
    "kernel_shap": "s",
    "feature_permutation": "<",
    "feature_ablation": "x",
    "deeplift": "H",
    "lime": ">",
}

cblind_palete = sns.color_palette("colorblind", as_cmap=True)
learner_colors = {
    "SLearner": cblind_palete[0],
    "TLearner": cblind_palete[1],
    "TARNet": cblind_palete[3],
    "CFRNet_0.01": cblind_palete[4],
    "CFRNet_0.001": cblind_palete[6],
    "CFRNet_0.0001": cblind_palete[7],
    "DRLearner": cblind_palete[8],
    "XLearner": cblind_palete[5],
    "Truth": cblind_palete[9],
}


def enable_reproducible_results(seed: int = 42) -> None:
    """
    Set a fixed seed for all the used libraries

    Args:
        seed: int
            The seed to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def dataframe_line_plot(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    explainers: list,
    learners: list,
    x_logscale: bool = True,
    aggregate: bool = False,
    aggregate_type: str = "mean",
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.set_style("white")
    for learner_name in learners:
        for explainer_name in explainers:
            sub_df = df.loc[
                (df["Learner"] == learner_name) & (df["Explainer"] == explainer_name)
            ]
            if aggregate:
                sub_df = sub_df.groupby(x_axis).agg(aggregate_type).reset_index()
            x_values = sub_df.loc[:, x_axis].values
            y_values = sub_df.loc[:, y_axis].values
            ax.plot(
                x_values,
                y_values,
                color=learner_colors[learner_name],
                marker=explainer_symbols[explainer_name],
            )

    learner_lines = [
        Line2D([0], [0], color=learner_colors[learner_name], lw=2)
        for learner_name in learners
    ]
    explainer_lines = [
        Line2D([0], [0], color="black", marker=explainer_symbols[explainer_name])
        for explainer_name in explainers
    ]

    legend_learners = plt.legend(
        learner_lines, learners, loc="lower left", bbox_to_anchor=(1.04, 0.7)
    )
    legend_explainers = plt.legend(
        explainer_lines,
        [abbrev_dict[explainer_name] for explainer_name in explainers],
        loc="lower left",
        bbox_to_anchor=(1.04, 0),
    )
    plt.subplots_adjust(right=0.75)
    ax.add_artist(legend_learners)
    ax.add_artist(legend_explainers)
    if x_logscale:
        ax.set_xscale("log")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return fig


def compute_pehe(
    cate_true: np.ndarray,
    cate_pred: torch.Tensor,
) -> tuple:
    pehe = np.sqrt(mean_squared_error(cate_true, cate_pred.detach().cpu().numpy()))
    return pehe


def compute_cate_metrics(
    cate_true: np.ndarray,
    y_true: np.ndarray,
    w_true: np.ndarray,
    mu0_pred: torch.Tensor,
    mu1_pred: torch.Tensor,
) -> tuple:
    mu0_pred = mu0_pred.detach().cpu().numpy()
    mu1_pred = mu1_pred.detach().cpu().numpy()

    cate_pred = mu1_pred - mu0_pred

    pehe = np.sqrt(mean_squared_error(cate_true, cate_pred))

    y_pred = w_true.reshape(len(cate_true),) * mu1_pred.reshape(len(cate_true),) + (
        1
        - w_true.reshape(
            len(cate_true),
        )
    ) * mu0_pred.reshape(
        len(cate_true),
    )
    factual_rmse = np.sqrt(
        mean_squared_error(
            y_true.reshape(
                len(cate_true),
            ),
            y_pred,
        )
    )
    return pehe, factual_rmse


def attribution_accuracy(
    target_features: list, feature_attributions: np.ndarray
) -> float:
    """
    Computes the fraction of the most important features that are truly important
    Args:
        target_features: list of truly important feature indices
        feature_attributions: feature attribution outputted by a feature importance method

    Returns:
        Fraction of the most important features that are truly important
    """

    n_important = len(target_features)  # Number of features that are important
    largest_attribution_idx = torch.topk(
        torch.from_numpy(feature_attributions), n_important
    )[
        1
    ]  # Features with largest attribution
    accuracy = 0  # Attribution accuracy
    for k in range(len(largest_attribution_idx)):
        accuracy += len(np.intersect1d(largest_attribution_idx[k], target_features))
    return accuracy / (len(feature_attributions) * n_important)
