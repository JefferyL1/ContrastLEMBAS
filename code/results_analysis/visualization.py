"""
Module for visualizing drug and protein embedding spaces.

Workflow:
    1. Accept a dict of embeddings (name -> tensor) and an optional dict of
       group labels (name -> group string) for coloring.
    2. Reduce the high-dimensional embeddings to 2D using PCA, t-SNE, or UMAP.
    3. Produce a scatter plot colored and labeled by group, and optionally save
       the figure to disk.

Note: drug and protein embeddings should be visualized separately by calling
the plotting function once per modality.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _get_reducer(method: str, n_components: int = 2):
    """
    Instantiate a dimensionality-reduction object for the given method.

    Args:
        method:       One of "pca", "tsne", or "umap" (case-insensitive).
        n_components: Output dimensionality. Defaults to 2.

    Returns:
        A fitted-able reducer with a fit_transform(X) method.

    Raises:
        ValueError: If method is not one of "pca", "tsne", "umap".
    """
    method = method.lower()
    if method == "pca":
        return PCA(n_components=n_components)
    elif method == "tsne":
        return TSNE(n_components=n_components, init="pca", learning_rate="auto")
    elif method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "umap-learn is required for UMAP visualization. "
                "Install it with: pip install umap-learn"
            ) from e
        return umap.UMAP(n_components=n_components)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'pca', 'tsne', 'umap'.")


def plot_embeddings(
    embeddings: dict,
    method: str = "tsne",
    groups: dict = None,
    title: str = None,
    output_path: str = None,
):
    """
    Reduce embeddings to 2D and produce a scatter plot colored by group.

    Args:
        embeddings:  Dict mapping name -> embedding tensor (shape [dim]).
                     All embeddings must have the same dimensionality.
        method:      Dimensionality-reduction method: "pca", "tsne", or "umap".
                     Defaults to "tsne".
        groups:      Optional dict mapping name -> group label string for
                     coloring and legend entries. Names absent from this dict
                     are assigned to group "other". If None, all points share
                     a single color.
        title:       Plot title. Defaults to "<METHOD> of embeddings".
        output_path: Optional file path to save the figure (e.g. "plot.png").
                     If None, the figure is only displayed.

    Returns:
        None
    """
    if not embeddings:
        raise ValueError("embeddings dict is empty.")

    names = list(embeddings.keys())
    matrix = np.stack([embeddings[n].detach().numpy() for n in names])

    reducer = _get_reducer(method)
    coords = reducer.fit_transform(matrix)  # [n, 2]

    # Assign each point a group label
    if groups is not None:
        point_groups = [groups.get(n, "other") for n in names]
    else:
        point_groups = ["all"] * len(names)

    unique_groups = sorted(set(point_groups))
    palette = cm.get_cmap("tab10", max(len(unique_groups), 1))
    color_map = {g: palette(i) for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for group in unique_groups:
        mask = [i for i, g in enumerate(point_groups) if g == group]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color_map[group]],
            label=group,
            s=18,
            alpha=0.7,
            linewidths=0,
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title or f"{method.upper()} of embeddings")
    if len(unique_groups) > 1:
        ax.legend(loc="best", fontsize=8, markerscale=1.5)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    plt.show()
