"""Curvas y matrices de confusión compartidas por la interfaz y la exportación."""

from matplotlib.figure import Figure
import numpy as np


def curves(history, selected_epoch=None):
    fig = Figure(figsize=(10, 6), dpi=100, layout="constrained")
    axes = fig.subplots(2, 2).flat
    for ax, key, title in zip(
        axes,
        ("loss", "accuracy", "macro_f1", "lr"),
        (
            "Entropía cruzada sin ponderar",
            "Acierto",
            "F1 macro · igual peso por bioma",
            "Learning rate utilizado",
        ),
    ):
        available = [h for h in history if key in h]
        if available:
            ax.plot(
                [h["epoch"] for h in available],
                [h[key] for h in available],
                label="LR" if key == "lr" else "Entrenamiento",
                color="#167d8d",
            )
            val = [h for h in history if "val_" + key in h]
            if val:
                ax.plot(
                    [h["epoch"] for h in val],
                    [h["val_" + key] for h in val],
                    label="Validación",
                    color="#d18328",
                )
            if selected_epoch is not None:
                ax.axvline(
                    selected_epoch,
                    color="#77818c",
                    ls="--",
                    alpha=0.6,
                    label="Modelo elegido",
                )
            ax.legend(fontsize=8)
        elif history:
            ax.text(
                0.5,
                0.5,
                "No disponible en este modelo",
                ha="center",
                transform=ax.transAxes,
            )
        ax.set(xlabel="Época", title=title)
        ax.grid(alpha=0.18)
        if key in ("accuracy", "macro_f1"):
            ax.set_ylim(0, 1)
        if key == "lr" and available:
            ax.set_yscale("log")
    return fig


def confusion(metrics):
    names = metrics["classes"]
    matrix = np.array(metrics["confusion_matrix"])
    fig = Figure(figsize=(8, 6), dpi=100, layout="constrained")
    ax = fig.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Bioma predicho",
        ylabel="Bioma real",
        title="Matriz de confusión · test reservado",
    )
    ax.tick_params(axis="x", labelrotation=45)
    if len(names) <= 20:
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "#14263d",
                )
    return fig
