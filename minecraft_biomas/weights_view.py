"""Vista acotada de los pesos de capas densas guardados en un checkpoint."""

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def dense_weight_layers(state_dict):
    """Devuelve (clave, entradas, salidas) para pesos de capas Linear."""
    return [
        (name, int(value.shape[1]), int(value.shape[0]))
        for name, value in state_dict.items()
        if name.endswith(".weight")
        and isinstance(value, torch.Tensor)
        and value.ndim == 2
    ]


def draw_dense_connections(weights, layer_name, max_neurons=12, max_connections=60):
    """Dibuja una selección de conexiones fuertes sin alterar el checkpoint.

    Para capas grandes, muestra las unidades con mayor peso absoluto y, dentro
    de esa submatriz, las conexiones de mayor magnitud. No son activaciones ni
    una estimación de la importancia causal de cada unidad.
    """
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise ValueError("La capa seleccionada no contiene una matriz de pesos 2D.")
    if max_neurons < 1 or max_connections < 1:
        raise ValueError("Los límites de visualización deben ser positivos.")

    matrix = weights.detach().to(device="cpu", dtype=torch.float32).numpy()
    outputs, inputs = matrix.shape
    if not inputs or not outputs or not np.isfinite(matrix).all():
        raise ValueError("La matriz de pesos está vacía o contiene valores no finitos.")

    magnitudes = np.abs(matrix)
    in_ids = np.sort(
        np.argsort(magnitudes.max(axis=0), kind="stable")[-min(max_neurons, inputs):]
    )
    out_ids = np.sort(
        np.argsort(magnitudes.max(axis=1), kind="stable")[-min(max_neurons, outputs):]
    )
    visible = matrix[np.ix_(out_ids, in_ids)]
    peak = float(np.max(np.abs(visible)))
    in_y = np.linspace(0.93, 0.07, len(in_ids)) if len(in_ids) > 1 else [0.5]
    out_y = np.linspace(0.93, 0.07, len(out_ids)) if len(out_ids) > 1 else [0.5]

    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")
    ax.set_title(f"Pesos entre unidades · {layer_name}", pad=35, fontsize=13)

    # Solo las conexiones más intensas de las unidades seleccionadas.
    order = np.argsort(np.abs(visible).ravel(), kind="stable")[::-1]
    painted = 0
    for flat in order[:max_connections]:
        out_pos, in_pos = np.unravel_index(flat, visible.shape)
        value = float(visible[out_pos, in_pos])
        if value == 0:
            continue
        strength = abs(value) / peak
        ax.plot(
            (0.22, 0.78), (in_y[in_pos], out_y[out_pos]),
            color="#167d8d" if value > 0 else "#d18328",
            alpha=0.2 + 0.7 * strength,
            linewidth=0.6 + 2.5 * strength,
            zorder=1,
        )
        painted += 1

    for idx, y in zip(in_ids, in_y):
        ax.scatter(0.22, y, s=85, color="#264e72", zorder=2)
        ax.text(0.19, y, f"E{idx}", ha="right", va="center", fontsize=8)
    for idx, y in zip(out_ids, out_y):
        ax.scatter(0.78, y, s=85, color="#264e72", zorder=2)
        ax.text(0.81, y, f"S{idx}", ha="left", va="center", fontsize=8)

    fig.legend(
        handles=[
            Line2D([0], [0], color="#167d8d", lw=2, label="Positivo"),
            Line2D([0], [0], color="#d18328", lw=2, label="Negativo"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False,
    )
    fig.text(
        0.5, 0.045,
        f"Vista parcial: {len(in_ids)}/{inputs} entradas · "
        f"{len(out_ids)}/{outputs} salidas · {painted} conexiones dibujadas. "
        "Grosor = |peso| relativo de la selección.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.79, bottom=0.14)
    return fig
