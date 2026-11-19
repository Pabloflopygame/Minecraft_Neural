"""Controles del barrido y comparación de experimentos guardados."""

import csv
import json
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.figure import Figure

from sweep import OPTIONS, axis_values, layer_parameters, run_sweep

METRICS = (
    "val_macro_f1",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_loss",
    "test_macro_f1",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_loss",
    "macro_fpr_pct",
    "macro_fnr_pct",
)


class SweepView:
    def __init__(self, app, tab, output):
        self.app = app
        self.tab = tab
        self.output = output
        self.rows = []
        ttk.Label(
            tab,
            text="Selecciona los biomas, la configuración base y la arquitectura en sus pestañas.",
        ).pack(anchor="w")
        grid = ttk.Frame(tab)
        grid.pack(fill="x", pady=8)
        for i, label in enumerate(
            (
                "Parámetro",
                "Inicio o lista separada por comas",
                "Fin (incluido si coincide)",
                "Incremento",
            )
        ):
            ttk.Label(grid, text=label).grid(row=0, column=i, sticky="w")
        self.axes = []
        for row, defaults in enumerate(
            (("lr", "0.001", "0.003", "0.001"), ("batch_size", "32", "64", "32")), 1
        ):
            selector = ttk.Combobox(
                grid, state="readonly", postcommand=self.refresh_keys, width=25
            )
            selector.grid(row=row, column=0)
            selector.set(defaults[0])
            fields = []
            for col, default in enumerate(defaults[1:], 1):
                entry = ttk.Entry(grid, width=25)
                entry.insert(0, default)
                entry.grid(row=row, column=col, padx=3, pady=2)
                fields.append(entry)
            self.axes.append((selector, fields))
            selector.bind(
                "<<ComboboxSelected>>",
                lambda event, s=selector, f=fields: self.axis_changed(s, f),
            )
        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        app.button(bar, "Entrenar combinaciones", self.start)
        app.button(bar, "Abrir resumen.csv…", self.open_csv)
        ttk.Label(
            tab,
            text="Compara la validación para elegir los valores. Cada línea representa un valor del segundo parámetro.",
        ).pack(anchor="w", pady=8)
        choices = ttk.Frame(tab)
        choices.pack(fill="x")
        self.axis = self.choice(choices, "Eje horizontal", ())
        self.metric = self.choice(choices, "Métrica", METRICS)
        self.metric.set(METRICS[0])
        table = ttk.Frame(tab)
        table.pack(fill="x", pady=6)
        self.results = ttk.Treeview(
            table, columns=("x", "y", "accuracy", "fp", "fn"), height=4, show="headings"
        )
        for key, heading in (
            ("accuracy", "Accuracy de test"),
            ("fp", "Falsos positivos (macro)"),
            ("fn", "Falsos negativos (macro)"),
        ):
            self.results.heading(key, text=heading)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.results.yview)
        self.results.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.results.pack(fill="x", expand=True)
        self.plot = ttk.Frame(tab)
        self.plot.pack(fill="both", expand=True)

    def choice(self, bar, label, values):
        ttk.Label(bar, text=label).pack(side="left", padx=4)
        box = ttk.Combobox(bar, values=values, state="readonly", width=23)
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda event: self.draw())
        return box

    def refresh_keys(self):
        keys = list(self.app.values)
        try:
            spec = json.loads(self.app.arch_editor.get("1.0", "end"))
            keys.extend(layer_parameters(spec))
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
        for selector, _ in self.axes:
            selector.configure(values=keys)

    def axis_changed(self, selector, fields):
        key = selector.get()
        if key in OPTIONS:
            fields[0].delete(0, "end")
            fields[0].insert(0, ",".join(OPTIONS[key]))
        for field in fields[1:]:
            field.configure(state="disabled" if key in OPTIONS else "normal")

    def start(self):
        try:
            entries, cfg = self.app.training_inputs(validate=False)
            axes = {}
            allowed = set(self.app.values) | set(layer_parameters(cfg.architecture))
            for selector, fields in self.axes:
                key = selector.get()
                if key not in allowed or key in axes:
                    raise ValueError(
                        "Elige dos parámetros distintos de la configuración actual."
                    )
                axes[key] = axis_values(key, *(f.get() for f in fields))
        except Exception as exc:
            return messagebox.showerror("Configuración del barrido", str(exc))
        self.app.history = []
        self.app.tabs.select(self.app.data_tab)
        self.app.launch(
            lambda: run_sweep(
                entries,
                cfg,
                axes,
                self.output,
                lambda k, v: self.app.events.put((k, v)),
                self.app.cancel,
            ),
            "swept",
        )

    def open_csv(self):
        path = filedialog.askopenfilename(filetypes=[("Resumen del barrido", "*.csv")])
        if path:
            self.load(path)

    def load(self, path):
        try:
            with Path(path).open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if (
                not rows
                or not {"x_parameter", "y_parameter", "x_value", "y_value", *METRICS}
                <= rows[0].keys()
            ):
                raise ValueError(
                    "Selecciona el resumen.csv de un barrido con redes completadas."
                )
            for row in rows:
                for metric in METRICS:
                    if row[metric] or not metric.startswith("val_"):
                        float(row[metric])
        except (OSError, ValueError, csv.Error) as exc:
            return messagebox.showerror("Abrir barrido", str(exc))
        self.rows = rows
        self.axis.configure(values=(rows[0]["x_parameter"], rows[0]["y_parameter"]))
        self.results.delete(*self.results.get_children())
        for axis in ("x", "y"):
            self.results.heading(axis, text=rows[0][axis + "_parameter"])
        for row in rows:
            self.results.insert(
                "",
                "end",
                values=(
                    row["x_value"],
                    row["y_value"],
                    f"{float(row['test_accuracy']):.2%}",
                    f"{float(row['macro_fpr_pct']):.2f} %",
                    f"{float(row['macro_fnr_pct']):.2f} %",
                ),
            )
        self.axis.current(0)
        self.metric.set(
            "val_macro_f1" if any(r["val_macro_f1"] for r in rows) else "test_accuracy"
        )
        self.draw()
        self.app.tabs.select(self.tab)
        self.app.status.set(f"{len(rows)} combinaciones guardadas en {path}")

    def draw(self):
        if not self.rows:
            return
        x, group = ("x", "y") if self.axis.current() == 0 else ("y", "x")
        metric = self.metric.get()
        fig = Figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.subplots()
        series = {}
        for row in self.rows:
            if row[metric]:
                value = row[x + "_value"]
                if self.axis.get() not in OPTIONS:
                    value = float(value)
                series.setdefault(row[group + "_value"], []).append(
                    (value, float(row[metric]))
                )
        for label, points in series.items():
            xs, ys = zip(*sorted(points))
            ax.plot(xs, ys, marker="o", label=label)
        ax.set(xlabel=self.axis.get(), ylabel=metric)
        if series:
            ax.legend(title=self.rows[0][group + "_parameter"])
        else:
            ax.set_title("Sin datos de validación para esta métrica")
        self.app.render(self.plot, fig)
