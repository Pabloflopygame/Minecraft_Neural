"""Iniciar con: python app.py. Toda interacción con Tk ocurre en el hilo principal."""

from pathlib import Path
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk, ImageOps
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core import Config, train, predict, load_checkpoint, image_files, Cancelled
from plots import curves, confusion
from weights_view import dense_weight_layers, draw_dense_connections
from architectures import ORIGINAL, PRESETS, validate_architecture

BASE = Path(__file__).resolve().parent
OUTPUT = BASE.parent / "OutPut"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Biomas · Laboratorio de clasificación")
        self.geometry("1220x950")
        self.minsize(1040, 820)
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.active = None
        self.history = []
        self.controls = []
        self.canvases = {}
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=7)
        style.configure("Treeview", rowheight=27)
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("TNotebook.Tab", padding=(16, 8))
        top = ttk.Frame(self, padding=16)
        top.pack(fill="x")
        ttk.Label(top, text="Biomas de Minecraft", style="Title.TLabel").pack(
            side="left"
        )
        gpu = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CUDA no disponible · CPU"
        )
        ttk.Label(top, text=gpu).pack(side="right")
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        self.data_tab = self.tab("1 · Datos y entrenamiento")
        self.results_tab = self.tab("2 · Resultados")
        self.predict_tab = self.tab("3 · Probar imagen")
        self.arch_tab = self.tab("4 · Arquitecturas")
        self.net_tab = self.tab("5 · Red neuronal")
        self.build_data()
        self.build_results()
        self.build_predict()
        self.refresh_selection()
        self.build_architectures()
        self.status = tk.StringVar(
            value="Selecciona carpetas locales; el dataset preparado por setup está disponible en Explorar carpeta raíz."
        )
        ttk.Label(self, textvariable=self.status, padding=8).pack(fill="x")
        self.poll_job = self.after(100, self.poll)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def tab(self, name):
        frame = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(frame, text=name)
        return frame

    def button(self, parent, text, command):
        b = ttk.Button(parent, text=text, command=command)
        b.pack(side="left", padx=3)
        self.controls.append((b, "normal"))
        return b

    def build_data(self):
        bar = ttk.Frame(self.data_tab)
        bar.pack(fill="x")
        self.button(bar, "Añadir bioma...", self.add_folder)
        self.button(bar, "Explorar carpeta raíz...", self.add_root)
        self.button(bar, "Renombrar", self.rename)
        self.button(bar, "Quitar", self.remove)
        self.button(bar, "Guardar selección", self.save_selection)
        self.button(bar, "Cargar selección", self.load_selection)
        ttk.Label(
            self.data_tab,
            text="Doble clic en una fila: incluir/excluir. Selecciona una fila y pulsa Renombrar para cambiar su etiqueta.",
        ).pack(anchor="w", pady=8)
        self.selection_info = tk.StringVar(
            value="Seleccionados: 0 · elige entre 5 y 10 biomas."
        )
        ttk.Label(self.data_tab, textvariable=self.selection_info).pack(
            anchor="w", pady=(0, 6)
        )
        box = ttk.Frame(self.data_tab)
        box.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(
            box,
            columns=("use", "name", "count", "folder"),
            show="headings",
            height=7,
            selectmode="extended",
        )
        for col, title, width in [
            ("use", "Usar", 55),
            ("name", "Nombre del bioma", 160),
            ("count", "Imágenes", 80),
            ("folder", "Carpeta local", 530),
        ]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, stretch=col == "folder")
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self.toggle)
        cfg = ttk.LabelFrame(
            self.data_tab, text="Configuración · test fijo: 20 %", padding=10
        )
        cfg.pack(fill="x", pady=10)
        self.values = {}
        fields = [
            ("epochs", "Épocas", "30"),
            ("batch_size", "Batch", "64"),
            ("image_size", "Tamaño", "64"),
            ("lr", "Learning rate", "0.001"),
            ("seed", "Semilla", "42"),
            ("validation_percent", "Validación % total", "16"),
            ("device", "Dispositivo", "cuda" if torch.cuda.is_available() else "cpu"),
            ("optimizer", "Optimizador", "AdamW"),
            ("scheduler", "Plan LR", "plateau"),
            ("weight_decay", "Weight decay", "0.0001"),
            ("class_weighting", "Pesos CE", "balanced"),
            ("max_images_per_class", "Máx. imágenes/bioma", "0"),
            ("selection_metric", "Elegir modelo por", "val_macro_f1"),
            ("scheduler_patience", "Paciencia LR", "2"),
        ]
        options = {
            "device": ("cuda", "cpu"),
            "image_size": ("32", "64", "96", "128"),
            "optimizer": ("Adam", "AdamW", "SGD"),
            "scheduler": ("none", "plateau", "cosine"),
            "class_weighting": ("none", "balanced"),
            "selection_metric": ("val_loss", "val_macro_f1"),
        }
        for i, (key, label, value) in enumerate(fields):
            row = (i // 7) * 2
            col = i % 7
            v = tk.StringVar(value=value)
            self.values[key] = v
            ttk.Label(cfg, text=label).grid(row=row, column=col, padx=4, sticky="w")
            if key in options:
                w = ttk.Combobox(
                    cfg, textvariable=v, values=options[key], width=13, state="readonly"
                )
                state = "readonly"
            else:
                w = ttk.Entry(cfg, textvariable=v, width=11)
                state = "normal"
            w.grid(row=row + 1, column=col, padx=4, pady=5, sticky="ew")
            self.controls.append((w, state))
            cfg.columnconfigure(col, weight=1)
        ttk.Label(
            cfg,
            text="Test fijo 20 %. Validación 16: 64/16/20. Máx. 0: todas; 100: hasta 100 por bioma, antes de dividir.",
        ).grid(row=4, column=0, columnspan=7, sticky="w")
        ttk.Label(
            cfg,
            text="Pesos balanced: compensa clases pequeñas. LR plateau: reduce al estancarse; cosine: descenso suave; none: constante.",
        ).grid(row=5, column=0, columnspan=7, sticky="w")
        runbar = ttk.Frame(self.data_tab)
        runbar.pack(fill="x")
        self.train_button = self.button(runbar, "Train · Entrenar", self.start_train)
        self.stop = ttk.Button(
            runbar, text="Cancelar", command=self.cancel.set, state="disabled"
        )
        self.stop.pack(side="left", padx=5)
        self.progress = ttk.Progressbar(runbar, maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=10)
        self.log = ScrolledText(
            self.data_tab, height=7, font=("Consolas", 10), state="disabled"
        )
        self.log.pack(fill="x", pady=(10, 0))

    def build_results(self):
        bar = ttk.Frame(self.results_tab)
        bar.pack(fill="x")
        self.button(bar, "Cargar modelo / resultados...", self.load_model)
        self.button(bar, "Exportar gráficas...", self.export_plots)
        self.summary = tk.StringVar(value="Todavía no hay un entrenamiento cargado.")
        ttk.Label(self.results_tab, textvariable=self.summary, wraplength=1050).pack(
            anchor="w", pady=10
        )
        sub = ttk.Notebook(self.results_tab)
        sub.pack(fill="both", expand=True)
        self.curve_plot = ttk.Frame(sub)
        self.cm_plot = ttk.Frame(sub)
        report_tab = ttk.Frame(sub)
        sub.add(self.curve_plot, text="Curvas")
        sub.add(self.cm_plot, text="Matriz de confusión")
        sub.add(report_tab, text="Métricas por bioma")
        self.report = ttk.Treeview(
            report_tab,
            columns=("name", "precision", "recall", "f1", "n"),
            show="headings",
        )
        for key, title in [
            ("name", "Bioma"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
            ("n", "Imágenes test"),
        ]:
            self.report.heading(key, text=title)
            self.report.column(key, width=150)
        self.report.pack(fill="both", expand=True)
        data_tab = ttk.Frame(sub)
        sub.add(data_tab, text="Datos utilizados")
        self.data_report = ttk.Treeview(
            data_tab,
            columns=("name", "available", "selected", "train", "val", "test", "weight"),
            show="headings",
        )
        for key, title in [
            ("name", "Bioma"),
            ("available", "Útiles originales"),
            ("selected", "Tras límite"),
            ("train", "Train"),
            ("val", "Validación"),
            ("test", "Test"),
            ("weight", "Peso CE"),
        ]:
            self.data_report.heading(key, text=title)
            self.data_report.column(key, width=130)
        self.data_report.pack(fill="both", expand=True)
        self.render(self.curve_plot, curves([]))

    def build_predict(self):
        bar = ttk.Frame(self.predict_tab)
        bar.pack(fill="x")
        self.button(bar, "Cargar modelo...", self.load_model)
        self.button(bar, "Seleccionar imagen...", self.choose_image)
        self.pred_title = tk.StringVar(value="Entrena o carga un modelo para empezar.")
        ttk.Label(
            self.predict_tab,
            textvariable=self.pred_title,
            font=("Segoe UI", 12, "bold"),
            wraplength=1000,
        ).pack(anchor="w", pady=12)
        body = ttk.Frame(self.predict_tab)
        body.pack(fill="both", expand=True)
        self.preview = ttk.Label(
            body, text="Vista previa de tu captura", anchor="center"
        )
        self.preview.pack(side="left", fill="both", expand=True, padx=8)
        self.prob_tree = ttk.Treeview(
            body, columns=("name", "percent"), show="headings", height=16
        )
        for key, title, width in [
            ("name", "Bioma", 240),
            ("percent", "Probabilidad softmax", 190),
        ]:
            self.prob_tree.heading(key, text=title)
            self.prob_tree.column(key, width=width)
        self.prob_tree.pack(side="right", fill="both", expand=True)
        ttk.Label(
            self.predict_tab,
            text="Los porcentajes suman 100 % entre las clases del modelo. No son certeza calibrada; el modelo no detecta biomas desconocidos.",
            wraplength=1000,
        ).pack(anchor="w", pady=10)

    def build_architectures(self):
        bar = ttk.Frame(self.arch_tab)
        bar.pack(fill="x")
        self.preset = tk.StringVar(value="Original")
        combo = ttk.Combobox(
            bar,
            textvariable=self.preset,
            values=list(PRESETS),
            state="readonly",
            width=30,
        )
        combo.pack(side="left")
        self.controls.append((combo, "readonly"))
        self.button(bar, "Usar plantilla", self.use_preset)
        self.button(bar, "Importar JSON", self.import_architecture)
        self.button(bar, "Exportar JSON", self.export_architecture)
        self.button(bar, "Validar / ver resumen", self.validate_editor)
        ttk.Label(
            self.arch_tab,
            text='Edita las capas en orden. Las entradas se calculan automáticamente; "classes" adapta las salidas a los biomas.\nEl JSON del editor se utiliza al iniciar el siguiente entrenamiento. Consulta docs/ARQUITECTURAS.md para el formato.',
            wraplength=1050,
        ).pack(anchor="w", pady=12)
        self.arch_editor = ScrolledText(
            self.arch_tab, font=("Consolas", 11), wrap="none", undo=True
        )
        self.arch_editor.pack(fill="both", expand=True)
        self.controls.append((self.arch_editor, "normal"))
        self.arch_editor.insert(
            "1.0", json.dumps(ORIGINAL, ensure_ascii=False, indent=2)
        )
        self.arch_editor.bind("<<Modified>>", self.architecture_edited)
        self.arch_notice = tk.StringVar(value="")
        ttk.Label(self.arch_tab, textvariable=self.arch_notice, wraplength=1050).pack(
            anchor="w", pady=8
        )
        self.net_text = ScrolledText(
            self.net_tab, font=("Consolas", 11), wrap="none", state="disabled",
            height=11,
        )
        self.net_text.pack(fill="x")
        weight_bar = ttk.Frame(self.net_tab)
        weight_bar.pack(fill="x", pady=(12, 4))
        ttk.Label(weight_bar, text="Conexiones aprendidas (capas densas):").pack(
            side="left", padx=(0, 8)
        )
        self.weight_layer = tk.StringVar()
        self.weight_selector = ttk.Combobox(
            weight_bar, textvariable=self.weight_layer, state="disabled", width=36,
        )
        self.weight_selector.pack(side="left")
        self.weight_selector.bind("<<ComboboxSelected>>", self.show_weights)
        self.weight_info = tk.StringVar(
            value="Entrena o carga un modelo para visualizar sus conexiones."
        )
        ttk.Label(
            self.net_tab, textvariable=self.weight_info, wraplength=1050,
        ).pack(anchor="w", pady=(0, 4))
        self.weight_plot = ttk.Frame(self.net_tab)
        self.weight_plot.pack(fill="both", expand=True)
        self.values["image_size"].trace_add(
            "write", lambda *_: self.refresh_architecture()
        )
        self.refresh_architecture()

    def architecture_edited(self, event=None):
        if self.arch_editor.edit_modified():
            self.arch_editor.edit_modified(False)
            self.refresh_architecture()

    def refresh_architecture(self):
        if not hasattr(self, "net_text"):
            return
        try:
            n = len(self.entries())
            classes = n if n >= 2 else 10
            spec = json.loads(self.arch_editor.get("1.0", "end"))
            summary = validate_architecture(
                spec, int(self.values["image_size"].get()), classes
            )
            text = "PREPARADA PARA EL PRÓXIMO ENTRENAMIENTO\n"
            if n < 2:
                text += "Vista previa con 10 salidas; selecciona entre 5 y 10 biomas.\n"
            text += summary
            self.arch_notice.set("Arquitectura válida. Se usará al pulsar Entrenar.")
        except Exception as exc:
            text = "CONFIGURACIÓN PENDIENTE DE CORREGIR\n" + str(exc)
            self.arch_notice.set("JSON sin validar: " + str(exc))
        if self.active:
            cp = self.active["checkpoint"]
            text += "\n\nMODELO ACTIVO PARA PREDICCIONES\n" + validate_architecture(
                cp.get("architecture") or ORIGINAL, cp["image_size"], len(cp["classes"])
            )
        self.net_text.configure(state="normal")
        self.net_text.delete("1.0", "end")
        self.net_text.insert("1.0", text)
        self.net_text.configure(state="disabled")

    def refresh_weights(self):
        """Actualiza las capas disponibles cuando cambia el modelo activo."""
        if not self.active:
            self.weight_selector.configure(values=(), state="disabled")
            self.weight_layer.set("")
            return
        cp = self.active["checkpoint"]
        layers = dense_weight_layers(cp["state_dict"])
        labels = [name for name, _, _ in layers]
        previous = self.weight_layer.get()
        self.weight_selector.configure(
            values=labels, state="readonly" if labels else "disabled"
        )
        if not labels:
            self.weight_layer.set("")
            self.weight_info.set("El modelo no contiene capas densas (Linear).")
            return
        self.weight_layer.set(previous if previous in labels else labels[-1])
        self.show_weights()

    def show_weights(self, event=None):
        if not self.active or not self.weight_layer.get():
            return
        cp = self.active["checkpoint"]
        key = self.weight_layer.get()
        try:
            fig = draw_dense_connections(cp["state_dict"][key], key)
            self.render(self.weight_plot, fig)
            self.weight_info.set(
                "Azul: pesos positivos; naranja: negativos. Se muestran "
                "hasta 12 unidades por lado y las 60 conexiones de mayor "
                "magnitud entre ellas. No representa las activaciones ni "
                "demuestra la importancia de una neurona."
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            self.weight_info.set("No se pudieron mostrar los pesos: " + str(exc))

    def use_preset(self):
        self.arch_editor.delete("1.0", "end")
        self.arch_editor.insert(
            "1.0", json.dumps(PRESETS[self.preset.get()], ensure_ascii=False, indent=2)
        )
        self.refresh_architecture()

    def validate_editor(self):
        self.refresh_architecture()
        self.tabs.select(self.net_tab)

    def import_architecture(self):
        path = filedialog.askopenfilename(filetypes=[("Arquitectura JSON", "*.json")])
        if not path:
            return
        try:
            spec = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            validate_architecture(
                spec, int(self.values["image_size"].get()), max(2, len(self.entries()))
            )
            self.arch_editor.delete("1.0", "end")
            self.arch_editor.insert(
                "1.0", json.dumps(spec, ensure_ascii=False, indent=2)
            )
            self.refresh_architecture()
        except Exception as exc:
            messagebox.showerror("Arquitectura", str(exc))

    def export_architecture(self):
        try:
            spec = json.loads(self.arch_editor.get("1.0", "end"))
            validate_architecture(
                spec, int(self.values["image_size"].get()), max(2, len(self.entries()))
            )
            path = filedialog.asksaveasfilename(
                defaultextension=".json", filetypes=[("Arquitectura JSON", "*.json")]
            )
            if path:
                Path(path).write_text(
                    json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except Exception as exc:
            messagebox.showerror("Arquitectura", str(exc))

    def render(self, parent, fig):
        if parent in self.canvases:
            old = self.canvases.pop(parent)
            old.get_tk_widget().destroy()
            old.figure.clear()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().configure(width=1, height=1)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()
        self.canvases[parent] = canvas

    def append_log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def launch(self, fn, kind):
        if self.busy:
            return
        self.busy = True
        self.cancel.clear()
        for control, _ in self.controls:
            control.configure(state="disabled")
        self.stop.configure(state="normal")
        self.status.set("Trabajando...")

        def work():
            try:
                self.events.put((kind, fn()))
            except Cancelled as exc:
                self.events.put(("cancelled", str(exc)))
            except Exception as exc:
                self.events.put(("error", str(exc)))
            finally:
                self.events.put(("idle", None))

        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "idle":
                    self.busy = False
                    for control, state in self.controls:
                        control.configure(state=state)
                    self.stop.configure(state="disabled")
                    self.refresh_selection()
                elif kind == "log":
                    self.append_log(value)
                    self.status.set(value)
                elif kind == "progress":
                    self.progress["value"] = value
                elif kind == "epoch":
                    self.history.append(value)
                    text = f"Época {value['epoch']} | loss {value['loss']:.4f} | acierto {value['accuracy']:.2%}"
                    if "lr" in value:
                        text += f" | LR {value['lr']:.2g}"
                    if "val_macro_f1" in value:
                        text += f" | val F1 macro {value['val_macro_f1']:.2%}"
                    if "val_loss" in value:
                        text += f" | val loss {value['val_loss']:.4f} | val acierto {value['val_accuracy']:.2%}"
                    self.append_log(text)
                    self.status.set(text)
                elif kind == "folders":
                    for item in value:
                        if item["folder"] not in [
                            self.tree.item(i, "values")[3]
                            for i in self.tree.get_children()
                        ]:
                            self.tree.insert(
                                "",
                                "end",
                                values=(
                                    "Sí" if item.get("use", True) else "No",
                                    item["name"],
                                    item["count"],
                                    item["folder"],
                                ),
                            )
                    self.status.set(
                        "Carpetas añadidas. Marca los biomas que quieras entrenar."
                    )
                elif kind in ("trained", "loaded"):
                    self.active = value
                    self.show_model()
                    self.status.set("Modelo listo: " + value["path"])
                    self.tabs.select(self.results_tab)
                elif kind == "prediction":
                    self.show_prediction(value)
                elif kind == "cancelled":
                    self.append_log(value)
                    self.status.set(value)
                elif kind == "error":
                    self.append_log("ERROR: " + value)
                    self.status.set(
                        "Operación fallida. El modelo activo anterior se conserva."
                    )
                    messagebox.showerror("No se pudo completar", value)
        except queue.Empty:
            pass
        self.poll_job = self.after(100, self.poll)

    def refresh_selection(self):
        n = len(self.entries())
        valid = 5 <= n <= 10
        self.selection_info.set(
            f"Seleccionados: {n} · salidas de la red: {n} · "
            + ("Listo para entrenar." if valid else "Elige entre 5 y 10 biomas.")
        )
        self.train_button.configure(
            state="normal" if valid and not self.busy else "disabled"
        )
        self.refresh_architecture()

    def scan(self, folders, selected=True):
        result = []
        for folder in folders:
            if self.cancel.is_set():
                raise Cancelled("Exploración cancelada.")
            n = len(image_files(folder))
            if n:
                result.append(
                    {
                        "folder": str(Path(folder).resolve()),
                        "name": Path(folder).name,
                        "count": n,
                        "use": selected,
                    }
                )
        if not result:
            raise ValueError("No hay imágenes compatibles en esas carpetas.")
        return result

    def add_folder(self):
        path = filedialog.askdirectory(
            title="Selecciona una carpeta que represente UN bioma"
        )
        if path:
            self.launch(lambda: self.scan([path]), "folders")

    def add_root(self):
        path = filedialog.askdirectory(
            title="Selecciona la carpeta que contiene las carpetas de biomas",
            initialdir=str(BASE / "minecraft_biomas")
            if (BASE / "minecraft_biomas").is_dir()
            else str(BASE),
        )
        if path:
            folders = sorted(
                p
                for p in Path(path).iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
            self.launch(lambda: self.scan(folders, selected=False), "folders")

    def toggle(self, event=None):
        if self.busy:
            return
        item = self.tree.identify_row(event.y) if event else self.tree.focus()
        if item:
            values = list(self.tree.item(item, "values"))
            values[0] = "No" if values[0] == "Sí" else "Sí"
            self.tree.item(item, values=values)
            self.refresh_selection()

    def rename(self):
        ids = self.tree.selection()
        if len(ids) != 1:
            return messagebox.showinfo("Renombrar", "Selecciona una sola fila.")
        values = list(self.tree.item(ids[0], "values"))
        name = simpledialog.askstring(
            "Nombre del bioma",
            "Etiqueta visible (no cambia la carpeta):",
            initialvalue=values[1],
        )
        if name and name.strip():
            values[1] = name.strip()
            self.tree.item(ids[0], values=values)

    def remove(self):
        for item in self.tree.selection():
            self.tree.delete(item)
        self.refresh_selection()

    def entries(self, all_rows=False):
        result = []
        for i in self.tree.get_children():
            use, name, count, folder = self.tree.item(i, "values")
            if use == "Sí" or all_rows:
                result.append(
                    {
                        "name": name,
                        "folder": folder,
                        "use": use == "Sí",
                        "count": int(count),
                    }
                )
        return result

    def save_selection(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("Selección", "*.json")]
        )
        if path:
            try:
                Path(path).write_text(
                    json.dumps(self.entries(True), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                messagebox.showerror("Guardar", str(exc))

    def load_selection(self):
        path = filedialog.askopenfilename(filetypes=[("Selección", "*.json")])
        if not path:
            return

        def work():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("La selección debe ser una lista de biomas.")
            result = []
            for e in data:
                if (
                    not isinstance(e, dict)
                    or not isinstance(e.get("name"), str)
                    or not isinstance(e.get("folder"), str)
                ):
                    raise ValueError("Selección inválida.")
                row = self.scan([e["folder"]])[0]
                row.update(name=e["name"], use=e.get("use", True))
                result.append(row)
            return result

        self.launch(work, "folders")

    def start_train(self):
        try:
            params = {
                k: (
                    v.get()
                    if k
                    in (
                        "device",
                        "optimizer",
                        "scheduler",
                        "class_weighting",
                        "selection_metric",
                    )
                    else float(v.get())
                    if k in ("lr", "weight_decay")
                    else int(v.get())
                )
                for k, v in self.values.items()
            }
            spec = json.loads(self.arch_editor.get("1.0", "end"))
            validate_architecture(spec, params["image_size"], len(self.entries()))
            cfg = Config(**params, architecture=spec)
            cfg.validate()
            entries = self.entries()
            if not 5 <= len(entries) <= 10:
                raise ValueError("Marca entre 5 y 10 biomas.")
        except Exception as exc:
            return messagebox.showerror("Configuración", str(exc))
        self.history = []
        self.progress["value"] = 0
        self.append_log(
            "Nuevo entrenamiento desde cero. El test no se consulta hasta terminar."
        )
        self.launch(
            lambda: train(
                entries, cfg, OUTPUT, lambda k, v: self.events.put((k, v)), self.cancel
            ),
            "trained",
        )

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Modelo PyTorch", "*.pth *.pt")])
        if path:
            self.launch(
                lambda: {"checkpoint": load_checkpoint(path), "path": path}, "loaded"
            )

    def show_model(self):
        cp = self.active["checkpoint"]
        metrics = cp.get("metrics")
        history = cp.get("history", [])
        self.render(
            self.curve_plot,
            curves(history, metrics.get("selected_epoch") if metrics else None),
        )
        self.report.delete(*self.report.get_children())
        if self.cm_plot in self.canvases:
            old = self.canvases.pop(self.cm_plot)
            old.get_tk_widget().destroy()
            old.figure.clear()
        if metrics:
            self.summary.set(
                f"Test: {metrics['accuracy']:.2%} · pérdida {metrics['loss']:.4f} · época elegida {metrics['selected_epoch']}\n"
                + " / ".join(f"{k}: {v}" for k, v in metrics["sizes"].items())
                + (
                    "\nF1 macro: "
                    + format(metrics["macro_f1"], ".2%")
                    + " · Balanced accuracy: "
                    + format(metrics["balanced_accuracy"], ".2%")
                    + " · Selección: "
                    + metrics.get("selection_metric", "val_loss")
                    if "macro_f1" in metrics
                    else ""
                )
                + "\n"
                + self.active["path"]
            )
            self.render(self.cm_plot, confusion(metrics))
            for i, name in enumerate(cp["classes"]):
                r = metrics["report"][f"class_{i}"]
                self.report.insert(
                    "",
                    "end",
                    values=(
                        name,
                        f"{r['precision']:.2%}",
                        f"{r['recall']:.2%}",
                        f"{r['f1-score']:.2%}",
                        r["support"],
                    ),
                )
        else:
            self.summary.set(
                "Modelo anterior cargado: permite predecir, pero no contiene historial ni métricas del test.\n"
                + self.active["path"]
            )
        self.data_report.delete(*self.data_report.get_children())
        for i, row in enumerate(cp.get("data_summary", [])):
            weights = cp.get("class_weights")
            self.data_report.insert(
                "",
                "end",
                values=(
                    *(
                        row[k]
                        for k in (
                            "name",
                            "available",
                            "selected",
                            "train",
                            "val",
                            "test",
                        )
                    ),
                    format(weights[i], ".3f") if weights else "1.000",
                ),
            )
        self.refresh_architecture()
        self.refresh_weights()
        self.pred_title.set(
            f"Modelo activo: {len(cp['classes'])} biomas · {cp['image_size']} × {cp['image_size']}"
        )
        self.prob_tree.delete(*self.prob_tree.get_children())
        self.preview.configure(image="", text="Selecciona una captura")

    def choose_image(self):
        if not self.active:
            return messagebox.showinfo("Modelo", "Primero entrena o carga un modelo.")
        path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return
        cp = self.active["checkpoint"]

        def work():
            probs = predict(cp, path, "cuda" if torch.cuda.is_available() else "cpu")
            with Image.open(path) as im:
                preview = ImageOps.contain(im.convert("RGB"), (480, 480)).copy()
            return {"path": path, "probs": probs, "preview": preview}

        self.launch(work, "prediction")

    def show_prediction(self, value):
        probs = sorted(value["probs"], key=lambda p: -p[1])
        self.prob_tree.delete(*self.prob_tree.get_children())
        for name, p in probs:
            self.prob_tree.insert("", "end", values=(name, f"{100 * p:.2f} %"))
        self.photo = ImageTk.PhotoImage(value["preview"])
        self.preview.configure(image=self.photo, text="")
        self.pred_title.set(
            f"Predicción: {probs[0][0]} · {probs[0][1]:.2%}\n{value['path']}"
        )
        self.status.set(
            "Imagen clasificada. Distribución softmax sobre los biomas del modelo."
        )

    def export_plots(self):
        if not self.active:
            return messagebox.showinfo(
                "Resultados", "Primero carga o entrena un modelo."
            )
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = filedialog.askdirectory(
            title="Destino de las gráficas", initialdir=str(OUTPUT)
        )
        if not path:
            return
        try:
            cp = self.active["checkpoint"]
            dest = Path(path)
            (dest / "red_neuronal.txt").write_text(
                validate_architecture(
                    cp.get("architecture") or ORIGINAL,
                    cp["image_size"],
                    len(cp["classes"]),
                ),
                encoding="utf-8",
            )
            if cp.get("history"):
                curves(
                    cp["history"], cp.get("metrics", {}).get("selected_epoch")
                ).savefig(dest / "curvas.png", dpi=150)
            if cp.get("metrics"):
                confusion(cp["metrics"]).savefig(dest / "matriz_confusion.png", dpi=150)
            self.status.set("Gráficas exportadas a " + path)
        except Exception as exc:
            messagebox.showerror("Exportación", str(exc))

    def destroy(self):
        if getattr(self, "poll_job", None):
            self.after_cancel(self.poll_job)
            self.poll_job = None
        super().destroy()

    def close(self):
        if self.busy:
            self.cancel.set()
            self.status.set("Cancelando; vuelve a cerrar cuando termine la operación.")
            return
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
