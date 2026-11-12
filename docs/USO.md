# Manual de uso

Instalación y ejecución: [README](../README.md). Arquitecturas: [ARQUITECTURAS](ARQUITECTURAS.md).

## 1. Seleccionar datos a mano

- **Añadir bioma…**: elige una carpeta que contenga las imágenes de UN bioma. Busca imágenes también en subcarpetas.
- **Explorar carpeta raíz…**: añade sus subcarpetas inmediatas como candidatos a bioma, inicialmente sin marcar para que elijas los que te interesan. Con el instalador, la carpeta raíz propuesta es `minecraft_biomas/minecraft_biomas`, con nombres de bioma ya convertidos. Si usas datos originales fuera del proyecto, elige `preprocessed_data`.
- Doble clic en una fila para cambiar **Usar: Sí/No**. Solo las filas con Sí entran en el modelo. Puedes quitar filas sin eliminar archivos.
- **Renombrar** cambia la etiqueta visible: por ejemplo, `biome_1` → `Llanura`, siempre que hayas comprobado que ese ID representa ese bioma. No cambia nombres en disco ni combina carpetas.
- Escoge entre **5 y 10 biomas** entre todos los disponibles. El contador muestra cuántos has marcado y Train solo se habilita dentro de ese intervalo. La última capa se crea automáticamente con exactamente ese número de salidas. No se seleccionan automáticamente los seis mayores.
- **Guardar selección** almacena rutas y nombres en JSON; **Cargar selección** añade carpetas que no estén ya en la lista. Si quieres sustituir la selección, quita primero las filas anteriores. Las rutas son locales: si mueves carpetas, vuelve a seleccionarlas.

Formatos: PNG, JPG/JPEG, BMP y WebP. Se requieren al menos 10 imágenes distintas por clase; para resultados útiles se necesitan bastantes más y variedad real. Las clases deben tener nombres únicos y carpetas sin solapamiento. Los nombres elegidos se guardan con el modelo y se usan al predecir.

## 2. Train y configuraciones

Modifica épocas, batch, resolución (32/64/96/128), learning rate, semilla, validación, dispositivo y las opciones de entrenamiento descritas abajo. La opción CUDA requiere una instalación NVIDIA operativa; no cambia silenciosamente a CPU.

- **Test: siempre 20 % del conjunto de los biomas seleccionados**, estratificado, consultado solo al terminar.
- **Validación: 0**: 80 % entrenamiento, 0 % validación, 20 % test. Se conserva el modelo de la última época.
- **Validación: 16**: aproximadamente 64 % entrenamiento, 16 % validación, 20 % test. Se evalúa validación por época y se conserva el modelo según el criterio elegido: menor pérdida (`val_loss`) o mayor F1 macro (`val_macro_f1`). En empates se conserva la primera época que obtuvo el mejor valor. No se usa el test para escogerlo. Hay redondeos a imágenes enteras.
- Para experimentar con configuraciones, utiliza validación y reserva el test para la evaluación final. Mirar repetidamente el test para elegir parámetros sesga su interpretación.
- Cada Train empieza desde cero, no continúa un entrenamiento anterior. Cargar un modelo sirve para inferencia y visualizar resultados, no para reanudar Adam.
- Antes de dividir se retiran copias con píxeles RGB idénticos. Duplicados con etiquetas distintas o imágenes corruptas se notifican como error para corregir el dataset. No se modifican las imágenes originales.
- La interfaz mantiene la respuesta mientras entrena; muestra barra de progreso y métricas por época. Cancelar se comprueba entre imágenes/lotes. Una operación cancelada no reemplaza el modelo activo.

La misma semilla y los mismos archivos/orden de clases reproducen la separación. Cambiar el orden de clases, archivos o clases seleccionadas puede cambiarla. La reproducibilidad numérica entre GPU, CPU y versiones no está garantizada. La detección de duplicados exactos no impide que imágenes casi iguales queden a ambos lados; si tienes secuencias o mundos identificados, utiliza una división por grupos para una evaluación más exigente.

## 3. Resultados y modelos

La pestaña Resultados muestra accuracy y loss de test, tamaños de particiones, época elegida y criterio, F1 macro y balanced accuracy, curvas (y validación si existe), matriz de confusión y precision/recall/F1 por bioma. Las curvas aparecen al terminar; las métricas por época se ven en el registro durante Train.

Cada entrenamiento completado crea una carpeta única en `OutPut`, en la raíz del repositorio. `resultados` conserva los entrenamientos históricos publicados:

- `biomas_cnn.pth`: pesos, orden/nombres de clases, resolución, configuración, historial y métricas.
- `resultados.json`: configuración, carpetas originales, historial y resultados.
- `particiones.csv`: imágenes, etiquetas, hashes y particiones, para auditar la separación.

**Cargar modelo / resultados…** recupera todo desde el `.pth` sin tener las imágenes originales. **Exportar gráficas…** propone `OutPut` y guarda curvas y matriz en PNG, y el resumen de red en TXT, en la carpeta elegida (los nombres de esos PNG se sobrescriben si exportas de nuevo ahí).

También se admite tu `biomas_cnn.pth` de la versión anterior: permite predecir con sus etiquetas originales, pero ese archivo no contiene curvas ni resultados del test. No es posible recuperar métricas ausentes a partir de los pesos. Carga solamente modelos de procedencia conocida; se usa `weights_only=True`.

### Visualización de pesos entre neuronas

En la pestaña **Red neuronal**, después de entrenar o cargar un checkpoint `.pth`, aparece un selector de capas densas (`Linear`). La vista dibuja las conexiones aprendidas de la capa seleccionada:

- **Azul**: peso positivo; **naranja**: peso negativo. El grosor y la opacidad corresponden a la magnitud absoluta del peso respecto a los mostrados.
- Para evitar una gráfica ilegible en capas grandes, se seleccionan como máximo **12 entradas y 12 salidas** según su peso absoluto máximo y se representan hasta **60 conexiones** entre ellas. El gráfico indica los índices de neuronas originales (`E` y `S`) y el número real de unidades de la capa.
- La vista **no modifica el modelo** ni necesita las imágenes originales: lee los tensores de `state_dict` del checkpoint. No se muestran aquí filtros convolucionales (`Conv2d`) ni la evolución histórica de pesos entre épocas.
- **Importante:** el valor o signo de un peso individual no demuestra por sí mismo la importancia causal de una neurona, ni equivale a su activación.

## 4. Imagen propia

Entrena o carga un modelo, abre **Probar imagen** y pulsa **Seleccionar imagen…**. Se presenta la captura, el bioma más probable y el porcentaje de **cada** clase, ordenados de mayor a menor. La distribución Softmax suma 100 % antes de redondear. Se reutilizan la resolución y normalización del modelo, aunque los controles de entrenamiento hayan cambiado.

Estos porcentajes no son una certeza calibrada. El clasificador siempre elige entre sus clases y no dispone de una clase «desconocido»; una captura ajena al dataset puede recibir un porcentaje alto.

## Red y organización del código

- `app.py`: ventana, selección de archivos, trabajos en segundo plano y presentación.
- `core.py`: datos, separación, ConvNet, entrenamiento, evaluación, checkpoints e inferencia. No ejecuta entrenamiento al importarlo.
- `plots.py`: curvas y matriz de confusión.
- `architectures.py`: valida y construye las arquitecturas JSON; la pestaña Red neuronal presenta un resumen textual.
- `tests/test_core.py`: comprobaciones con datos sintéticos.

Arquitectura: RGB → Conv(3,16,5) + ReLU + MaxPool(2) → Conv(16,32,5) + ReLU + MaxPool(2) → Flatten → Linear(8192,120) + ReLU → Linear(120,N). Padding 2, stride 1 en ambas convoluciones. CrossEntropyLoss recibe logits; Softmax se aplica solo para mostrar porcentajes. El optimizador elegido (Adam, AdamW o SGD) actualiza pesos. N es el número de clases seleccionadas, entre 5 y 10. El límite se aplica a nuevos entrenamientos; los modelos antiguos siguen pudiendo cargarse para inferencia.

## Verificación

Se prueba en CPU con imágenes sintéticas: selección y duplicados, particiones disjuntas/repetibles, entrenamiento con validación, evaluación, carga/guardado, formato anterior, probabilidades, cancelación y tamaños de entrada. Esas pruebas no miden la calidad del clasificador sobre Minecraft. También se verificó la interfaz en un escritorio virtual Linux: selección, renombrado, entrenamiento en segundo plano, resultados, predicción y recarga de un modelo. No se ha entrenado esta versión en una GPU NVIDIA ni probado la ventana en Windows en este entorno.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Fuentes

Apuntes y estructura didáctica: https://cayetanoguerra.github.io/ia/ (Tutorial PyTorch 8).
Dataset original, para obtenerlo manualmente: https://www.kaggle.com/datasets/willowc/minecraft-biomes .
Tkinter: https://docs.python.org/3/library/tkinter.html .
PyTorch, guardado y carga: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html .


## Nuevas opciones: límites, desequilibrio y aprendizaje

CUDA sigue siendo el dispositivo de cálculo, no el algoritmo. Puedes combinar CUDA con cualquiera de los optimizadores.

| Control | Opciones / significado |
|---|---|
| Optimizador | `Adam`, `AdamW`, `SGD` (momentum fijo de 0.9). Cada algoritmo puede requerir un learning rate distinto; ninguno garantiza mejorar todos los datasets. |
| Plan LR | `none`: LR constante; `plateau`: ReduceLROnPlateau, reduce a la mitad cuando deja de mejorar la métrica elegida de validación; `cosine`: CosineAnnealingLR, descenso coseno a lo largo de las épocas. |
| Paciencia LR | Solo para `plateau`. Número de épocas malas toleradas antes de reducir; con 2, reduce tras superar dos épocas sin mejora significativa. Mínimo LR: el menor entre 1e-6 y el LR inicial. |
| Weight decay | Regularización del optimizador; 0 desactiva. AdamW utiliza decaimiento desacoplado. |
| Pesos CE | `none`: todas las imágenes pesan igual; `balanced`: CrossEntropyLoss con pesos inversos a la frecuencia efectiva de cada clase de **entrenamiento**. |
| Máx. imágenes/bioma | `0`: sin límite; un entero ≥10: máximo de imágenes útiles que se utilizará por bioma. |
| Elegir modelo por | `val_loss`: menor CE sin ponderar; `val_macro_f1`: mayor F1 macro de validación, dando igual peso a cada bioma. Sin validación, solo se permite `val_loss` y se conserva la última época. |

La interfaz propone como punto de partida: 30 épocas, batch 64, tamaño 64, LR 0.001, semilla 42, validación 16, AdamW, plateau, weight decay 0.0001, pesos balanced, sin límite y selección por val_macro_f1. No son valores óptimos comprobados en Minecraft. Para reproducir el algoritmo anterior usa Adam, none, weight decay 0, pesos none y val_loss, con los mismos datos, semilla y orden de clases. Para desactivar validación pon también Plan LR `none` o `cosine` y criterio `val_loss`.

### Cómo se aplica el máximo

1. Lee y comprueba todas las imágenes de los biomas seleccionados; retira duplicados exactos. Esta comprobación inicial sigue recorriendo los archivos aunque el máximo sea pequeño.
2. Elige sin reemplazo `min(máximo, imágenes útiles)` por clase mediante la semilla; no toma simplemente los primeros nombres de archivo.
3. Separa esas imágenes en entrenamiento/validación/test. El **20 % reservado se calcula sobre ese conjunto reducido**, con redondeo y estratificación.
4. Calcula los pesos usando exclusivamente los recuentos de entrenamiento, después de limitar y dividir.

Ejemplo: máximo 100, una clase con 3000 imágenes útiles aporta 100 y otra con 86 aporta sus 86. Las clases pequeñas no se duplican, las restantes imágenes quedan fuera de esa ejecución y ningún archivo se borra o modifica. Se conserva el mínimo existente de 10 imágenes útiles por clase.

La pestaña **Datos utilizados** muestra por bioma: útiles originales, tras límite, train/validación/test y peso CE. Estos datos y los pesos se guardan en el checkpoint y en `resultados.json`; `particiones.csv` contiene exactamente las imágenes utilizadas. El peso es `N_train / (n_clases * N_train_de_esa_clase)`.

Cambiar el máximo cambia el conjunto evaluado y posiblemente qué imágenes pertenecen al test: **no compares directamente accuracies de tests distintos como si fueran la misma prueba**. Para comparar optimizadores conserva máximo, datos, clases, orden y semilla, y elige usando validación. Un máximo bajo descarta datos útiles; 0 con pesos balanced es una alternativa al recorte. Activar ambos es válido: si ya quedaron equilibrados, los pesos serán cercanos a 1.

### Nuevas curvas y métricas

Se muestran cuatro curvas: CE sin ponderar, accuracy, F1 macro y learning rate utilizado por época. Una línea vertical marca el modelo seleccionado. El test se usa una sola vez al final y nunca controla el LR ni la selección de época.

Cuando activas pesos, el gradiente utiliza la CE ponderada, pero la curva de pérdida muestra **CE sin ponderar en train y validación** para que sean comparables. El objetivo ponderado se registra por separado como `objective_loss` en el historial del checkpoint/JSON; se agrega con su denominador de pesos correcto. Las métricas de entrenamiento se acumulan sobre las predicciones de los lotes mientras se actualiza el modelo; validación se calcula con pesos fijos al terminar cada época.

El resumen final incluye F1 macro y balanced accuracy (media de recall por clase), además de accuracy, loss y métricas por bioma. Los modelos antiguos siguen funcionando; las nuevas curvas o resúmenes que no tengan guardados no se inventan.

Las pruebas añadidas cubren el caso 3000/86 con límite 100, ausencia de duplicación, selección reproducible, pesos calculados solo desde train, normalización de CE ponderada, los tres optimizadores, reducción de LR y selección por F1. No se ha medido una mejora de precisión real en Minecraft con esta versión.

Referencias de las opciones: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html , https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html , https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html .

