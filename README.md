# Minecraft Neural

Aplicación de escritorio para clasificar biomas de Minecraft con Python,
Tkinter y PyTorch. Entrena en el ordenador del usuario con CPU o CUDA NVIDIA;
permite configurar arquitecturas JSON, comparar métricas y guardar modelos.

## Instalar desde cero en Windows con PyCharm

Esta guía está pensada para alguien que recibe **solo el enlace a GitHub** (por ejemplo,
un profesor). No necesita copiar archivos ni entornos virtuales de otro ordenador.

### 1. Instalar los requisitos antes de clonar

1. Instala [Python 3.12 de 64 bits](https://www.python.org/downloads/) con **pip**
   y **Tcl/Tk (Tkinter)**. Python 3.11–3.13 también se admite, pero se recomienda 3.12.
2. Instala [Git para Windows](https://git-scm.com/downloads/win).
3. Instala [Git LFS](https://git-lfs.com/) **antes de clonar**. Git LFS es necesario
   para descargar los modelos históricos `.pth` y las imágenes publicadas mediante LFS.
   En PowerShell ejecuta una vez:

   ``powershell
   git lfs install
   ``

4. Instala [PyCharm](https://www.jetbrains.com/pycharm/download/).
5. Si quieres entrenar con **CUDA**, necesitas una GPU NVIDIA con controlador compatible
   y actualizado. **Sin GPU NVIDIA también funciona**: se instalará PyTorch para CPU.

> Los modelos históricos de Git LFS y las dependencias pueden requerir varios GB.
> Para preparar el dataset se necesita además al menos 1 GiB de espacio temporal.

### 2. Clonar desde PyCharm

1. Abre PyCharm y selecciona **Clone Repository** / **Get from VCS**
   (si ya tienes otro proyecto abierto: **Git → Clone**).
2. En **URL** pega:

   ``text
   https://github.com/Pabloflopygame/Minecraft_Neural.git
   ``

3. Elige una carpeta local vacía (por ejemplo
   `C:\Users\Profesor\PycharmProjects\Minecraft_Neural`) y pulsa **Clone**.
   El repositorio es público y no exige iniciar sesión en GitHub para leerlo.
4. Abre el proyecto clonado y, si PyCharm propone crear un entorno virtual,
   **no es necesario crearlo manualmente**: `setup.py` creará `.venv`.

Una vez clonado, en la terminal de PyCharm (situada **en la raíz del proyecto**)
comprueba que Git LFS ha terminado de descargar los modelos:

```powershell
git lfs pull
```

Si prefieres **entrenar desde cero** y no necesitas cargar los modelos históricos,
puedes omitir `git lfs pull`; para cargar uno de esos `.pth` más adelante sí
deberás descargarlo. Un archivo `.pth` de unos pocos bytes es solo un puntero
LFS, **no** un modelo cargable.

### 3. Instalar y abrir la aplicación

Abre **Terminal** dentro de PyCharm. Debes ver una ruta terminada en
`Minecraft_Neural>`. Comprueba Python:

```powershell
py -3.12 --version
```

Después ejecuta **desde la raíz**, donde están `README.md` y `setup.py`:

```powershell
py -3.12 setup.py
```

La primera ejecución descarga y comprueba el dataset público, crea `.venv`,
instala PyTorch/torchvision y las dependencias de `minecraft_biomas/requirements.txt`,
verifica el entorno y **abre automáticamente la aplicación gráfica**.
No hace falta ejecutar `pip install` manualmente. La descarga total puede ser
de varios GB; las ejecuciones posteriores reutilizan el entorno y los datos.

Si quieres forzar CPU aunque tengas NVIDIA:

```powershell
py -3.12 setup.py --cpu
```

Si el dataset no se puede descargar de Kaggle, consulta la sección
[Dataset reproducible](#dataset-reproducible): puedes descargarlo desde la web
e importarlo con `--dataset-source`. Para abrir la app sin descargar datos:

```powershell
py -3.12 setup.py --skip-dataset
```

> `setup.py` es un **instalador y lanzador**, no un paquete setuptools:
> no uses `pip install .`.

### 4. Seleccionar el intérprete correcto en PyCharm

Cuando termine el instalador, ve a **Settings → Project → Python Interpreter**,
elige **Add Interpreter → Add Local Interpreter → Existing** (el texto puede variar
entre versiones de PyCharm) y selecciona el Python ya creado:

```text
<carpeta del repositorio>\.venv\Scripts\python.exe
```

**No** selecciones el Python global ni crees otro `venv`: `.venv` es el entorno
utilizado por `setup.py`. Para reabrir la aplicación más adelante, usa de nuevo
`py -3.12 setup.py` en la terminal; si instalaste el modo CPU con `--cpu`,
vuelve a indicar `--cpu`.

### 5. Comprobar que funciona

En la ventana **Biomas de Minecraft** tienes dos formas de comprobarlo:

- **Entrenar desde cero:** pulsa **Explorar carpeta raíz…**, acepta la ruta del
  dataset preparado, activa entre **5 y 10 biomas**, configura el entrenamiento
  y comienza. El cálculo se realiza **localmente** en CPU o CUDA del ordenador.
- **Cargar un modelo ya entrenado:** pulsa **Cargar modelo / resultados…**
  y escoge un `biomas_cnn.pth` de `minecraft_biomas/resultados/`. Esto permite
  ver resultados y probar imágenes sin volver a entrenar; **requiere haber
  descargado esos modelos con Git LFS**.

Para iterar sobre una versión, pulsa **Cargar versión…** en **Datos y entrenamiento**.
Se recuperan los parámetros guardados, la arquitectura y las carpetas de los biomas
en el orden de las salidas. Si moviste las imágenes, **Vincular dataset…** permite
seleccionar su nueva raíz. En versiones anteriores conserva `resultados.json`
junto al modelo para recuperar las rutas originales.

Con **Reutilizar pesos** activado, **Entrenar** parte de los pesos de la versión activa.
Las épocas indican cuántas épocas adicionales entrenar. El optimizador y su plan de
learning rate comienzan con los ajustes visibles. Cada iteración se guarda en una
carpeta nueva de `OutPut/`. En **Resultados → Guardar copia como…** puedes conservar
la versión activa con un nombre propio. En un barrido, todas las combinaciones parten de la
misma versión. Para comparar sobre la misma partición, conserva las imágenes, la
semilla, el límite por bioma y el porcentaje de validación.

En **Probar imagen** puedes seleccionar una captura de Minecraft para obtener
la predicción y los porcentajes de las clases del modelo cargado.
Para el resto de opciones consulta el [Manual de uso](docs/USO.md).

Si quieres comprobar además el código con los tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

### 6. Si algo falla

| Síntoma | Qué comprobar |
| --- | --- |
| `py -3.12` no se reconoce | Instala Python 3.12 para Windows y comprueba `py -3.12 --version`. |
| Falta `tkinter` | Reinstala Python incluyendo Tcl/Tk y IDLE. |
| No se detecta CUDA o falla su instalación | Actualiza el driver NVIDIA o prueba `py -3.12 setup.py --cpu`. |
| Error al descargar Kaggle | Usa el ZIP original con `--dataset-source` o abre con `--skip-dataset`. |
| Un `.pth` no carga y ocupa ~130 bytes | No se ha descargado de Git LFS: ejecuta `git lfs pull`. |
| PyCharm marca `torch` como inexistente | Selecciona el intérprete `.venv\Scripts\python.exe`. |

## Opciones del instalador

| Opción tras `py -3.12 setup.py` | Uso |
| --- | --- |
| `--cpu` | Instalar y ejecutar con PyTorch CPU. |
| `--cuda` | Exigir CUDA; informar si no está disponible. |
| `--install-only` | Preparar datos y dependencias sin abrir la ventana. |
| `--dataset-only` | Preparar solo los datos, sin PyTorch ni interfaz. |
| `--dataset-source "C:/Descargas/minecraft-biomes.zip"` | Importar un ZIP local en lugar de descargarlo. |
| `--dataset-source "C:/Datos/preprocessed_data"` | Copiar y renombrar una carpeta local, conservando el origen. |
| `--skip-dataset` | Trabajar con otros datos o solo cargar modelos, sin preparar el dataset. |
| `--reinstall` | Reinstalar dependencias en `.venv`. |

Se pueden combinar opciones, por ejemplo `--dataset-only --dataset-source RUTA`.
Si instalas con `--cpu`, úsalo de nuevo al abrir para mantener esa distribución.

## Dataset reproducible

Fuente: [Minecraft Biomes de Willow C en Kaggle](https://www.kaggle.com/datasets/willowc/minecraft-biomes),
versión 1, licencia CC0. Contiene **22.169 imágenes y 41 biomas**.
La descarga pública se ha probado sin credenciales. Si Kaggle rechaza la descarga,
puedes obtener el ZIP desde su web e importarlo con `--dataset-source`.

El instalador transforma `preprocessed_data/biome_ID` en
`minecraft_biomas/minecraft_biomas/nombre_del_bioma`, conservando exactamente los
nombres publicados anteriormente, incluido `Plains`. No cambia las imágenes ni
sus nombres de archivo. La correspondencia, recuentos y huellas SHA-256 están en
[`scripts/dataset_manifest.json`](scripts/dataset_manifest.json).

La preparación se hace en una carpeta temporal y se valida antes de colocar el
resultado en su destino. Si hay un dataset existente incompleto o modificado,
**no se sobrescribe ni se borra**: muévelo a otra carpeta para regenerarlo, o usa
`--skip-dataset` para seguir trabajando con él. La misma opción permite abrir la
app sin conexión cuando se usan datos propios.

## Documentación y estructura

- [Manual de uso](docs/USO.md): datos, entrenamiento, validación, gráficas y predicción.
- [Arquitecturas JSON](docs/ARQUITECTURAS.md): formato de capas y ejemplos.
- `setup.py`: entrada única para instalar y ejecutar.
- `scripts/`: preparación del dataset, sin dependencias de PyTorch.
- `minecraft_biomas/app.py`: interfaz; `core.py`: datos y entrenamiento;
  `architectures.py`: construcción de redes; `plots.py`: gráficas.
- `tests/`: pruebas de datos, arquitecturas, entrenamiento e instalador.
- `minecraft_biomas/resultados/`: experimentos históricos publicados para ver
  la evolución y cargar redes entrenadas. `archivo_OutPut/` conserva las salidas antiguas.
- `OutPut/`: nuevos entrenamientos y destino inicial para exportar gráficas;
  Git solo guarda `.gitkeep`. El resto son archivos personales ignorados.
- `.venv/` y `minecraft_biomas/minecraft_biomas/`: dependencias y datos locales,
  generados por el instalador y excluidos de Git.

`setup.py` es un instalador/lanzador, no un archivo de empaquetado setuptools.
No se instala el proyecto con `pip install .`.

## Desarrollo y pruebas

```powershell
py -3.12 setup.py --cpu --skip-dataset --install-only
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

En Linux sustituye `.\.venv\Scripts\python.exe` por `.venv/bin/python`.
Se utiliza `pytest` porque recoge tanto las pruebas `unittest` como las funciones
de prueba de arquitecturas. Las pruebas sintéticas de CPU no miden precisión
sobre Minecraft ni validan el rendimiento CUDA. Las pruebas del importador son
offline; la descarga completa se comprueba aparte con `--dataset-only`.

## Actualizar una copia existente

Antes del primer `git pull` que retire los datos de Git, copia fuera del repositorio
tu dataset, `OutPut` y cualquier configuración local que quieras conservar:
Git puede eliminar archivos que antes estaban versionados. Después puedes importar
la copia con `--dataset-source` y devolver tus salidas personales a `OutPut`.
Los modelos históricos siguen publicados. Las selecciones antiguas con rutas
absolutas de otro ordenador requieren volver a elegir las carpetas desde la app.

Los modelos e imágenes históricos usan Git LFS. Sigue la instalación de Git LFS
[antes de clonar](#1-instalar-los-requisitos-antes-de-clonar) y ejecuta `git lfs pull`.
Un ZIP de código de GitHub puede contener punteros LFS en lugar de los modelos.

Eliminar datos del árbol actual no borra los commits antiguos: reducir el tamaño
del historial requeriría una operación separada con copias de seguridad.

