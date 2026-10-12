# TC5061 – Experimentación con MLflow

**Alumno:** Diego Cameros (A01840280)
**Materia:** TC5061 Operaciones de Aprendizaje Automático – Maestría en Inteligencia Artificial Aplicada, Tecnológico de Monterrey

Script de entrenamiento parametrizado e instrumentado con MLflow Tracking. Entrena una regresión logística sobre el dataset *Bank Marketing* (UCI) para predecir si un cliente contratará un depósito a plazo.

## Estructura

```
├── data/bank-additional/bank-additional-full.csv   Dataset (UCI Bank Marketing)
├── notebooks/                                      Notebook baseline (solo exploración)
├── train.py                                        Script de entrenamiento
├── requirements.txt                                Dependencias con versiones
└── README.md
```

## Instalación

Requiere Python 3.11.

```bash
git clone <URL-del-repositorio>
cd <carpeta-del-repositorio>
pip install -r requirements.txt
```

## Ejecución

Con los valores por defecto (reproduce el baseline):

```bash
python train.py
```

Con hiperparámetros distintos:

```bash
python train.py --C 0.1 --class_weight balanced
```

### Argumentos

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la regularización. Un valor menor da un modelo más simple. |
| `--class_weight` | `none` | `none` o `balanced`. `balanced` da más peso a la clase minoritaria. |
| `--solver` | `lbfgs` | `lbfgs`, `liblinear` o `newton-cg`. |
| `--max_iter` | `1000` | Máximo de iteraciones del optimizador. |
| `--seed` | `42` | Semilla aleatoria (división de datos y modelo). |
| `--test_size` | `0.2` | Fracción de datos para prueba. |
| `--experiment_name` | `bank-marketing-logreg-tuning` | Nombre del experimento en MLflow. |
| `--data_path` | `data/bank-additional/bank-additional-full.csv` | Ruta al CSV. |

`python train.py --help` muestra esta misma lista.

## Qué registra cada corrida en MLflow

- **Parámetros:** `C`, `class_weight`, `solver`, `max_iter`, `seed`, `test_size`, `modelo`, `n_train`, `n_test`.
- **Métricas:** `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (sobre el conjunto de prueba).
- **Artefacto:** `matriz_confusion.png`.
- **Modelo:** el pipeline completo (preprocesamiento + clasificador) con `mlflow.sklearn.log_model`.

## Ver y comparar las corridas

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir http://127.0.0.1:5000, entrar al experimento `bank-marketing-logreg-tuning`, seleccionar las corridas y presionar **Compare**.

## Corridas ejecutadas

Todas con `--seed 42` y `--test_size 0.2`.

| Comando | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| `python train.py` | 0.8980 | 0.6507 | 0.2047 | 0.3115 | 0.8003 |
| `python train.py` (repetida) | 0.8980 | 0.6507 | 0.2047 | 0.3115 | 0.8003 |
| `python train.py --C 0.01` | 0.8987 | 0.6911 | 0.1832 | 0.2896 | 0.7979 |
| `python train.py --C 100` | 0.8980 | 0.6517 | 0.2037 | 0.3103 | 0.8002 |
| `python train.py --class_weight balanced` | 0.8294 | 0.3576 | 0.6455 | 0.4602 | 0.8001 |
| `python train.py --C 0.01 --class_weight balanced` | 0.8201 | 0.3439 | 0.6573 | 0.4515 | 0.7991 |
| `python train.py --C 0.1 --class_weight balanced` | 0.8284 | 0.3561 | 0.6466 | 0.4592 | 0.8000 |
| `python train.py --class_weight balanced --solver liblinear` | 0.8294 | 0.3576 | 0.6455 | 0.4602 | 0.8001 |

## Reproducibilidad

La semilla (`--seed`) fija la división entrenamiento/prueba y el estado aleatorio del modelo. Dos ejecuciones con los mismos argumentos producen las mismas métricas (ver las dos primeras filas de la tabla). Las versiones de las dependencias están fijadas en `requirements.txt`.

## Datos

Moro, S., Rita, P. y Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306 (licencia CC BY 4.0).

Limpieza aplicada en `train.py`: se eliminan 12 filas duplicadas y las columnas `duration` (fuga de información) y `pdays` (código 999 inconsistente); los valores `unknown` se conservan como categoría.
