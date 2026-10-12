import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RUTA_DATOS = "data/bank-additional/bank-additional-full.csv"

TRACKING_URI = "sqlite:///mlflow.db"


def leer_argumentos():
    parser = argparse.ArgumentParser(
        description="Entrena una regresión logística sobre Bank Marketing y registra la corrida en MLflow."
    )
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverso de la regularización: menor valor = modelo más simple.")
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="none",
                        help="'balanced' da más peso a la clase minoritaria (los 'sí').")
    parser.add_argument("--solver", choices=["lbfgs", "liblinear", "newton-cg"], default="lbfgs",
                        help="Algoritmo de optimización.")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Máximo de iteraciones del optimizador.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para la división de datos y el modelo.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fracción de datos reservada para prueba.")
    parser.add_argument("--experiment_name", default="bank-marketing-logreg-tuning",
                        help="Nombre del experimento en MLflow.")
    parser.add_argument("--data_path", default=RUTA_DATOS,
                        help="Ruta al archivo CSV.")
    return parser.parse_args()


def cargar_datos(ruta):
    df = pd.read_csv(ruta, sep=";")
    return df


def limpiar_datos(df):
    df = df.drop_duplicates()

    df = df.drop(columns=["duration", "pdays"])

    df["y"] = df["y"].map({"yes": 1, "no": 0})
    return df


def dividir_datos(df, test_size, seed):
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def construir_modelo(X, params, seed):
    num_features = X.select_dtypes(include="number").columns.tolist()
    cat_features = X.select_dtypes(exclude="number").columns.tolist()

    preprocesador = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ])

    modelo = Pipeline([
        ("prep", preprocesador),
        ("clf", LogisticRegression(**params, random_state=seed)),
    ])
    return modelo


def evaluar(modelo, X_test, y_test):
    y_proba = modelo.predict_proba(X_test)[:, 1]

    y_pred = modelo.predict(X_test)

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    ruta_matriz = "matriz_confusion.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["no", "sí"], cmap="Blues", ax=ax
    )
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig(ruta_matriz, dpi=120)
    plt.close(fig)

    return metricas, ruta_matriz


def main():
    args = leer_argumentos()

    params = {
        "C": args.C,
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "solver": args.solver,
        "max_iter": args.max_iter,
    }

    df = cargar_datos(args.data_path)
    df = limpiar_datos(df)
    X_train, X_test, y_train, y_test = dividir_datos(df, args.test_size, args.seed)
    print("Train:", X_train.shape, "| Test:", X_test.shape)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(args.experiment_name)

    nombre_corrida = f"C={args.C}_cw={args.class_weight}_{args.solver}"

    with mlflow.start_run(run_name=nombre_corrida) as run:
        for nombre, valor in params.items():
            mlflow.log_param(nombre, valor)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("modelo", "LogisticRegression")
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        modelo = construir_modelo(X_train, params, args.seed)
        modelo.fit(X_train, y_train)

        metricas, ruta_matriz = evaluar(modelo, X_test, y_test)
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)
            print(f"{nombre:>10}: {valor:.4f}")

        mlflow.log_artifact(ruta_matriz)

        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo",
            input_example=X_train.head(5),
        )

        print("Corrida registrada:", nombre_corrida, "| ID:", run.info.run_id)


if __name__ == "__main__":
    main()
