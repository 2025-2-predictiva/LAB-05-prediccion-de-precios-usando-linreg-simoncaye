#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import json
import gzip
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

def cargar_datos(train_path: str, test_path: str):
    df_train = pd.read_csv(train_path, compression="zip")
    df_test = pd.read_csv(test_path, compression="zip")
    return df_train, df_test

def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    df.drop(columns=["Year", "Car_Name"], inplace=True)
    return df

def separar_xy(df: pd.DataFrame):
    X = df.drop(columns="Present_Price")
    y = df["Present_Price"]
    return X, y

def construir_pipeline(categoricas, numericas) -> Pipeline:
    preprocesador = ColumnTransformer([
        ("cat", OneHotEncoder(), categoricas),
        ("num", MinMaxScaler(), numericas)
    ])

    modelo = Pipeline(steps=[
        ("preprocesador", preprocesador),
        ("selector", SelectKBest(score_func=f_regression)),
        ("regresor", LinearRegression())
    ])

    return modelo

def ajustar_modelo(pipeline: Pipeline, X, y):
    parametros = {
        "selector__k": range(1, 15),
        "regresor__fit_intercept": [True, False],
        "regresor__positive": [True, False]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    return grid.fit(X, y)

def guardar_modelo(modelo, ruta: Path):
    """Guarda el modelo entrenado en un archivo comprimido .pkl.gz."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ruta, "wb") as f:
        pickle.dump(modelo, f)

def evaluar_modelo(modelo, X_train, y_train, X_test, y_test, ruta_salida: Path):
    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    met_train = {
        "type": "metrics",
        "dataset": "train",
        "r2": float(r2_score(y_train, pred_train)),
        "mse": float(mean_squared_error(y_train, pred_train)),
        "mad": float(median_absolute_error(y_train, pred_train)),
    }

    met_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": float(r2_score(y_test, pred_test)),
        "mse": float(mean_squared_error(y_test, pred_test)),
        "mad": float(median_absolute_error(y_test, pred_test)),
    }

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(json.dumps(met_train) + "\n")
        f.write(json.dumps(met_test) + "\n")

def main():
    df_train, df_test = cargar_datos(
        "files/input/train_data.csv.zip",
        "files/input/test_data.csv.zip"
    )

    df_train = preprocesar(df_train)
    df_test = preprocesar(df_test)

    X_train, y_train = separar_xy(df_train)
    X_test, y_test = separar_xy(df_test)

    cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    pipeline = construir_pipeline(cat_cols, num_cols)

    mejor_modelo = ajustar_modelo(pipeline, X_train, y_train)

    guardar_modelo(mejor_modelo, Path("files/models/model.pkl.gz"))

    evaluar_modelo(
        mejor_modelo,
        X_train, y_train,
        X_test, y_test,
        Path("files/output/metrics.json")
    )

if __name__ == "__main__":
    main()