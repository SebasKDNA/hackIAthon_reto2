from pathlib import Path
import json
import joblib
import pandas as pd

# # empezamos con el proceso de predicción
# MAPA = {0: "Bajo", 1: "Medio", 2: "Alto"}

# def cargar_artefactos_svm(model_dir="models"):
#     clf = joblib.load(f"{model_dir}/modelo_svm.joblib")
#     scaler = joblib.load(f"{model_dir}/scaler.joblib")
#     with open(f"{model_dir}/feature_columns.json", "r", encoding="utf-8") as f:
#         feature_cols = json.load(f)
#     # por si acaso, aseguramos que 'expediente' no esté en features
#     feature_cols = [c for c in feature_cols if c.lower() != "expediente"]
#     return clf, scaler, feature_cols

# 2) Predicción por expediente
def predecir_por_expediente(expediente: int):
    # ----- Rutas absolutas basadas en ESTE archivo (funtions.py) -----
    base_dir   = Path(__file__).resolve().parent          # .../reto_2
    data_dir   = base_dir / "data"
    models_dir = base_dir / "models"

    df_fin_path     = data_dir / "df_score.csv"
    df_bi_path      = data_dir  / "bi_ranking.csv"
    model_path      = models_dir / "modelo_svm.joblib"
    scaler_path     = models_dir / "scaler.joblib"
    feat_cols_path  = models_dir / "feature_columns.json"

    # ----- Existencia de archivos clave (errores claros) -----
    if not df_fin_path.exists():
        raise FileNotFoundError(f"No existe el CSV de puntajes: {df_fin_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo SVM: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No existe el scaler: {scaler_path}")
    if not feat_cols_path.exists():
        raise FileNotFoundError(f"No existe feature_columns.json: {feat_cols_path}")

    # 1) Cargar datos
    col_expediente = "expediente"
    df_fin = pd.read_csv(df_fin_path, encoding="utf-8")

    if df_bi_path.exists():
        df_bi = pd.read_csv(
            df_bi_path,
            usecols=[col_expediente],
            dtype={col_expediente: "int64"},
            sep=",",
            encoding="utf-8"
        )
    else:
        # Si no existe, mantenemos el flujo original pero sin romper
        df_bi = pd.DataFrame(columns=[col_expediente])

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Asegurar tipo numérico del expediente recibido
    exp_int = int(expediente)

    with open(feat_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # # por si acaso, quitar 'expediente' de features
    # feature_cols = [c for c in feature_cols if c != col_expediente]

    # 2) Buscar expediente
    if df_fin[col_expediente].dtype.kind not in ("i", "u"):
        df_fin[col_expediente] = pd.to_numeric(df_fin[col_expediente], errors="coerce").astype("Int64")

    fila = df_fin[df_fin[col_expediente] == exp_int]
    if fila.empty:
        if not df_bi.empty and (df_bi[col_expediente] == exp_int).any():
            return {"status": "no_pyme", "msg": "Su compañía no es PYME"}
        return {"status": "no_expediente", "msg": "Alerta: no existe expediente en la Super de Compañías"}
    # Verificar que todas las columnas del modelo existan
    missing = [c for c in feature_cols if c not in fila.columns]
    if missing:
        return {
            "status": "missing_features",
            "msg": f"Faltan columnas en el CSV: {missing[:10]}",
            "features": {},
            "pred_num": None,
            "pred_texto": None,
            "total_score": float(fila.iloc[0]["score_final"]) if "score_final" in fila.columns else None,
        }
    
    # 3) Construir X SOLO con las columnas del modelo (en ese orden)
    X = fila.loc[:, feature_cols].astype(float)
    X_scaled = scaler.transform(X)

    print(list(X.columns)); print(feature_cols)

    # 4) Predicción
    pred = int(model.predict(X_scaled)[0])

    # Para la gráfica: valores normalizados (los de df_score), no escalados por StandardScaler
    features_plot = {c: float(fila.iloc[0][c]) for c in feature_cols}

    total_score = float(fila.iloc[0]["score_final"]) if "score_final" in fila.columns else None

    MAPA = {0: "Bajo", 1: "Medio", 2: "Alto"}
    return {
        "status": "ok",
        "pred_num": pred,
        "pred_texto": MAPA[pred],
        "features": features_plot,
        "total_score": total_score
    }
