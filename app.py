from __future__ import annotations
import os, io
from flask import Flask, render_template,request, redirect, url_for, session
import pandas as pd
import pdfplumber
import re
import io
import os


from funtions import predecir_por_expediente
from nlp import reviews_sentiment
from utils.scraper import scrape_stats_from_url


APP_DIR = os.path.dirname(os.path.abspath(__file__))
# WEIGHTS = os.path.join(APP_DIR, "weights", "matriz_riesgo.yaml")



app = Flask(__name__)

app.secret_key = os.urandom(24)

# Funcion de PDF

# 1) Función: extraer texto de un PDF (usa pdfplumber)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text.append(t)
    return "\n".join(text)

# 2) Función: parsear Razón Social, Expediente y RUC desde el texto
def parse_certificado(texto: str):
    # Normalizar espacios/saltos
    t = texto.replace("\r", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)

    # 1) Razón social: bloque entre los dos labels
    #    Usamos DOTALL para capturar en varias líneas si fuera el caso
    rs = None
    m_rs = re.search(
        r"RAZ[ÓO]N\s+O\s+DENOMINACI[ÓO]N\s*(?:[:\-]\s*)?(?P<rs>.*?)(?=\s*EXPEDIENTE\s*:)",
        t, flags=re.IGNORECASE | re.DOTALL
    )
    if m_rs:
        rs = m_rs.group("rs")
        rs = re.sub(r"\s+", " ", rs).strip(" .:-")

        # Limpieza de encabezados comunes por si el motor de texto los mezcló
        bad = {"DATOS GENERALES DE LA COMPAÑÍA", "DATOS GENERALES DE LA COMPANIA", "DATOS GENERALES DE LA COMPAÑIA"}
        if rs.upper() in bad:
            rs = None

    # 2) Expediente
    expediente = None
    m_exp = re.search(r"EXPEDIENTE\s*:\s*([0-9]+)", t, flags=re.IGNORECASE)
    if m_exp:
        expediente = m_exp.group(1)

    # 3) RUC
    ruc = None
    m_ruc = re.search(r"RUC\s*:\s*([0-9]{10,13})", t, flags=re.IGNORECASE)
    if m_ruc:
        ruc = m_ruc.group(1)

    return {
        "razon_social": rs,
        "expediente": expediente,
        "ruc": ruc
    }

# 3) Guardar expediente (ejemplo simple: archivo txt; usa tu persistencia preferida)
def save_expediente(expediente: str, path="data/expedientes_guardados.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(expediente + "\n")

# Guardar razón social, expediente y RUC en un archivo de texto
def save_certificado_info(razon_social, expediente, ruc, path="data/certificados_guardados.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rs = (razon_social or "").strip()
    ex = (expediente or "").strip()
    ru = (ruc or "").strip()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{rs}\t{ex}\t{ru}\n")

# 4) Endpoint 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1) Recibir archivo y URL
        file = request.files.get("certificado_pdf")
        social_url = request.form.get("social_url", "").strip()

        # 2) Validaciones mínimas
        if not file or file.filename == "":
            return render_template("index.html", error="Sube el PDF del certificado.")
        if not file.filename.lower().endswith(".pdf"):
            return render_template("index.html", error="El archivo debe ser PDF.")

        # 3) Extraer texto del PDF
        try:
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                texto = "\n".join([(p.extract_text() or "") for p in pdf.pages])
        except Exception as e:
            return render_template("index.html", error=f"No se pudo leer el PDF: {e}")

        # 4) Parsear certificado (usa tu función existente)
        datos = parse_certificado(texto)  # {'razon_social','expediente','ruc'}

        # 5) Guardar en sesión para usar en /result
        session["razon_social"] = datos.get("razon_social")
        session["expediente"]   = datos.get("expediente")
        session["ruc"]          = datos.get("ruc")
        session["social_url"]   = social_url

        # 6) (Opcional) guardar en txt para auditoría
        if session["razon_social"] or session["expediente"] or session["ruc"]:
            save_certificado_info(session["razon_social"], session["expediente"], session["ruc"])

        # 7) Redirigir a la página de resultados
        return redirect(url_for("result"))

    # GET: solo renderiza el formulario
    return render_template("index.html", error=None)

@app.route("/result", methods=["GET"])
def result():
    # 1) Recuperar lo capturado
    razon_social = session.get("razon_social")
    expediente   = session.get("expediente")
    ruc          = session.get("ruc")
    social_url   = session.get("social_url", "").strip() if session.get("social_url") else ""

    if not expediente:
        return render_template(
            "result.html",
            status="error",
            msg="No se pudo extraer el N° de expediente del PDF.",
            datos={"razon_social": razon_social, "expediente": expediente, "ruc": ruc},
            social_url=social_url,
            social={},   # sin métricas
        )

    # 2) Predicción por expediente (tu función en funtions.py)
    resp = predecir_por_expediente(expediente)
    
    # Métricas de red social (IG/FB)
    social_metrics = {}
    if social_url:
        try:
            social_metrics = scrape_stats_from_url(social_url) or {}
        except Exception as e:
            social_metrics = {"error": str(e)}

    # 3) Render con todo lo que necesitas en pantalla
    return render_template(
        "result.html",
        features_json=resp.get("features") or {},
        status=resp.get("status"),
        msg=resp.get("msg"),
        datos={"razon_social": razon_social, "expediente": expediente, "ruc": ruc},
        social_url=social_url,
        pred_num=resp.get("pred_num"),
        pred_texto=resp.get("pred_texto"),
        total_score=resp.get("total_score"),
        social=social_metrics,   # ⬅️ aquí viajan las métricas
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
