"""El inspector final: la cadena de los 6 procesos con las técnicas que ganaron.

    from golillas.inspector import Inspector
    r = Inspector().inspeccionar(cv2.imread("foto.jpg"))
    print(r.aprobada, r.razones, r.ms_total)

Las técnicas y la calibración vienen de `calibracion.json`, que escribe el
experimento e06 a partir de las decisiones medidas (ver docs/adr/).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bordes, decision, medicion, preproceso, rasgos, segmentacion

CALIBRACION = Path(__file__).with_name("calibracion.json")

NOMBRES_HUMANOS = {
    "relleno_anillo": "falta o sobra material en el anillo",
    "excentricidad": "agujero descentrado",
    "fourier_ext": "contorno exterior irregular (deformada o con muesca)",
    "desv_int_p95": "borde del agujero dañado o cortado",
}


@dataclass
class Resultado:
    aprobada: bool
    puntaje: float
    umbral: float
    razones: list[str] = field(default_factory=list)
    rasgos: dict = field(default_factory=dict)
    circulo_ext: medicion.Circulo | None = None
    circulo_int: medicion.Circulo | None = None
    mascara: np.ndarray | None = None
    ms: dict = field(default_factory=dict)

    @property
    def ms_total(self) -> float:
        return float(sum(self.ms.values()))


class Inspector:
    def __init__(self, calibracion: str | Path | dict | None = None):
        if isinstance(calibracion, dict):
            cal = calibracion
        else:
            cal = json.loads(Path(calibracion or CALIBRACION).read_text(encoding="utf-8"))
        self.cal = cal
        d = cal["decisiones"]
        self.pre, self.seg, self.borde, self.ajuste = d["preproceso"], d["segmentacion"], d["bordes"], d["medicion"]
        self.nombres = cal["rasgos"]
        self.modelo = cal["modelo"]
        self.par = cal["parametros"]
        self.exp = cal["explicacion"]

    def _puntaje(self, x):
        if self.modelo.startswith("mahalanobis"):
            return float(decision.mahalanobis_desde(self.par, x)[0]), float(self.par["umbral"])
        if self.modelo == "tolerancias":
            z = np.abs(np.asarray(x, float) - np.array(self.par["mediana"])) / np.array(self.par["escala"])
            return float(np.where(np.isfinite(z), z, 1e3).max()), float(self.par["k"])
        raise ValueError(f"modelo sin soporte en el inspector: {self.modelo}")

    def inspeccionar(self, img: np.ndarray) -> Resultado:
        ms = {}
        t = time.perf_counter()
        g = preproceso.preprocesar(img, self.pre)
        ms["preproceso"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        m = segmentacion.segmentar(g, self.seg)
        ms["segmentacion"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        pe, pi = bordes.bordes(g, m, self.borde)
        ms["bordes"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        ce, ci = medicion.medir(g, m, pe, pi, self.ajuste)
        ms["medicion"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        todos = rasgos.calcular(g, m, pe, pi, ce, ci)
        x = np.array([[todos[n] for n in self.nombres]], float)
        ms["rasgos"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        puntaje, umbral = self._puntaje(x)
        aprobada = puntaje <= umbral
        razones = []
        if not aprobada:
            z = np.abs(x[0] - np.array(self.exp["mediana"])) / np.array(self.exp["escala"])
            z = np.where(np.isfinite(z), z, 1e3)
            for i in np.argsort(-z):
                if z[i] > self.exp["k"]:
                    n = self.nombres[i]
                    razones.append(f"{NOMBRES_HUMANOS.get(n, n)} ({n}: {z[i]:.0f} desv.)"
                                   if np.isfinite(x[0, i]) else f"no se pudo medir {n}")
            if not razones:
                razones.append("combinación de rasgos fuera de lo normal")
        ms["decision"] = (time.perf_counter() - t) * 1000
        return Resultado(aprobada, puntaje, umbral, razones, todos, ce, ci, m, ms)
