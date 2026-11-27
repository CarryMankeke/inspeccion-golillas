"""Proceso 6 · Decisión: ¿la golilla se aprueba o se retira?

Dos familias:
- De una clase: aprenden solo de golillas buenas ("así se ve lo normal") y retiran
  lo que se aleja. Es como se calibra en planta con piezas patrón.
- Supervisadas: aprenden de buenas y malas. Necesitan ejemplos de cada defecto.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, f
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier


def _limpio(X):
    X = np.asarray(X, float)
    return np.where(np.isfinite(X), X, 1e3)  # sin medición = muy anómalo


class Tolerancias:
    """Una banda por rasgo: mediana ± k·MAD de las buenas. Se retira si algún rasgo se sale."""

    una_clase = True

    def __init__(self, k: float = 6.0):
        self.k = k

    def fit(self, Xb, X=None, y=None):
        Xb = np.asarray(Xb, float)
        self.med = np.nanmedian(Xb, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(Xb - self.med), axis=0)
        self.esc = np.maximum(mad, 1e-3 * np.maximum(np.abs(self.med), 1e-3))
        return self

    def puntaje(self, X):
        z = np.abs(np.asarray(X, float) - self.med) / self.esc
        return np.where(np.isfinite(z), z, 1e3).max(axis=1)

    def predecir(self, X):
        return self.puntaje(X) > self.k

    def parametros(self):
        return dict(k=self.k, mediana=self.med.tolist(), escala=self.esc.tolist())

    def culpable(self, x, nombres):
        z = np.abs(np.asarray(x, float) - self.med) / self.esc
        z = np.where(np.isfinite(z), z, 1e3)
        return [nombres[i] for i in np.argsort(-z) if z[i] > self.k]


class Mahalanobis:
    """Distancia de Mahalanobis² a la nube de golillas buenas.

    Umbral "chi2": supone que la media y la covarianza se conocen exactas.
    Umbral "hotelling": región de predicción T² para una pieza NUEVA cuando media y
    covarianza se estimaron con solo n piezas: d(n+1)(n−1)/(n(n−d))·F(d, n−d).
    Con n = 14 y d = 4 el umbral sube de 18,5 a ≈ 63.
    """

    una_clase = True

    def __init__(self, p=0.999, umbral="chi2"):
        self.p = p
        self.tipo = umbral

    def fit(self, Xb, X=None, y=None):
        Xb = _limpio(Xb)
        n, d = Xb.shape
        self.sc = StandardScaler().fit(Xb)
        self.cov = LedoitWolf().fit(self.sc.transform(Xb))
        if self.tipo == "hotelling" and n > d:
            self.t = d * (n + 1) * (n - 1) / (n * (n - d)) * f.ppf(self.p, d, n - d)
        else:
            self.t = chi2.ppf(self.p, d)
        return self

    def puntaje(self, X):
        return self.cov.mahalanobis(self.sc.transform(_limpio(X)))

    def predecir(self, X):
        return self.puntaje(X) > self.t

    def parametros(self):
        return dict(media=self.sc.mean_.tolist(), escala=self.sc.scale_.tolist(),
                    centro=self.cov.location_.tolist(), precision=self.cov.precision_.tolist(),
                    umbral=float(self.t), p=self.p)


def mahalanobis_desde(par: dict, X) -> np.ndarray:
    """Distancia de Mahalanobis² con parámetros guardados (sin scikit-learn)."""
    Z = (_limpio(np.atleast_2d(X)) - np.array(par["media"])) / np.array(par["escala"]) - np.array(par["centro"])
    return np.einsum("ij,jk,ik->i", Z, np.array(par["precision"]), Z)


class _SkUnaClase:
    una_clase = True

    def __init__(self, modelo):
        self.modelo = modelo

    def fit(self, Xb, X=None, y=None):
        self.p = make_pipeline(StandardScaler(), self.modelo).fit(_limpio(Xb))
        return self

    def puntaje(self, X):
        return -self.p.decision_function(_limpio(X))

    def predecir(self, X):
        return self.p.predict(_limpio(X)) == -1


class _SkSupervisado:
    una_clase = False

    def __init__(self, modelo):
        self.modelo = modelo

    def fit(self, Xb, X, y):
        self.p = make_pipeline(StandardScaler(), self.modelo).fit(_limpio(X), y)
        return self

    def puntaje(self, X):
        return self.p.predict_proba(_limpio(X))[:, 1]

    def predecir(self, X):
        return self.p.predict(_limpio(X)) == 1


def modelos():
    return {
        "tolerancias": Tolerancias(),
        "mahalanobis": Mahalanobis(),
        "mahalanobis_hotelling": Mahalanobis(umbral="hotelling"),
        "isolation_forest": _SkUnaClase(IsolationForest(n_estimators=200, random_state=0)),
        "ocsvm": _SkUnaClase(OneClassSVM(nu=0.05, gamma="scale")),
        "logistica": _SkSupervisado(LogisticRegression(class_weight="balanced", max_iter=2000)),
        "arbol": _SkSupervisado(DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)),
        "svm_rbf": _SkSupervisado(SVC(class_weight="balanced", probability=True, random_state=0)),
        "random_forest": _SkSupervisado(
            RandomForestClassifier(200, class_weight="balanced", random_state=0)),
        "knn": _SkSupervisado(KNeighborsClassifier(3)),
    }
