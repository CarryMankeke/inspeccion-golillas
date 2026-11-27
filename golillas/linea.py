"""Proceso 7 · Línea de producción simulada: de la decisión a la golilla fuera de la cinta.

Vista desde arriba: la cinta avanza en +x. En x = 0 hay un sensor fotoeléctrico que
dispara la cámara. Aguas abajo, en x = L, una boquilla de aire sopla la golilla
mala fuera de la cinta. El problema es de tiempo: hay que saber CUÁNDO la golilla
mala pasa frente a la boquilla, aunque la cinta cambie de velocidad y las piezas
resbalen un poco.

Estrategias de seguimiento (las opciones N3 de este proceso):
- "temporizador": dispara a t_foto + L / v_nominal. Supone velocidad constante.
- "encoder": cuenta el avance real de la cinta con un encoder; dispara cuando la
  cinta avanzó L. Corrige los cambios de velocidad, no el resbalamiento.
- "encoder+sensor": como el anterior, pero un segundo sensor 40 mm antes de la
  boquilla vuelve a ubicar la pieza. Corrige también el resbalamiento.

Unidades: mm, s. Todo es reproducible con la semilla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ESTRATEGIAS = ("temporizador", "encoder", "encoder+sensor")


@dataclass
class Parametros:
    v0: float = 500.0  # velocidad nominal de la cinta (mm/s)
    L: float = 250.0  # distancia cámara → boquilla (mm)
    d_sensor2: float = 40.0  # segundo sensor, mm antes de la boquilla
    hueco_min: float = 30.0  # separación mínima entre piezas (mm, borde a borde aprox.)
    hueco_medio: float = 40.0  # separación extra media (exponencial, mm)
    p_mala: float = 0.10
    oscilacion: float = 0.08  # ±8 % de velocidad, periodo 3 s
    ruido_v: float = 0.02  # ruido lento de velocidad (fracción)
    resbale: tuple = (0.0, 0.04)  # tras la cámara cada pieza avanza (1 − ε)·cinta, ε uniforme
    t_captura: float = 0.005  # cámara + transferencia (s)
    t_valvula: float = 0.010  # retardo de apertura de la válvula (s)
    t_soplo: float = 0.015  # duración del soplo (s)
    medio_ancho_soplo: float = 7.0  # el chorro retira piezas con centro a ±7 mm
    resolucion_encoder: float = 0.1  # mm por pulso
    dt: float = 0.0005  # paso de integración (s)


@dataclass
class Pieza:
    x0: float  # posición inicial (mm, negativa: aún no llega a la cámara)
    mala: bool
    resbale: float
    imagen: int = -1  # índice de la variante de imagen que ve la cámara
    t_foto: float = np.nan
    t_decision: float = np.nan
    dice_mala: bool = False
    t_disparo: float = np.nan
    a_tiempo: bool = True
    retirada: bool = False
    t_retiro: float = np.nan


@dataclass
class Simulacion:
    par: Parametros
    estrategia: str
    t: np.ndarray
    S: np.ndarray  # avance acumulado de la cinta (mm)
    piezas: list[Pieza]
    soplos: list[tuple[float, float]] = field(default_factory=list)

    def x(self, p: Pieza, t):
        return posicion(p, np.interp(t, self.t, self.S))

    def resumen(self) -> dict:
        malas = [p for p in self.piezas if p.mala and np.isfinite(p.t_foto)]
        buenas = [p for p in self.piezas if not p.mala and np.isfinite(p.t_foto)]
        dur = self.t[-1]
        return dict(
            estrategia=self.estrategia, v0=self.par.v0,
            piezas_min=60 * len(malas + buenas) / dur,
            malas=len(malas), buenas=len(buenas),
            malas_retiradas=np.mean([p.retirada for p in malas]) if malas else np.nan,
            buenas_retiradas=np.mean([p.retirada for p in buenas]) if buenas else np.nan,
            malas_no_vistas=np.mean([not p.dice_mala for p in malas]) if malas else np.nan,
            malas_tarde=np.mean([p.dice_mala and not p.a_tiempo for p in malas]) if malas else np.nan,
            malas_soplo_errado=np.mean([p.dice_mala and p.a_tiempo and not p.retirada for p in malas])
            if malas else np.nan,
            buenas_falsa_alarma=np.mean([p.dice_mala for p in buenas]) if buenas else np.nan,
            buenas_arrastradas=np.mean([p.retirada and not p.dice_mala for p in buenas]) if buenas else np.nan,
        )


def posicion(p: Pieza, S):
    """Antes de la cámara la pieza va pegada a la cinta; después resbala un poco
    (vibración): avanza (1 − ε) de lo que avanza la cinta."""
    d = p.x0 + S
    return np.minimum(d, (1 - p.resbale) * d)


def simular(par: Parametros, estrategia: str, vision, duracion: float = 60.0, semilla: int = 0) -> Simulacion:
    """`vision(pieza, rng) -> (dice_mala, ms_procesamiento)` decide cada pieza."""
    rng = np.random.default_rng(semilla)
    t = np.arange(0, duracion + 2, par.dt)
    # velocidad real de la cinta: oscilación + ruido lento (paseo suavizado)
    # ruido blanco filtrado con una media móvil de 1 s (vía suma acumulada: rápido)
    w = max(int(1.0 / par.dt), 1)
    c = np.cumsum(np.concatenate([np.zeros(w), rng.normal(0, 1, t.size + w)]))
    ruido = (c[w:][: t.size] - c[: t.size]) / w
    ruido = par.ruido_v * ruido / max(np.std(ruido), 1e-9)
    v = par.v0 * (1 + par.oscilacion * np.sin(2 * np.pi * t / 3.0 + rng.uniform(0, 6.28)) + ruido)
    S = np.concatenate([[0], np.cumsum(v[:-1] * par.dt)])
    # piezas en fila
    piezas, x = [], -60.0
    while x > -(par.v0 * duracion):
        mala = rng.random() < par.p_mala
        piezas.append(Pieza(x, mala, rng.uniform(*par.resbale)))
        x -= 16 + par.hueco_min + rng.exponential(par.hueco_medio)
    sim = Simulacion(par, estrategia, t, S, piezas)

    def cruce(p, xpos):
        """Primer instante en que la pieza llega a xpos (interpolado)."""
        xs = posicion(p, S)
        k = np.searchsorted(xs, xpos)
        if k == 0 or k >= len(t):
            return np.nan
        return t[k - 1] + (xpos - xs[k - 1]) / (xs[k] - xs[k - 1]) * par.dt

    def encoder(t_):  # lectura cuantizada del avance de la cinta
        return np.floor(np.interp(t_, t, S) / par.resolucion_encoder) * par.resolucion_encoder

    libre = 0.0  # la visión procesa una pieza a la vez (cola)
    for p in piezas:
        p.t_foto = cruce(p, 0.0)
        if not np.isfinite(p.t_foto) or p.t_foto > duracion:
            p.t_foto = np.nan
            continue
        p.dice_mala, ms = vision(p, rng)
        inicio = max(p.t_foto + par.t_captura, libre)
        p.t_decision = inicio + ms / 1000
        libre = p.t_decision
        if not p.dice_mala:
            continue
        lag = 1 - np.mean(par.resbale)  # resbalamiento medio, calibrado: todas lo compensan
        if estrategia == "temporizador":
            llegada = p.t_foto + par.L / (par.v0 * lag)
        elif estrategia == "encoder":
            meta = encoder(p.t_foto) + par.L / lag
            k = np.searchsorted(S, meta)
            llegada = t[min(k, len(t) - 1)]
        elif estrategia == "encoder+sensor":
            t_s2 = cruce(p, par.L - par.d_sensor2)
            meta = encoder(t_s2) + par.d_sensor2 / lag
            k = np.searchsorted(S, meta)
            llegada = t[min(k, len(t) - 1)]
        else:
            raise ValueError(estrategia)
        # se ordena el soplo para que el chorro quede centrado en la llegada estimada
        p.t_disparo = llegada - par.t_valvula - par.t_soplo / 2
        if p.t_decision > p.t_disparo:
            p.a_tiempo = False
            continue
        t_medio = p.t_disparo + par.t_valvula + par.t_soplo / 2
        sim.soplos.append((p.t_disparo + par.t_valvula, p.t_disparo + par.t_valvula + par.t_soplo))
        for q in piezas:
            if not q.retirada and np.isfinite(q.t_foto) and abs(sim.x(q, t_medio) - par.L) <= par.medio_ancho_soplo:
                q.retirada = True
                q.t_retiro = t_medio
    return sim
