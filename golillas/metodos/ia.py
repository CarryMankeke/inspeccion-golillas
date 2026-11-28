"""Los métodos de anomalib 2.7.1 del ADR 0017, con su configuración por defecto y la interfaz común.

Se usan los modelos de PyTorch de anomalib sin su `Engine`, para leer el puntaje crudo de cada vista
(sin el post-procesador, que lo normaliza y lo recorta a [0, 1]) y aplicar después el umbral
conformal. El pre-procesador es el de cada modelo (su tamaño y su normalización por defecto).

Reglas del ADR 0017 cuando no alcanzan los recursos (cada caso queda en `info`):
- Sin entrenamiento (PatchCore, PaDiM, AnomalyDINO): si no cabe en memoria, se ajusta con una
  submuestra de piezas al azar (semilla 0) que baja a la mitad hasta que quepa.
- Con entrenamiento (Dinomaly): si no cabe, el lote baja a la mitad; si pasa de 2 horas, se corta.
- WinCLIP no aprende de las buenas: compara cada vista con frases de «washer» buena y dañada.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

LOTE = 32  # el train_batch_size por defecto de los datamodules de anomalib
PASOS_DINOMALY = 5000  # MAX_STEPS_DEFAULT de Dinomaly en anomalib 2.7.1
TOPE_S = 2 * 3600  # ADR 0017: el ajuste se corta a las 2 horas por tipo
CLASE_WINCLIP = "washer"  # el nombre del producto en las frases de WinCLIP (no se busca otro)
MODELOS = {"patchcore": "Patchcore", "padim": "Padim", "anomalydino": "AnomalyDINO", "dinomaly": "Dinomaly",
           "winclip": "WinClip"}


def sin_memoria(e: BaseException) -> bool:
    return isinstance(e, (torch.OutOfMemoryError, MemoryError)) or "out of memory" in str(e).lower()


class Anomalib:
    def __init__(self, nombre: str, dev: str | None = None, lote: int = LOTE, pasos: int = PASOS_DINOMALY,
                 tope_s: float = TOPE_S, semilla: int = 0):
        self.nombre, self.lote, self.pasos, self.tope_s, self.semilla = nombre, lote, pasos, tope_s, semilla
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.info: dict = {}
        self.mod = self._nuevo()

    def _nuevo(self):
        import anomalib.models as am

        kw = dict(post_processor=False, evaluator=False, visualizer=False)
        if self.nombre == "winclip":
            kw["class_name"] = CLASE_WINCLIP
        return getattr(am, MODELOS[self.nombre])(**kw)

    # ------------------------------------------------------------ entrada
    def _x(self, vistas: list[np.ndarray], dev: str | None = None) -> torch.Tensor:
        """Vistas BGR uint8 → tensor RGB en [0, 1] con el pre-procesador del modelo (una por una: tamaños distintos)."""
        pre = self.mod.pre_processor.transform
        dev = dev or self.dev
        xs = [torch.from_numpy(np.ascontiguousarray(v[..., ::-1])).permute(2, 0, 1)[None].float().div(255).to(dev)
              for v in vistas]
        return torch.cat([pre(x) for x in xs])

    # ------------------------------------------------------------ ajuste
    def ajustar(self, piezas):
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.nombre == "winclip":
            self.mod.model.setup(CLASE_WINCLIP, None)
            self.mod.to(self.dev).eval()
        elif self.nombre == "dinomaly":
            self._entrenar([v for p in piezas for v in p.vistas])
        else:
            self._banco(piezas)
        self.info["s_ajuste"] = time.time() - t0
        if torch.cuda.is_available():
            self.info["gpu_mb_max"] = torch.cuda.max_memory_allocated() / 2**20
        return self

    def _banco(self, piezas):
        usadas = list(range(len(piezas)))
        while True:
            try:
                self.mod = self._nuevo().to(self.dev)
                self.mod.model.train()
                vistas = [v for i in usadas for v in piezas[i].vistas]
                with torch.no_grad():
                    for a in range(0, len(vistas), self.lote):
                        self.mod.model(self._x(vistas[a : a + self.lote]))
                    self.mod.fit()
                break
            except (RuntimeError, MemoryError) as e:
                if not sin_memoria(e) or len(usadas) <= 1:
                    raise
                self.mod = None
                torch.cuda.empty_cache()
                rng = np.random.default_rng(self.semilla)
                usadas = sorted(rng.choice(len(piezas), max(1, len(usadas) // 2), replace=False).tolist())
                self.info["submuestra_piezas"] = len(usadas)
        self.info.setdefault("submuestra_piezas", None)
        self.info["piezas_ajuste"] = len(usadas)
        self.mod.model.eval()

    def _entrenar(self, vistas):
        from anomalib.models.image.dinomaly.components import StableAdamW, WarmCosineScheduler
        from anomalib.models.image.dinomaly.lightning_model import TRAINING_CONFIG

        lote = min(self.lote, len(vistas))
        while True:
            try:
                torch.manual_seed(self.semilla)
                self.mod = self._nuevo().to(self.dev)
                self.mod.model.train()
                params = [p for p in self.mod.trainable_modules.parameters() if p.requires_grad]
                opt = StableAdamW([{"params": self.mod.trainable_modules.parameters()}], **TRAINING_CONFIG["optimizer"])
                sch = WarmCosineScheduler(opt, **{**TRAINING_CONFIG["scheduler"], "total_iters": self.pasos})
                rng, orden, t0, paso = np.random.default_rng(self.semilla), [], time.time(), 0
                cortado = False
                while paso < self.pasos:
                    if len(orden) < lote:  # una época barajada, sin el último lote incompleto
                        orden = rng.permutation(len(vistas)).tolist()
                    idx, orden = orden[:lote], orden[lote:]
                    perdida = self.mod.model(self._x([vistas[i] for i in idx]), global_step=paso)
                    opt.zero_grad(set_to_none=True)
                    perdida.backward()
                    torch.nn.utils.clip_grad_norm_(params, TRAINING_CONFIG["trainer"]["gradient_clip_val"])
                    opt.step()
                    sch.step()
                    paso += 1
                    if time.time() - t0 > self.tope_s:
                        cortado = True
                        break
                break
            except (RuntimeError, MemoryError) as e:
                if not sin_memoria(e) or lote <= 1:
                    raise
                self.mod = None
                torch.cuda.empty_cache()
                lote //= 2
        self.info.update(lote=lote, pasos=paso, cortado_2h=cortado, perdida_final=float(perdida.detach()))
        self.mod.model.eval()

    # ------------------------------------------------------------ puntajes
    @torch.no_grad()
    def _puntajes(self, vistas, dev=None, lote=None) -> np.ndarray:
        lote, out = lote or self.lote, []
        for a in range(0, len(vistas), lote):
            r = self.mod.model(self._x(vistas[a : a + lote], dev))
            out.append(torch.as_tensor(r.pred_score).reshape(-1).float().cpu().numpy())
        return np.concatenate(out)

    def puntajes_vistas(self, piezas) -> np.ndarray:
        v = len(piezas[0].vistas)
        lote = self.lote
        while True:
            try:
                s = self._puntajes([x for p in piezas for x in p.vistas], lote=lote)
                return s.reshape(len(piezas), v)
            except (RuntimeError, MemoryError) as e:
                if not sin_memoria(e) or lote <= 1:
                    raise
                torch.cuda.empty_cache()
                lote //= 2

    def en_cpu(self):
        otro = copy.copy(self)
        otro.mod, otro.dev = copy.deepcopy(self.mod).to("cpu"), "cpu"
        return otro

    def una_pieza(self, pieza) -> float:
        return float(self._puntajes(pieza.vistas, lote=len(pieza.vistas)).max())
