"""PatchCore propio (e09 a e12): banco de parches de ResNet-18 (layer2+3), coreset del 1 %.

La clase `PatchCore` es la de e09, movida aquí sin cambios para que e13 la use como método;
e09 a e12 la importan desde este módulo. El control de e13 comprueba que repite en Kaggle los
números de e12.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from torchvision.models import ResNet18_Weights, resnet18

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CAPAS = {"l2+l3": (2, 3), "l1+l2+l3": (1, 2, 3)}
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESV = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LADO = 224
POCAS = 50  # con menos buenas que esto, el banco suma las 8 simetrías (como e09 y e11)
RED = None


def u8(imgs):
    """Lista de recortes RGB H×W×3 → tensor uint8 N×3×H×W (ocupa 4 veces menos que float)."""
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()


def norm(x, dev=None):
    dev = dev or DEV
    return (x.to(dev).float() / 255 - MEDIA.to(dev)) / DESV.to(dev)


def gira(x, k):
    """Una de las 8 simetrías exactas (giros de 90° y espejos) de un lote NCHW."""
    y = torch.rot90(x, k % 4, dims=(2, 3))
    return torch.flip(y, dims=(3,)) if k >= 4 else y


@torch.no_grad()
def kcentro(f, m, dim=128):
    """Coreset voraz (k-centros) sobre una proyección aleatoria a `dim` dimensiones, como PatchCore."""
    g = torch.Generator().manual_seed(0)
    z = f @ (torch.randn(f.shape[1], dim, generator=g) / dim**0.5).to(f.device)
    sel = torch.zeros(m, dtype=torch.long, device=f.device)
    dmin = ((z - z[0]) ** 2).sum(1)
    for t in range(1, m):  # todo en la GPU, sin sincronizar en cada paso
        j = torch.argmax(dmin).view(1)
        sel[t : t + 1] = j
        dmin = torch.minimum(dmin, ((z - z.index_select(0, j)) ** 2).sum(1))
    return sel


class PatchCore:
    def __init__(self, capas, coreset, dev=None):
        global RED
        dev = dev or DEV
        if dev == DEV:
            if RED is None:
                RED = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(DEV)
            red = RED
        else:
            red = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(dev)
        self.capas, self.coreset, self.net, self.dev = capas, coreset, red, dev

    @torch.no_grad()
    def rasgos(self, x):
        n = self.net
        h = n.maxpool(n.relu(n.bn1(n.conv1(norm(x, self.dev)))))
        l1 = n.layer1(h)
        l2 = n.layer2(l1)
        l3 = n.layer3(l2)
        mapas = {1: Fn.adaptive_avg_pool2d(l1, l2.shape[-2:]), 2: l2,
                 3: Fn.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)}
        f = Fn.avg_pool2d(torch.cat([mapas[c] for c in self.capas], 1), 3, 1, 1)
        return f.permute(0, 2, 3, 1).reshape(x.shape[0], -1, f.shape[1])  # N × parches × C

    @torch.no_grad()
    def ajustar(self, xb, giros=8):
        partes, duenos = [], []
        for k in range(giros):
            for i in range(0, len(xb), 32):
                f = self.rasgos(gira(xb[i : i + 32], k))
                partes.append(f.reshape(-1, f.shape[-1]))
                duenos.append(torch.arange(i, i + len(f), device=self.dev).repeat_interleave(f.shape[1]))
        banco, dueno = torch.cat(partes), torch.cat(duenos)
        del partes
        if self.coreset < 1:
            sel = kcentro(banco, max(1, int(self.coreset * len(banco))))
            banco, dueno = banco[sel], dueno[sel]
        self.banco = banco
        # umbral por LOO interno: cada buena contra el banco sin sus propios parches
        s = [self._puntaje(self.rasgos(xb[i : i + 1])[0], banco[dueno != i]) for i in range(len(xb))]
        self.umbral, self.n_banco, self.loo = float(max(s)), len(banco), np.array(s)
        return self

    @staticmethod
    def _puntaje(f, banco, trozo=65536):
        dmin = torch.full((f.shape[0],), float("inf"), device=f.device)
        for j in range(0, len(banco), trozo):
            dmin = torch.minimum(dmin, torch.cdist(f, banco[j : j + trozo]).min(1).values)
        return float(dmin.max())

    @torch.no_grad()
    def puntaje(self, x):
        return np.array([self._puntaje(f, self.banco) for i in range(0, len(x), 16)
                         for f in self.rasgos(x[i : i + 16])])

    def a(self, dev):
        """Copia liviana en otro dispositivo (para medir la latencia en CPU)."""
        otro = PatchCore.__new__(PatchCore)
        otro.capas, otro.coreset, otro.dev, otro.umbral = self.capas, self.coreset, dev, self.umbral
        otro.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(dev)
        otro.banco = self.banco.to(dev)
        return otro


def a_tensor(vistas: list[np.ndarray]) -> torch.Tensor:
    """Vistas BGR → 224 × 224 RGB uint8, sin recortar (las vistas de MANTA ya vienen centradas)."""
    return u8([cv2.resize(v, (LADO, LADO), interpolation=cv2.INTER_AREA)[..., ::-1].copy() for v in vistas])


class PatchCorePropio:
    """El PatchCore de e09–e12 con la interfaz del ADR 0017 (layer2+3, coreset 1 %)."""

    def __init__(self, capas: str = "l2+l3", coreset: float = 0.01, dev: str | None = None):
        self.capas, self.coreset, self.dev = CAPAS[capas], coreset, dev or DEV

    def ajustar(self, piezas):
        x = a_tensor([v for p in piezas for v in p.vistas])
        self.pc = PatchCore(self.capas, self.coreset, self.dev).ajustar(x, giros=8 if len(x) < POCAS else 1)
        self.info = dict(n_vistas=len(x), n_banco=self.pc.n_banco)
        return self

    def puntajes_vistas(self, piezas) -> np.ndarray:
        v = len(piezas[0].vistas)
        return self.pc.puntaje(a_tensor([x for p in piezas for x in p.vistas])).reshape(len(piezas), v)

    def en_cpu(self):
        otro = PatchCorePropio.__new__(PatchCorePropio)
        otro.capas, otro.coreset, otro.dev, otro.pc = self.capas, self.coreset, "cpu", self.pc.a("cpu")
        return otro

    def una_pieza(self, pieza) -> float:
        return float(self.puntajes_vistas([pieza]).max())
