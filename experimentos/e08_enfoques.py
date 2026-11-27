"""N1 · Enfoques: ¿la visión clásica elegida o una red neuronal?

Tres caminos, medidos con las mismas pruebas del proceso 6:
- Clásica: la cadena de 6 procesos decidida en e01–e06 (Mahalanobis con Hotelling).
- Red supervisada: ResNet-18 preentrenada en ImageNet, ajustada para "buena/mala"
  con aumento de datos (giros, espejos, brillo). Necesita ejemplos de malas.
- Anomalías con red (estilo PatchCore): la misma ResNet-18 sin reentrenar; se guardan
  los parches de las golillas buenas y una pieza es mala si algún parche suyo no se
  parece a ninguno guardado. Solo necesita buenas.

Las dos redes reciben el recorte centrado en la golilla (lo entrega la segmentación
clásica: en el fondo son híbridas). Pruebas: A (reales dejando una fuera), R (las
mismas giradas) y C (600 sintéticas de entrenamiento → 600 de prueba).
Corre en la GTX 1060 de norea-node (mirar `nvidia-smi` antes).
"""

import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from comun import AQUA, AZUL, NARANJA, RESULTADOS, barras_h, decision, estilo, guardar
from matplotlib import pyplot as plt
from torchvision.models import ResNet18_Weights, resnet18

from golillas import preproceso, segmentacion, sintetico
from golillas.datos import cargar_reales

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LISTA = ["ninguno"] * 300 + [d for d in sintetico.DEFECTOS for _ in range(50)]
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESV = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
torch.manual_seed(0)


def recorte(img):
    g = preproceso.preprocesar(img, decision("preproceso"))
    m = segmentacion.segmentar(g, decision("segmentacion"))
    mm = cv2.moments(m, binaryImage=True)
    cx, cy = (mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]) if mm["m00"] else (240, 320)
    r = 80
    p = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
    c = p[int(cy) : int(cy) + 2 * r, int(cx) : int(cx) + 2 * r]
    return cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA)[..., ::-1].copy()  # RGB


def tensor(imgs):
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255
    return (x - MEDIA) / DESV


def d4(x):
    """Las 8 simetrías exactas (giros de 90° y espejos) de un lote NCHW."""
    out = []
    for k in range(8):
        y = torch.rot90(x, k % 4, dims=(2, 3))
        out.append(torch.flip(y, dims=(3,)) if k >= 4 else y)
    return out


def girar_np(img, k):
    img = np.rot90(img, k % 4)
    return np.ascontiguousarray(img[:, ::-1] if k >= 4 else img)


# ---------------------------------------------------------------- PatchCore liviano
class Parches:
    def __init__(self):
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(DEV)
        self.net = net

    @torch.no_grad()
    def rasgos(self, x):
        n = self.net
        h = n.maxpool(n.relu(n.bn1(n.conv1(x.to(DEV)))))
        l2 = n.layer2(n.layer1(h))
        l3 = n.layer3(l2)
        f = torch.cat([l2, Fn.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)], 1)
        f = Fn.avg_pool2d(f, 3, 1, 1)
        return f.permute(0, 2, 3, 1).reshape(x.shape[0], -1, f.shape[1])  # N × parches × C

    def ajustar(self, x_buenas):
        self.banco = torch.cat([self.rasgos(v).reshape(-1, self.rasgos(v).shape[-1]) for v in d4(x_buenas)])
        # umbral: cada buena contra el banco sin sus propios parches (dejando una fuera)
        n, s = x_buenas.shape[0], []
        por = self.banco.shape[0] // (8 * n)
        idx = torch.arange(self.banco.shape[0], device=self.banco.device)
        dueno = (idx % (n * por)) // por
        for i in range(n):
            s.append(self._puntaje(x_buenas[i : i + 1], self.banco[dueno != i])[0])
        s = np.array(s)
        self.umbral = float(s.max() + 2 * s.std()) if n > 1 else float(s.max() * 1.5)
        return self

    @torch.no_grad()
    def _puntaje(self, x, banco):
        f = self.rasgos(x)
        out = []
        for fi in f:
            d = torch.cdist(fi, banco).min(dim=1).values
            out.append(float(d.max()))
        return out

    def puntaje(self, x):
        return np.array(self._puntaje(x, self.banco))

    def predecir(self, x):
        return self.puntaje(x) > self.umbral


# ---------------------------------------------------------------- red supervisada
def red_supervisada(x, y, epocas=15):
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(512, 1)
    net = net.to(DEV).train()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    y = torch.tensor(y, dtype=torch.float32)
    peso = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)])
    g = torch.Generator().manual_seed(0)
    for _ in range(epocas):
        orden = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), 16):
            b = orden[i : i + 16]
            xb = x[b]
            k = int(torch.randint(0, 8, (1,), generator=g))
            xb = d4(xb)[k] * (1 + 0.2 * (torch.rand(len(b), 1, 1, 1, generator=g) - 0.5))
            loss = Fn.binary_cross_entropy_with_logits(net(xb.to(DEV)).squeeze(1), y[b].to(DEV),
                                                       pos_weight=peso.to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    class Modelo:
        @torch.no_grad()
        def predecir(self, xx):
            return (torch.sigmoid(net(xx.to(DEV)).squeeze(1)) > 0.5).cpu().numpy()

    return Modelo()


def metricas(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    return dict(deteccion=float(p[y].mean()) if y.any() else np.nan, falsas=float(p[~y].mean()))


def main():
    global DEV
    estilo()
    reales = cargar_reales()
    Xr = tensor([recorte(m.imagen()) for m in reales])
    yr = np.array([m.es_mala for m in reales])
    Xg = [tensor([recorte(girar_np(m.imagen(), k)) for k in range(1, 8)]) for m in reales]
    semillas = (("entrenamiento", 5), ("prueba", 6))  # las mismas sintéticas que e05
    Xs = {s: tensor([recorte(sintetico.generar(sem * 100_000 + i, defecto=LISTA[i]).imagen)
                     for i in range(len(LISTA))]) for s, sem in semillas}
    ys = {s: np.array([d != "ninguno" for d in LISTA]) for s, _ in semillas}
    print("recortes listos", DEV)

    filas = []
    for nombre in ("anomalias_red", "red_supervisada"):
        t0 = time.time()
        pa, pr = np.zeros(len(yr), bool), []
        for i in range(len(yr)):
            f = np.arange(len(yr)) != i
            if nombre == "anomalias_red":
                mod = Parches().ajustar(Xr[f][~yr[f]])
            else:
                mod = red_supervisada(Xr[f], yr[f])
            pa[i] = mod.predecir(Xr[i : i + 1])[0]
            pr.append(mod.predecir(Xg[i]))
        a = metricas(yr, pa)
        r = metricas(np.repeat(yr, 7), np.concatenate(pr))
        if nombre == "anomalias_red":
            sub = Xs["entrenamiento"][~ys["entrenamiento"]][:30]  # 30 buenas × 8 giros para el banco
            mod = Parches().ajustar(sub)
        else:
            mod = red_supervisada(Xs["entrenamiento"], ys["entrenamiento"], epocas=8)
        pc = np.concatenate([mod.predecir(Xs["prueba"][i : i + 32]) for i in range(0, 600, 32)])
        c = metricas(ys["prueba"], pc)
        # latencia por pieza (GPU y CPU)
        x1 = Xr[:1]
        lat = {}
        for dev in ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]:
            DEV_old, DEV = DEV, dev
            if nombre == "anomalias_red":
                mm = Parches().ajustar(Xr[~yr][:5])
            else:
                mm = red_supervisada(Xr[:4], yr[:4], epocas=1)
            for _ in range(3):
                mm.predecir(x1)
            t1 = time.perf_counter()
            for _ in range(10):
                mm.predecir(x1)
            lat[dev] = (time.perf_counter() - t1) * 100
            DEV = DEV_old
        filas.append(dict(enfoque=nombre, A_deteccion=a["deteccion"], A_falsas=a["falsas"],
                          R_deteccion=r["deteccion"], R_falsas=r["falsas"], C_deteccion=c["deteccion"],
                          C_falsas=c["falsas"], ms_gpu=lat.get("cuda", np.nan), ms_cpu=lat["cpu"],
                          minutos=(time.time() - t0) / 60))
        print(filas[-1])

    clas = pd.read_csv(RESULTADOS / "e06_decision.csv").set_index("modelo").loc[decision("decision")]
    tiempos = pd.read_csv(RESULTADOS / "e05_tiempos.csv", index_col=0).iloc[:, 0].sum()
    filas.insert(0, dict(enfoque="clasica", A_deteccion=clas.A_deteccion, A_falsas=clas.A_falsas,
                         R_deteccion=clas.R_deteccion, R_falsas=clas.R_falsas, C_deteccion=clas.C_deteccion,
                         C_falsas=clas.C_falsas, ms_gpu=np.nan, ms_cpu=tiempos, minutos=np.nan))
    T = pd.DataFrame(filas).set_index("enfoque")
    T.to_csv(RESULTADOS / "e08_enfoques.csv")
    (RESULTADOS / "e08_dispositivo.json").write_text(json.dumps({"dispositivo": torch.cuda.get_device_name(0)
                                                                if torch.cuda.is_available() else "cpu"}))

    orden = list(T.index)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    for ax, p, titulo in zip(axs, "ARC", ["A · reales, dejando una fuera", "R · reales giradas",
                                          "C · sintéticas (600)"]):
        barras_h(ax, orden, [100 * T.loc[orden, f"{p}_deteccion"], 100 * T.loc[orden, f"{p}_falsas"]],
                 ["malas detectadas (%)", "buenas retiradas (%)"], colores=[AZUL, NARANJA], elegido="clasica",
                 fmt="{:.0f}", lim=(0, 118))
        ax.set_title(titulo, fontsize=10)
    barras_h(axs[3], orden, [T.loc[orden, "ms_cpu"], T.loc[orden, "ms_gpu"]], ["CPU i3", "GPU GTX 1060"],
             colores=[AQUA, AZUL], elegido="clasica", fmt="{:.0f}")
    axs[3].set_title("ms por pieza", fontsize=10)
    guardar(fig, "e08_enfoques.png")
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
