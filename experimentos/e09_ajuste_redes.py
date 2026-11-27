"""N1 · e09: ajuste justo de las redes. ¿Sigue ganando la visión clásica?

En e08 las redes corrieron con desventaja: hiperparámetros fijos, la red supervisada
entrenada solo con 20 fotos reales, PatchCore sin coreset y con un umbral conservador
(máximo + 2σ), y sin AUC. Aquí cada red recibe un ajuste justo, decidido ANTES de ver
los resultados y sin mirar las etiquetas reales:

- Red supervisada: ResNet-18 (ImageNet) preentrenada con las 600 sintéticas de
  entrenamiento (semilla 5). Búsqueda lr {1e-4, 3e-4, 1e-3} × épocas {10, 20}, elegida
  con 600 sintéticas de VALIDACIÓN (semilla 9, nueva): mayor AUC (3 decimales);
  empate → menor pérdida. En la prueba A se ajusta con las 20 reales de cada vuelta con
  una receta fija (lr/10, 10 épocas, lotes de 10). B = la preentrenada sin ajuste fino.
- PatchCore: capas {layer2+3, layer1+2+3} × coreset {100 %, 10 %, 1 %}, elegido con la
  misma validación sintética (banco con 40 buenas de entrenamiento): mayor AUC; si otra
  queda a ≤ 0,005 del mejor, la de banco más chico. Umbral por LOO interno: el mayor
  puntaje de las buenas del banco, cada una contra el banco sin sus propios parches.
- Se informa el AUC (sin umbral; IC95 por bootstrap en A) además del punto de operación.

Pruebas como en e08: A (21 reales dejando una fuera), R (sus 7 giros exactos), B
(sintético → real) y C (600 sintéticas de entrenamiento → 600 de prueba, semilla 6).
Las 21 fotos reales NO eligen nada. `e09_configs.csv` muestra todas las configuraciones
en las reales solo para informar. Este script reescribe la decisión "enfoque" de e08.
Corre en la GTX 1060 de norea-node (mirar `nvidia-smi` antes).
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from comun import AQUA, AZUL, NARANJA, RESULTADOS, TINTA2, barras_h, decidir, decision, estilo, guardar
from e08_enfoques import LISTA, girar_np, recorte
from matplotlib import pyplot as plt
from torchvision.models import ResNet18_Weights, resnet18

from golillas import decision as dec
from golillas import sintetico
from golillas.datos import RAIZ, cargar_reales

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = RAIZ / "cache" / "e09_recortes.pt"
SEMILLAS = {"entrenamiento": 5, "validacion": 9, "prueba": 6}
LRS, EPOCAS = (1e-4, 3e-4, 1e-3), (10, 20)
FINO = dict(div_lr=10, epocas=10, lote=10)  # receta fija del ajuste con reales
CAPAS = {"l2+l3": (2, 3), "l1+l2+l3": (1, 2, 3)}
CORESETS = (1.0, 0.1, 0.01)
N_BANCO_SINT = 40
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESV = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- datos y utilidades
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


def datos():
    if CACHE.exists():
        return torch.load(CACHE, weights_only=False)
    t0 = time.time()
    reales = cargar_reales()
    d = dict(Xr=u8([recorte(m.imagen()) for m in reales]), yr=np.array([m.es_mala for m in reales]),
             Xg=torch.stack([u8([recorte(girar_np(m.imagen(), k)) for k in range(1, 8)]) for m in reales]),
             ys=np.array([x != "ninguno" for x in LISTA]))
    for s, sem in SEMILLAS.items():
        d[s] = u8([recorte(sintetico.generar(sem * 100_000 + i, defecto=LISTA[i]).imagen) for i in range(len(LISTA))])
    CACHE.parent.mkdir(exist_ok=True)
    torch.save(d, CACHE)
    print(f"recortes listos en {time.time() - t0:.0f} s")
    return d


def auc(y, s):
    """Área bajo la curva ROC = probabilidad de que una mala puntúe más alto que una buena."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    a, b = s[y][:, None], s[~y][None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def auc_ic(y, s, n=2000):
    """IC95 del AUC por bootstrap estratificado (se remuestrean malas y buenas por separado)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    rng = np.random.default_rng(0)
    im, ib = np.where(y)[0], np.where(~y)[0]
    yy = np.r_[np.ones(len(im), bool), np.zeros(len(ib), bool)]
    v = [auc(yy, np.r_[s[rng.choice(im, len(im))], s[rng.choice(ib, len(ib))]]) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def metricas(prefijo, y, s, umbral, ic=False):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    p = s > umbral
    out = {f"{prefijo}_deteccion": float(p[y].mean()), f"{prefijo}_falsas": float(p[~y].mean()),
           f"{prefijo}_auc": auc(y, s)}
    if ic:
        out[f"{prefijo}_auc_lo"], out[f"{prefijo}_auc_hi"] = auc_ic(y, s)
    return out


def latencia(f, reps=10):
    for _ in range(3):
        f()
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps * 1000


# ---------------------------------------------------------------- red supervisada
def nueva_red(estado=None):
    torch.manual_seed(0)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(512, 1)
    if estado is not None:
        net.load_state_dict(estado)
    return net.to(DEV)


def entrenar(net, x, y, lr, epocas, lote=16, semilla=0, copias_en=()):
    """AdamW con giros/espejos y contraste, como e08. Devuelve copias del estado en `copias_en`."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.tensor(y, dtype=torch.float32)
    peso = torch.tensor([(len(yt) - yt.sum()) / max(float(yt.sum()), 1.0)], device=DEV)
    g = torch.Generator().manual_seed(semilla)
    copias = {}
    for ep in range(1, epocas + 1):
        net.train()
        orden = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), lote):
            b = orden[i : i + lote]
            k = int(torch.randint(0, 8, (1,), generator=g))
            xb = norm(gira(x[b], k)) * (1 + 0.2 * (torch.rand(len(b), 1, 1, 1, generator=g) - 0.5)).to(DEV)
            loss = Fn.binary_cross_entropy_with_logits(net(xb).squeeze(1), yt[b].to(DEV), pos_weight=peso)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep in copias_en:
            copias[ep] = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.eval()
    return copias


@torch.no_grad()
def prob(net, x, dev=None):
    net.eval()
    return torch.cat([torch.sigmoid(net(norm(x[i : i + 64], dev)).squeeze(1)).cpu()
                      for i in range(0, len(x), 64)]).numpy()


def perdida(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def supervisada_reales(D, estado, lr):
    """A y R: LOO con ajuste fino en las 20 reales de cada vuelta. B: la preentrenada sin ajuste."""
    yr = D["yr"]
    pa, pr = np.zeros(len(yr)), np.zeros((len(yr), 7))
    for i in range(len(yr)):
        f = np.arange(len(yr)) != i
        net = nueva_red(estado)
        entrenar(net, D["Xr"][f], yr[f], lr / FINO["div_lr"], FINO["epocas"], lote=FINO["lote"], semilla=i)
        pa[i] = prob(net, D["Xr"][i : i + 1])[0]
        pr[i] = prob(net, D["Xg"][i])
    pre = nueva_red(estado)
    return pa, pr, prob(pre, D["Xr"]), prob(pre, D["Xg"].reshape(-1, *D["Xg"].shape[2:])).reshape(-1, 7)


# ---------------------------------------------------------------- PatchCore
RED = None


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
    def __init__(self, capas, coreset):
        global RED
        if RED is None:
            RED = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(DEV)
        self.capas, self.coreset, self.net, self.dev = capas, coreset, RED, DEV

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
    def ajustar(self, xb):
        partes, duenos = [], []
        for k in range(8):
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
        self.umbral, self.n_banco = float(max(s)), len(banco)
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


def patchcore_reales(D, capas, coreset):
    yr = D["yr"]
    pa, pr, um = np.zeros(len(yr)), np.zeros((len(yr), 7)), np.zeros(len(yr))
    for i in range(len(yr)):
        f = np.arange(len(yr)) != i
        pc = PatchCore(capas, coreset).ajustar(D["Xr"][f][~yr[f]])
        pa[i], pr[i], um[i] = pc.puntaje(D["Xr"][i : i + 1])[0], pc.puntaje(D["Xg"][i]), pc.umbral
    return pa, pr, um


# ---------------------------------------------------------------- visión clásica (puntajes de Mahalanobis)
def clasica():
    df = pd.read_csv(RESULTADOS / "e05_rasgos.csv")
    F, nombre = decision("rasgos").split(","), decision("decision")
    tr, te, re, rg = (df[df.split == s] for s in ("entrenamiento", "prueba", "real", "real_giro"))
    mala = {k: (v.clase == "mala").values for k, v in dict(tr=tr, te=te, re=re, rg=rg).items()}
    Xre, Xrg, Xtr = re[F].values, rg[F].values, tr[F].values
    sa, sr, ua, ur = np.zeros(len(re)), np.zeros(len(rg)), np.zeros(len(re)), np.zeros(len(rg))
    for i, id_ in enumerate(re.id.values):
        f = np.arange(len(re)) != i
        mod = dec.modelos()[nombre].fit(Xre[f][~mala["re"][f]], Xre[f], mala["re"][f].astype(int))
        sel = (rg.id == id_).values
        sa[i], sr[sel], ua[i], ur[sel] = mod.puntaje(Xre[i : i + 1])[0], mod.puntaje(Xrg[sel]), mod.t, mod.t
    mod = dec.modelos()[nombre].fit(Xtr[~mala["tr"]], Xtr, mala["tr"].astype(int))
    fila = dict(enfoque="clasica", config=nombre, **metricas("A", mala["re"], sa - ua, 0, ic=True),
                **metricas("R", mala["rg"], sr - ur, 0), **metricas("B", mala["re"], mod.puntaje(Xre), mod.t),
                **metricas("C", mala["te"], mod.puntaje(te[F].values), mod.t),
                ms_cpu=pd.read_csv(RESULTADOS / "e05_tiempos.csv", index_col=0).iloc[:, 0].sum(), ms_gpu=np.nan)
    return fila


# ---------------------------------------------------------------- experimento
def calcular():
    D = datos()
    ys, yr = D["ys"], D["yr"]
    yg = np.repeat(yr, 7)
    filas, configs = [clasica()], []
    print(filas[0])

    # --- red supervisada: búsqueda en validación sintética
    t0 = time.time()
    estados = {}
    for lr in LRS:
        net = nueva_red()
        copias = entrenar(net, D["entrenamiento"], ys, lr, max(EPOCAS), copias_en=EPOCAS)
        for ep, est in copias.items():
            net.load_state_dict(est)
            p = prob(net, D["validacion"])
            estados[(lr, ep)] = est
            configs.append(dict(red="supervisada", config=f"lr={lr:g}, {ep} épocas", lr=lr, epocas=ep,
                                auc_val=auc(ys, p), perdida_val=perdida(ys, p)))
            print(configs[-1], f"{time.time() - t0:.0f} s")
    bus = pd.DataFrame([c for c in configs if c["red"] == "supervisada"])
    bus["auc_r"] = bus.auc_val.round(3)
    mejor = bus.sort_values(["auc_r", "perdida_val"], ascending=[False, True]).iloc[0]
    sup_elegida = (float(mejor.lr), int(mejor.epocas))

    # todas las configuraciones en reales (solo informativo) y la elegida en la tabla principal
    for c in configs:
        if c["red"] != "supervisada":
            continue
        lr, ep = c["lr"], c["epocas"]
        pa, pr, pb, _ = supervisada_reales(D, estados[(lr, ep)], lr)
        c.update(**metricas("A", yr, pa, 0.5), **metricas("R", yg, pr.ravel(), 0.5), **metricas("B", yr, pb, 0.5),
                 elegida=(lr, ep) == sup_elegida)
        if (lr, ep) == sup_elegida:
            net = nueva_red(estados[(lr, ep)])
            pc_ = prob(net, D["prueba"])
            x1 = D["Xr"][:1]
            net_cpu = nueva_red(estados[(lr, ep)]).to("cpu")
            filas.append(dict(enfoque="red_supervisada", config=c["config"] + " + ajuste fino con reales",
                              **metricas("A", yr, pa, 0.5, ic=True), **metricas("R", yg, pr.ravel(), 0.5),
                              **metricas("B", yr, pb, 0.5), **metricas("C", ys, pc_, 0.5),
                              ms_gpu=latencia(lambda n=net, x=x1: prob(n, x)) if DEV == "cuda" else np.nan,
                              ms_cpu=latencia(lambda n=net_cpu, x=x1: prob(n, x, "cpu"))))
            print(filas[-1])
    print(f"supervisada: {(time.time() - t0) / 60:.1f} min")

    # --- PatchCore: búsqueda en validación sintética
    t0 = time.time()
    buenas_ent = D["entrenamiento"][~ys][:N_BANCO_SINT]
    pcs = []
    for nombre, capas in CAPAS.items():
        for cs in CORESETS:
            pc = PatchCore(capas, cs).ajustar(buenas_ent)
            s = pc.puntaje(D["validacion"])
            pcs.append(dict(red="patchcore", config=f"{nombre}, coreset {cs:.0%}", capas=nombre, coreset=cs,
                            auc_val=auc(ys, s), n_banco_sint=pc.n_banco))
            print(pcs[-1], f"{time.time() - t0:.0f} s")
    bus = pd.DataFrame(pcs)
    cerca = bus[bus.auc_val >= bus.auc_val.max() - 0.005]
    mejor = cerca.sort_values(["n_banco_sint", "auc_val"], ascending=[True, False]).iloc[0]
    pc_elegida = (mejor.capas, float(mejor.coreset))
    for c in pcs:
        pa, pr, um = patchcore_reales(D, CAPAS[c["capas"]], c["coreset"])
        c.update(**metricas("A", yr, pa - um, 0), **metricas("R", yg, (pr - um[:, None]).ravel(), 0),
                 elegida=(c["capas"], c["coreset"]) == pc_elegida)
        if (c["capas"], c["coreset"]) == pc_elegida:
            pc = PatchCore(CAPAS[c["capas"]], c["coreset"]).ajustar(buenas_ent)
            sc = pc.puntaje(D["prueba"])
            final = PatchCore(CAPAS[c["capas"]], c["coreset"]).ajustar(D["Xr"][~yr])  # banco con las 15 buenas
            x1 = D["Xr"][:1]
            cpu = final.a("cpu")
            filas.append(dict(enfoque="anomalias_red", config=c["config"] + ", umbral por LOO interno",
                              **metricas("A", yr, pa - um, 0, ic=True),
                              **metricas("R", yg, (pr - um[:, None]).ravel(), 0), **metricas("C", ys, sc, pc.umbral),
                              n_banco_real=final.n_banco,
                              ms_gpu=latencia(lambda m=final, x=x1: m.puntaje(x)) if DEV == "cuda" else np.nan,
                              ms_cpu=latencia(lambda m=cpu, x=x1: m.puntaje(x), reps=5)))
            print(filas[-1])
        print(c, f"{time.time() - t0:.0f} s")
    configs += pcs
    print(f"patchcore: {(time.time() - t0) / 60:.1f} min")

    T = pd.DataFrame(filas).set_index("enfoque")
    C = pd.DataFrame(configs)
    T.to_csv(RESULTADOS / "e09_enfoques.csv")
    C.to_csv(RESULTADOS / "e09_configs.csv", index=False)
    (RESULTADOS / "e09_dispositivo.json").write_text(json.dumps(
        {"dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
         "torch": torch.__version__}))
    return T, C


def main():
    estilo()
    csv = RESULTADOS / "e09_enfoques.csv"
    if csv.exists() and "--recalcular" not in sys.argv:
        T, C = pd.read_csv(csv, index_col="enfoque"), pd.read_csv(RESULTADOS / "e09_configs.csv")
    else:
        T, C = calcular()

    # Decisión N1 (mismo criterio que e08): detectar todas las malas reales (A y R) manda; luego menos
    # falsas alarmas; empate → la más rápida en CPU. El AUC se informa, no decide.
    cumple = T[(T.A_deteccion >= 1) & (T.R_deteccion >= 0.99)]
    base = cumple if len(cumple) else T
    falsas = base.A_falsas + base.R_falsas
    elegido = base[falsas <= falsas.min() + 1e-9].ms_cpu.idxmin()
    e = T.loc[elegido]
    decidir("enfoque", elegido,
            "100 % de malas reales detectadas (A y giros R); luego menos falsas alarmas; empate → más rápida en CPU "
            "(e09: redes con ajuste justo)",
            experimento="e09", A_deteccion=round(float(e.A_deteccion), 3), R_deteccion=round(float(e.R_deteccion), 3),
            A_auc=round(float(e.A_auc), 3), C_deteccion=round(float(e.C_deteccion), 3),
            ms_cpu=round(float(e.ms_cpu), 1),
            otros={k: dict(config=T.loc[k, "config"], A=round(float(T.loc[k, "A_deteccion"]), 3),
                           R=round(float(T.loc[k, "R_deteccion"]), 3), A_auc=round(float(T.loc[k, "A_auc"]), 3),
                           C=round(float(T.loc[k, "C_deteccion"]), 3)) for k in T.index if k != elegido})

    orden = list(T.index)
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6), sharey=True)
    for ax, p, titulo in zip(axs, "ARC", ["A · 21 reales, dejando una fuera", "R · las mismas, giradas",
                                          "C · 600 sintéticas"]):
        barras_h(ax, orden, [100 * T.loc[orden, f"{p}_deteccion"], 100 * T.loc[orden, f"{p}_falsas"]],
                 ["malas detectadas (%)", "buenas retiradas por error (%)"], colores=[AZUL, NARANJA],
                 elegido=elegido, fmt="{:.0f}", lim=(0, 118), leyenda=False)
        ax.set_title(titulo, fontsize=10)
    barras_h(axs[3], orden, [T.loc[orden, "A_auc"], T.loc[orden, "R_auc"], T.loc[orden, "C_auc"]],
             ["AUC en A", "AUC en R", "AUC en C"], colores=[AZUL, AQUA, TINTA2], elegido=elegido, fmt="{:.2f}",
             lim=(0.5, 1.12), leyenda=False)
    axs[3].set_title("AUC (1 = separa perfecto, sin umbral)", fontsize=10)
    h1, l1 = axs[0].get_legend_handles_labels()
    h2, l2 = axs[3].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("N1 con ajuste justo (e09) · clásica contra redes preentrenadas con sintéticas y PatchCore "
                 "con coreset",
                 x=0.01, ha="left", fontsize=12, y=1.03)
    guardar(fig, "e09_enfoques.png")

    fig, axs = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw=dict(wspace=0.55))
    for ax, red, titulo in zip(axs, ("supervisada", "patchcore"),
                               ("Red supervisada: lr × épocas", "PatchCore: capas × coreset")):
        c = C[C.red == red]
        nombres = list(c.config)
        el = c[c.elegida.astype(bool)].config.iloc[0]
        barras_h(ax, nombres, [c.auc_val, c.A_auc], ["validación sintética (elige)", "21 reales (solo informa)"],
                 colores=[AZUL, NARANJA], elegido=el, fmt="{:.3f}", lim=(0.5, 1.1))
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("AUC")
    guardar(fig, "e09_busqueda.png")
    print(T.drop(columns=["config"]).round(3).T.to_string())
    print(C.round(3).to_string())


if __name__ == "__main__":
    main()
