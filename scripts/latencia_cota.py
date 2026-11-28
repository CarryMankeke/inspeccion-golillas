"""Cota inferior de la latencia en CPU de cada método del ADR 0017 (medición previa, no decide).

Mide solo la red base de cada método, con pesos al azar: el tiempo no depende de los pesos, y
así no se baja nada. El método completo tarda esto más su búsqueda o su decodificador.
Mediana de 15 corridas después de 3 de calentamiento, con 4 hilos, para 1 y 5 vistas.

Uso (en el servidor, con el extra `sota` instalado):  CUDA_VISIBLE_DEVICES="" python scripts/latencia_cota.py
"""

from __future__ import annotations

import statistics
import time

import open_clip
import timm
import torch
import torchvision


def resnet18_hasta_layer3():
    m = torchvision.models.resnet18(weights=None)
    return torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)


def clip_imagen():
    return open_clip.create_model("ViT-B-16-plus-240", pretrained=None).visual


REDES = {
    "PatchCore propio (ResNet-18 hasta layer3, 224 px)": (resnet18_hasta_layer3, 224),
    "PatchCore (WideResNet-50, layer2+3, 256 px)": (
        lambda: timm.create_model("wide_resnet50_2", pretrained=False, features_only=True, out_indices=(2, 3)), 256),
    "PaDiM (ResNet-18, layer1-3, 256 px)": (
        lambda: timm.create_model("resnet18", pretrained=False, features_only=True, out_indices=(1, 2, 3)), 256),
    "AnomalyDINO (DINOv2 ViT-S/14, 252 px)": (
        lambda: timm.create_model("vit_small_patch14_dinov2", pretrained=False, img_size=252), 252),
    "Dinomaly, solo el codificador (DINOv2-reg ViT-B/14, 392 px)": (
        lambda: timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False, img_size=392), 392),
    "WinCLIP, una pasada del codificador (ViT-B-16-plus, 240 px)": (clip_imagen, 240),
}


def mediana_ms(red, x, calentar=3, veces=15):
    with torch.inference_mode():
        for _ in range(calentar):
            red(x)
        t = []
        for _ in range(veces):
            t0 = time.perf_counter()
            red(x)
            t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    for nombre, (crear, lado) in REDES.items():
        red = crear().eval()
        ms = [mediana_ms(red, torch.randn(b, 3, lado, lado)) for b in (1, 5)]
        print(f"{nombre} | {ms[0]:.0f} ms (1 vista) | {ms[1]:.0f} ms (5 vistas)", flush=True)


if __name__ == "__main__":
    main()
