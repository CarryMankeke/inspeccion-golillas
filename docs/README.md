# Documentación · Inspección de golillas

Esta carpeta se lee en GitHub o se abre como bóveda de [Obsidian](https://obsidian.md) (abrir la carpeta `docs/`). Los enlaces son relativos, así que funcionan en los dos.

## Por dónde empezar
1. [Árbol de decisiones](arbol.md): el diseño completo en un diagrama, de lo general (N0) a cada técnica probada (N3).
2. [Glosario](glosario.md): los términos técnicos en una línea cada uno.
3. Los registros de decisión (ADR), uno por decisión:

| ADR | Decisión | Evidencia clave |
|---|---|---|
| [0001](adr/0001-n0-objetivo.md) | N0 · objetivo, límites y criterio de éxito | metas y resultado final |
| [0002](adr/0002-n1-lenguaje.md) | N1 · Python + OpenCV | por restricciones (no medida) |
| [0003](adr/0003-n1-linea-base-matlab.md) | N1 · línea base: reglas del MATLAB original | 21/21 en sus fotos; 38 % de falsas en sintéticas |
| [0004](adr/0004-n1-enfoque.md) | N1 · visión clásica antes que redes (en revisión) | 6/6 contra 5/6 y 4/6 con ajuste justo (e09); en la prueba ciega, 31 % contra 67 % de PatchCore (e10) |
| [0005](adr/0005-datos-sinteticos.md) | datos: 21 reales + generador sintético | verdad exacta; límites sim2real |
| [0006](adr/0006-p1-preproceso.md) | P1 · dividir por el fondo | IoU medio 0,946 |
| [0007](adr/0007-p2-segmentacion.md) | P2 · Otsu | IoU 0,975 en 4 ms |
| [0008](adr/0008-p3-bordes.md) | P3 · bordes subpíxel por gradiente | dispersión 0,16 px |
| [0009](adr/0009-p4-medicion.md) | P4 · círculo por RANSAC | 0,26 px con defectos |
| [0010](adr/0010-p5-rasgos.md) | P5 · 4 rasgos de forma | AUC 0,998 sintético, 1,0 real |
| [0011](adr/0011-p6-decision.md) | P6 · Mahalanobis + Hotelling | 6/6 malas, 0/15 falsas |
| [0012](adr/0012-p7-linea.md) | P7 · encoder en la cinta | 100 % retiradas hasta 1.500 mm/s |
| [0013](adr/0013-datos-publicos.md) | datos públicos y prueba ciega | Edge Impulse: clásica 31 %, PatchCore 67 % (AUC 0,75 contra 0,95) |
| [0014](adr/0014-n1-v3-varios-montajes.md) | N1 v3 · varios montajes | ningún candidato cumple (e11) |
| [0015](adr/0015-n1-v4-opciones.md) | N1 v4 · umbral conformal | PatchCore 86/100 en Kaggle sin falsas; ningún candidato cumple (e12) |
| [0016](adr/0016-n1-v5-fotos-propias.md) | N1 v5 · un candidato fijado y 180 fotos propias (rechazada) | la v5 va con datasets públicos de varios tipos de golilla; ADR por escribir |

## Cómo está escrito cada ADR
Contexto → opciones → criterio (escrito en el script antes de ver el ranking) → tabla de resultados → decisión → revisiones, si el criterio o el código se corrigió. Cada gráfico sale de un script en `experimentos/`, y cada decisión queda también en `resultados/decisiones.json`.

## Figuras
Están en [figuras/](figuras/). Las genera `experimentos/`; no se editan a mano.
