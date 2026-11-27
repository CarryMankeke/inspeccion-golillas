# Árbol de decisiones (N0 → N3)

El proyecto se diseñó de lo general a lo particular, sin saltar niveles:

- **N0**: qué se quiere lograr, con qué límites y cómo se sabe si se logró.
- **N1**: los enfoques posibles; se elige uno midiendo.
- **N2**: los procesos (pasos) del enfoque elegido.
- **N3**: por cada proceso, todas las técnicas candidatas, probadas con los mismos datos; gana la que cumple un criterio escrito antes de mirar el ranking.

Cada decisión tiene su registro (ADR) con la tabla de resultados, el gráfico y el script que lo reproduce.

```mermaid
flowchart TD
    N0["N0 · Retirar de la cinta las golillas defectuosas<br/>ADR 0001"]
    N0 --> L["N1 · Lenguaje: Python + OpenCV<br/>ADR 0002"]
    N0 --> B["N1 · Línea base: reglas del MATLAB original<br/>21/21 en sus fotos, 38 % de falsas fuera<br/>ADR 0003"]
    N0 --> E{"N1 · Enfoque<br/>ADR 0004"}
    E -->|"6/6 malas reales · 36 ms en CPU"| C["Visión clásica ★"]
    E -.->|"5/6 · AUC 1,0 (e09)"| RS["Red supervisada (ResNet-18)"]
    E -.->|"4/6 · AUC 0,96 (e09)"| PC["Anomalías con red (PatchCore)"]
    C --> D["Datos: 21 fotos reales + generador sintético<br/>ADR 0005"]
    D -.-> DP["Prueba ciega con fotos ajenas (e10)<br/>clásica 31 %, PatchCore 67 %<br/>ADR 0013"]
    DP -.-> V3["N1 v3 (e11): varios montajes<br/>ningún candidato cumple<br/>ADR 0014"]
    V3 -.-> V4["N1 v4 (propuesta): umbral conformal<br/>y regla que respeta las falsas alarmas<br/>ADR 0015"]
    C --> P1["P1 Preprocesado<br/>★ dividir por el fondo<br/>ADR 0006"]
    P1 --> P2["P2 Segmentación<br/>★ Otsu<br/>ADR 0007"]
    P2 --> P3["P3 Bordes<br/>★ subpíxel por gradiente<br/>ADR 0008"]
    P3 --> P4["P4 Medición<br/>★ círculo por RANSAC<br/>ADR 0009"]
    P4 --> P5["P5 Rasgos<br/>★ 4 números de forma<br/>ADR 0010"]
    P5 --> P6["P6 Decisión<br/>★ Mahalanobis + Hotelling<br/>ADR 0011"]
    P6 --> P7["P7 Línea<br/>★ encoder en la cinta<br/>ADR 0012"]
```

## N3: lo que se probó en cada proceso

| Proceso | Técnicas probadas | Elegida | Criterio | ADR |
|---|---|---|---|---|
| P1 Preprocesado | ninguno, gauss, mediana, bilateral, CLAHE, restar fondo, dividir por fondo, dividir + gauss (8) | dividir por el fondo | IoU medio de las 11 segmentaciones | [0006](adr/0006-p1-preproceso.md) |
| P2 Segmentación | Otsu, triángulo, adaptativo media/gauss, Bradley, Sauvola, k-means, Canny + relleno, GrabCut, watershed, Chan-Vese (11) | Otsu | mejor IoU con topología perfecta en buenas; empate → la más rápida | [0007](adr/0007-p2-segmentacion.md) |
| P3 Bordes | Sobel, Scharr, Prewitt, LoG, Canny, Canny automático, contorno, subpíxel iso, subpíxel gradiente (9) | subpíxel por gradiente | error de radio + dispersión de puntos | [0008](adr/0008-p3-bordes.md) |
| P4 Medición | Kåsa, Taubin, geométrico, robusto, RANSAC, elipse, mínimo círculo, momentos, Hough, Hough alt. (10) | RANSAC | error medio en buenas y con defecto | [0009](adr/0009-p4-medicion.md) |
| P5 Rasgos | 17 rasgos de forma, borde, Fourier y textura | 4 rasgos | selección hacia adelante por AUC | [0010](adr/0010-p5-rasgos.md) |
| P6 Decisión | tolerancias, Mahalanobis (χ² y Hotelling), Isolation Forest, OC-SVM, logística, árbol, SVM, random forest, k-NN (10) | Mahalanobis + Hotelling | 100 % de malas reales; menos falsas alarmas | [0011](adr/0011-p6-decision.md) |
| P7 Línea | temporizador, encoder, encoder + segundo sensor (3) | encoder | ≥ 99 % de malas retiradas y ≤ 1 % de buenas arrastradas | [0012](adr/0012-p7-linea.md) |

Volver al [índice](README.md).
