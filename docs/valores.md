# De dónde sale cada valor (auditoría del 28 nov 2026)

Regla del [ADR 0020](adr/0020-regla-justificacion-de-valores.md): cada parámetro, umbral o valor de medición o comparación lleva su justificación. Lo ideal es que salga de una norma o sea más exigente que ella; si no hay norma, se dice y se cita la referencia o la razón estadística. Lo que no tiene justificación queda **para la v6**, sin cambiar e13.

Auditoría de solo lectura sobre el código y los ADR (repo en el commit e53bc3e). Las justificaciones son las que da el propio repo. «ADR 0017 §X» es una sección de [ADR 0017](adr/0017-n0-v2-estado-del-arte.md). Estado: **justificado**, **débil** (hay razón, pero sin norma ni referencia) o **para la v6** (sin justificación).

| # | Valor | Dónde se usa | De dónde sale | Tipo de fuente | Estado |
|---|---|---|---|---|---|
| 1 | α = 3 % (umbral conformal) y mínimo de 33 buenas que calibran | `e13` `ALFA`; `decision.umbral_conformal` | ADR 0017 §Protocolo 6: con 289 que calibran la tasa real sigue Beta(8, 282), media 2,8 %, pasa del 5 % con prob. 2 % (Bates et al., Prop. 1, arxiv 2104.08279). 33 = α ≥ 1/(n+1) (Angelopoulos y Bates, apéndice D; ADR 0015). Queda bajo el 5 % del AIAG | razón estadística | justificado |
| 2 | Fórmula del umbral: puntaje ⌈(n+1)(1−α)⌉ | `decision.umbral_conformal` | Docstring: Angelopoulos y Bates §1 y apéndice D (arxiv 2107.07511) | paper | justificado |
| 3 | α/2 = 1,5 % por parte en la fusión | ADR 0015/0016 (e13 no fusiona) | ADR 0015: «para que las dos juntas no pasen del 3 %» (cota de unión) | razón estadística | justificado |
| 4 | Los 66 de α/2 (mínimo de buenas con α/2) | ADR 0015 §Tamaño de la calibración; regla Roboflow | ADR 0015: 1/(n+1) ≤ 1,5 % ⇒ n ≥ 66 | razón estadística | justificado |
| 5 | No detección ≤ 2 % y falsas ≤ 5 % (clase «aceptable») | `decision.AIAG`, `clase_aiag`; `e13.elegir` | ADR 0017 «De dónde salen los números»: AIAG MSA 4.ª ed., números tomados de guías secundarias (TechieQuality 2024); el ADR admite que el manual es pagado y no se leyó | norma (secundaria) | justificado |
| 6 | Clase «marginal» (≤ 5 % y ≤ 10 %) y efectividad ≥ 90 % (marginal ≥ 80 %) | `decision.AIAG`; ADR 0017; `estado-del-arte.md` §6 | El mismo criterio AIAG, misma fuente secundaria | norma (secundaria) | justificado |
| 7 | 100 ms por pieza (600 piezas/min) | `e13` `MS_MAX` | ADR 0017: clasificadora óptica de golillas hasta 600 pzas/min (Fastener+Fixing 2021) y guía Keyence de 20–100 ms; «ninguna norma fija la latencia» | fabricante | justificado |
| 8 | 50 piezas para medir latencia, mediana, 3 de calentamiento | `e13.mediana_ms`, `N_LATENCIA` | ADR 0017 §Detalles: solo lo describe, sin razón | sin justificación | para la v6 |
| 9 | 4 hilos de CPU | `e13.latencias` | ADR 0017: «CPU del servidor (Intel i3-9100, 4 núcleos)». Razón de hardware, sin referencia de que sea la CPU objetivo de una línea | convención propia | débil |
| 10 | Cota de latencia: 15 corridas + 3 de calentamiento; entradas 224/256/252/240/392 px | `scripts/latencia_cota.py` | Docstring lo declara; ADR 0017 §Medición previa lista los tamaños sin razón. No decide | sin justificación | para la v6 |
| 11 | Tope de 2 h de ajuste por tipo | `ia.TOPE_S`; ADR 0017 §Protocolo 4 | Solo enunciado | sin justificación | para la v6 |
| 12 | Empate de 2 puntos de detección | `e13.elegir` (`max − 0.02`) | ADR 0017 §Elección: «2 puntos o menos», sin razón. Con 72–74 malas, 1 pieza ≈ 1,4 puntos | sin justificación | para la v6 |
| 13 | Orden de desempate: clásica, sin entrenamiento, con entrenamiento, texto; luego el más rápido | `metodos.EXPLICA`; `e13.elegir` | ADR 0017: «gana el más fácil de explicar». Hay razón, sin referencia | convención propia | débil |
| 14 | Roboflow: 66 buenas en train+valid (33 ajustan, 33 calibran) | `sello.REGLA` | ADR 0017 §Regla de tamaño: «el mínimo para α = 3 %» (ver fila 1) | razón estadística | justificado |
| 15 | Roboflow: ≥ 20 buenas y ≥ 20 malas de prueba | `sello.REGLA` | Sin razón. El propio ADR 0017 calcula que probar no detección ≤ 2 % con 95 % de confianza pide ≥ 149 malas | sin justificación | para la v6 |
| 16 | Margen de recorte del 10 % en Roboflow | ADR 0017 §Qué es una pieza (párrafo original) | Sin razón; además quedó sin uso: el cambio del 28 nov pasó la pieza a «foto entera» y el código no lo aplica | sin justificación | para la v6 |
| 17 | 1.024 px de lado máximo | `conjuntos.LADO_MAX` | Comentario: renuka-p trae 4.024 × 3.036. Razón del 1.024, ninguna. Afecta solo a sets que informan | sin justificación | para la v6 |
| 18 | Lote de 32 | `ia.LOTE`; PatchCore propio (`ajustar`, bloques de 32) | Comentario: `train_batch_size` por defecto de los datamodules de anomalib | valor por defecto de la librería | justificado |
| 19 | 5.000 pasos de Dinomaly y su receta (lr 2e-3, calentamiento 100, caída coseno a 2e-4, recorte 0,1) | `ia.PASOS_DINOMALY`, `TRAINING_CONFIG` | Comentario: `MAX_STEPS_DEFAULT` de Dinomaly en anomalib 2.7.1 | valor por defecto de la librería | justificado |
| 20 | Semilla 0 (partición, submuestras, bootstrap, giros, coreset, RANSAC, Dinomaly) | `sello.SEMILLA`, `conjuntos.submuestra`, `kcentro`, `ransac`, etc. | Ninguna razón, solo reproducibilidad implícita | convención propia | débil |
| 21 | 2.000 remuestreos del IC del AUC | `e09.auc_ic(n=2000)` | ADR 0017: «(2.000, semilla 0)» sin razón. Solo informa | sin justificación | para la v6 |
| 22 | IC al 95 % (Clopper-Pearson exacto para tasas; bootstrap percentil 2,5–97,5 para el AUC) | `decision.ic_exacto`; `e09.auc_ic` | ADR 0017: informar intervalos según ISO/IEC TS 4213 (solo el índice leído). El 95 % no se argumenta | razón estadística | débil |
| 23 | Radio exterior mediano de 50 px (escala de la clásica) | `clasica.RADIO_OBJETIVO` | ADR 0017: «la del N0 v1» (golillas de las 21 fotos propias, radio ≈ 50 px; Kaggle ≈ 53) | convención propia | débil |
| 24 | «La clásica aplica si mide ≥ 90 %» | `clasica.MIDE_MIN` | ADR 0017 §Protocolo 2: solo enunciado | sin justificación | para la v6 |
| 25 | Coreset 1 % (PatchCore propio) | `PatchCorePropio(coreset=0.01)` | ADR 0017: «el de e09 a e12»; e09 lo eligió entre {100, 10, 1 %} con datos de desarrollo | convención propia | débil |
| 26 | Coreset 10 % (PatchCore anomalib) | `ia.Anomalib("patchcore")` | ADR 0017 §Métodos: «configuración por defecto» de anomalib, sin cita del valor | valor por defecto de la librería | justificado |
| 27 | 5 vistas de MANTA | `conjuntos.VISTAS_MANTA` | Formato del PNG (1.280 × 256); `vistas_manta` falla si no calza. El repo no cita dónde lo dicen los autores | paper | débil |
| 28 | 256 → 224 px en el PatchCore propio, sin recorte | `patchcore_propio.LADO`, `a_tensor` | ADR 0017: «las vistas ya vienen centradas»; ADR 0016: «el de e08 a e12» (continuidad con el control) | convención propia | débil |
| 29 | 289/289 (roja) y 292/292 (amarillo-verde) en ajusta/calibra | `sello.mitades` | Mitad y mitad de 578 y 584 (ADR 0017 §Protocolo 3). La razón de partir 50/50 no se da; 289 ≥ 33 cumple fila 1 | convención propia | débil |
| 30 | Conteos de MANTA: 578/584 de entrenamiento; prueba 148+74 y 144+72 | `sello.MANTA_TIPOS` | `distribution.md` de figshare, verificado contra las carpetas (`ValueError` si no calzan) | fabricante | justificado |
| 31 | Mitad si no cabe: submuestra de piezas, lote a la mitad | `ia._banco`, `ia._entrenar` | ADR 0017 §Protocolo 4: regla de recursos, sin razón de la mitad | convención propia | débil |
| 32 | Perfil: polaridad con percentiles 0,2/50/99,8 y mayoría > 50 %; vista de arriba = la que más mide, desempate por excentricidad | `clasica.golilla_clara`, `perfil` | «regla de e10» sin etiquetas (ADR 0014); sin referencia externa | convención propia | débil |
| 33 | Cadena P1: cierre 17×17 a ¼ de escala, mediana 5, Gauss σ=2, ×200 | `preproceso.py` | ADR 0006 eligió el método (IoU), no estas constantes | sin justificación | para la v6 |
| 34 | Cadena P2: Otsu; limpieza con 5 % del área y peso 0,05 en piezas que tocan el borde | `segmentacion.py` | ADR 0007: Otsu por IoU y velocidad; las constantes de `limpiar` no tienen razón | sin justificación | para la v6 |
| 35 | Cadena P3: banda ±3 px (elipse 7×7), Canny a 0,5·p95 del gradiente y 0,4 de ese alto, corrección parabólica ±0,5 px | `bordes.py` | Docstring de `_umbrales_banda`: umbral según el contraste del propio borde (corrección 27 nov). Los factores 0,5/0,4 y p95 no tienen razón; ADR 0008 eligió el método | convención propia | débil |
| 36 | Cadena P4: RANSAC con 300 hipótesis y tolerancia 1,5 px | `medicion.ransac` | ADR 0009 los fija y los prueba en sintéticas; no explica por qué esos números | convención propia | débil |
| 37 | Cadena P5: armónicos 2–16, 128 bins, hueco > 20 px, K5, erosión | `rasgos.py` | Docstrings describen qué hacen, no por qué esos valores | sin justificación | para la v6 |
| 38 | 4 rasgos de `calibracion.json` y Mahalanobis (StandardScaler + Ledoit-Wolf) | `clasica.COLUMNAS`, `decision.Mahalanobis` | ADR 0010: selección hacia adelante, parar con mejora < 0,002 o 6 rasgos (valores sin razón); ADR 0011: Mahalanobis por A/R/B/C | convención propia | débil |
| 39 | p = 0,999 de Mahalanobis (χ²/Hotelling) | `decision.Mahalanobis` | ADR 0011 justifica Hotelling, no el 0,999. En e13 el umbral es conformal y ese `t` no se usa | sin justificación | para la v6 |
| 40 | Otros modelos de `decision.py`: k=6 MAD, IsolationForest 200, OCSVM ν=0,05, árbol prof. 3, kNN 3, RF 200 | `decision.modelos()` | ADR 0011 compara modelos; los hiperparámetros no se argumentan. e13 no los usa | sin justificación | para la v6 |
| 41 | PatchCore propio: 8 simetrías si hay < 50 vistas | `patchcore_propio.POCAS` | Comentario: «como e09 y e11» | convención propia | débil |
| 42 | PatchCore propio: avg-pool 3, proyección a 128 dim, capas l2+l3, ResNet-18 | `patchcore_propio.py` | Comentario: «como PatchCore» (Roth et al.), sin la cita del valor | paper | débil |
| 43 | WinCLIP: `k_shot = 0`, clase «washer» | `ia.CLASE_WINCLIP` | ADR 0017: valor por defecto, «sin buscar otro» | valor por defecto de la librería | justificado |
| 44 | Pocas muestras: 1, 4, 16, 64 piezas y 5 semillas | `e13.POCAS_N`, `POCAS_SEMILLAS` | ADR 0017 §Cómo se informa: solo enunciado. Solo informa | sin justificación | para la v6 |
| 45 | Humo: 40 piezas por clase, 20 pasos de Dinomaly, 5 piezas de latencia | `e13` `--humo` | Prueba de funcionamiento, no decide ni se publica | convención propia | débil |
| 46 | p < 0,05 (McNemar exacto, dos colas) para la dominancia | `scripts/combinaciones.py` | ADR 0019: prueba estándar para comparar dos clasificadores en el mismo conjunto (Dietterich, 1998, *Neural Computation* 10(7)); el 95 % es la confianza de los IC del ADR 0017 | paper | justificado |
| 47 | Punto ROC con ≤ floor(5 % · buenas) falsas | `scripts/combinaciones.py` | ADR 0019: el criterio de falsas del AIAG (fila 5) | norma (secundaria) | justificado |
| 48 | 1 a 3 finalistas en la prueba sellada | ADR 0019 | Sin razón todavía | sin justificación | para la v6 |

Constantes ignoradas por no afectar resultados: tamaños de figura, colores, fuentes, trozos de memoria (65.536), tolerancias numéricas (1e-12, 1e-9).

## Valores «para la v6», de más a menos peso en el resultado

1. **Empate de 2 puntos (12).** Decide quién gana entre los que cumplen. Fuente posible (por verificar): IC exacto de la detección con 72–74 malas, o la prueba de hipótesis de la VDI 2632-3.1; calculable con `ic_exacto`.
2. **50 piezas y 3 de calentamiento (8).** Decide si un método pasa los 100 ms (PatchCore propio y PaDiM andan cerca según la cota). Fuente posible (por verificar): IC no paramétrico de la mediana con n = 50 (razón estadística), y la latencia en ISO/IEC TS 4213.
3. **«La clásica aplica si mide ≥ 90 %» (24).** Puede excluir a la clásica de un tipo. Fuente posible (por verificar): reusar el 90 % de efectividad del AIAG, o VDI 2632-3.
4. **Roboflow ≥ 20 buenas y ≥ 20 malas (15) y margen del 10 % (16).** El 20/20 define qué sets deciden y contradice el cálculo de 149 malas del propio ADR; derivarlo de ese cálculo o de la VDI 2632-3.1 (por verificar). El 10 % está obsoleto: conviene borrarlo del ADR.
5. **Tope de 2 h (11).** Afecta a Dinomaly. Fuente posible: múltiplo del tiempo de ajuste medido en la corrida de humo (razón empírica), por verificar.
6. **Constantes de la cadena clásica P1–P5 (33, 34, 37).** Fijan si la clásica mide. Fuentes posibles (por verificar): documentación de OpenCV y los papers de Canny, Otsu y Fischler-Bolles; o derivarlas de la dispersión medida (0,155 px, ADR 0008) para fijar tolerancias.
7. **p = 0,999 y modelos extra de `decision.py` (39, 40).** No entran a la decisión de e13: retirarlos o marcarlos como fuera de alcance.
8. **1.024 px (17).** Solo sets informativos. Fuente posible: ser mayor que el lado de entrada de cualquier red (224–392 px), por verificar.
9. **2.000 remuestreos (21) y pocas muestras 1/4/16/64 y 5 semillas (44).** Solo informan. Fuente posible (por verificar): Efron y Tibshirani para el bootstrap; el rango 1–64 de AnomalyDINO/WinCLIP para la curva.
10. **Cota de latencia: 15 corridas y tamaños de entrada (10).** No decide. Alinear con la regla de la fila 8 y citar el tamaño por defecto de cada red en anomalib.
