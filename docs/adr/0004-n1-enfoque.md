# ADR 0004 · N1: visión clásica o red neuronal

**Estado:** aceptada, en revisión (falta el ajuste justo de las redes, e09) · **Fecha:** 2026-11-27 · **Nivel:** N1 · **Script:** `experimentos/e08_enfoques.py`

## Contexto
Hoy muchas inspecciones visuales se resuelven con redes neuronales. Antes de quedarse con la visión clásica había que medirla contra ellas con las mismas pruebas.

## Opciones
1. **Visión clásica**: la cadena de 6 procesos de los ADR 0006 a 0011 (decide con Mahalanobis).
2. **Red supervisada**: ResNet-18 preentrenada en ImageNet, ajustada para "buena/mala" con giros, espejos y cambios de brillo. Necesita ejemplos de malas.
3. **Anomalías con red (PatchCore)**: la misma ResNet-18 sin reentrenar. Guarda trozos de las imágenes buenas y marca una pieza si algún trozo no se parece a ninguno. Solo necesita buenas.

Las dos redes reciben el recorte centrado en la golilla que entrega la segmentación clásica.

## Criterio
Primero, detectar el 100 % de las malas reales (prueba A dejando una fuera y prueba R con giros). Luego, menos falsas alarmas. Si empatan, la más rápida en CPU, porque el PC de la línea no tiene por qué tener GPU.

## Resultado
| Enfoque | A: malas | A: falsas | R: malas | R: falsas | C: malas | C: falsas | ms CPU | ms GPU |
|---|---|---|---|---|---|---|---|---|
| **Clásica ★** | **6/6** | **0/15** | **100 %** | **0 %** | 100 % | 3,0 % | 36 | — |
| Red supervisada | 5/6 | 0/15 | 88 % | 2,9 % | 98 % | 0,7 % | 23 | 5 |
| PatchCore | 4/6 | 0/15 | 67 % | 0 % | 78 % | 0 % | 122 | 13 |

![Clásica contra redes](../figuras/e08_enfoques.png)

## Decisión
**Visión clásica.** Es la única que detecta todas las malas reales. La red supervisada aprende con solo 5 malas por vuelta de la prueba A, muy pocas para una red. PatchCore dejó pasar 2 de 6; una explicación probable, **no verificada**, es que los defectos chicos se diluyen al reducir el recorte a 224 px.

## Límites de esta comparación
Las redes no tuvieron un ajuste justo: tasa de aprendizaje y épocas fijas, entrenadas con 20 fotos reales; PatchCore sin coreset y con un umbral simple (máximo + 2σ); y no se informó el AUC, solo un punto de operación. **"La clásica gana" vale en ese marco.** El experimento e09 repetirá la comparación preentrenando con sintéticas, buscando hiperparámetros en validación sintética (no en las reales), probando capas y coreset en PatchCore e informando el AUC. Si cambia el resultado, este ADR se reemplaza.

## Consecuencias
- La cadena clásica explica sus decisiones ("borde del agujero dañado"); las redes no.
- Con cientos de fotos de malas, la red supervisada podría ganar: en sintéticas, donde tuvo 300 malas, llegó al 98 %. Vale la pena repetir esta comparación cuando haya más datos reales.

Volver al [árbol](../arbol.md).
