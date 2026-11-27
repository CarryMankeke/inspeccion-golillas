# ADR 0001 · N0: objetivo, límites y criterio de éxito

**Estado:** aceptada · **Fecha:** 2026-11-27 · **Nivel:** N0

## Objetivo
Detectar con una cámara las golillas defectuosas que pasan por una cinta transportadora y sacarlas de la cinta con un soplo de aire, sin tocar las buenas.

## Límites
- Una golilla por foto, vista desde arriba, sobre un fondo claro. La cámara está a altura fija (radio exterior ≈ 50 px).
- Datos reales: **21 fotos** del proyecto MATLAB original (15 buenas y 6 malas: 3 deformadas, 1 con el borde del agujero dañado, 1 con corte radial y 1 con muesca exterior). Son pocas, así que todo se mide también con golillas sintéticas ([ADR 0005](0005-datos-sinteticos.md)).
- El computador de la línea no tiene por qué tener GPU. El cómputo pesado del proyecto corre en un servidor con una GTX 1060.
- No hay una línea física: la cinta, el encoder y el soplador se simulan ([ADR 0012](0012-p7-linea.md)).

## Criterio de éxito
| Qué | Meta | Resultado |
|---|---|---|
| Malas reales detectadas (dejando una fuera) | 6/6 | **6/6** (IC95 54–100 %) |
| Buenas reales retiradas por error | ≤ 1/15 | **0/15** (IC95 0–22 %) |
| Las mismas fotos giradas | 100 % y 0 % | **42/42 y 0/105** |
| Sintéticas (600 de prueba) | ≥ 99 % detectadas, ≤ 5 % falsas | **100 % y 3 %** |
| Tiempo por pieza en CPU | < 100 ms | **≈ 36 ms** |
| Línea simulada | ≥ 99 % de malas marcadas retiradas, ≤ 1 % de buenas arrastradas | **100 % y 0 % hasta 1.500 mm/s** |

Los intervalos son anchos porque 6 y 15 fotos son pocas: el resultado real es "ninguna falla en esta muestra", no "nunca falla". Además, las 21 fotos reales sirvieron para elegir técnicas (por ejemplo, el modelo del proceso 6) y también para informar estos números. Son una **validación, no una prueba ciega**: hace falta un set real nuevo que no se use para decidir nada (ver "Limitaciones y próximos pasos" en el README).

## Consecuencias
- Cada proceso (N3) se decide con una métrica y un criterio escritos en su script, y el resultado queda en `resultados/decisiones.json`.
- Antes de comparar técnicas se fijó contra qué ganarle: las reglas del MATLAB original ([ADR 0003](0003-n1-linea-base-matlab.md)).

Volver al [árbol](../arbol.md).
