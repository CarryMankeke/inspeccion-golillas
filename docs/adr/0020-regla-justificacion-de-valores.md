# ADR 0020 · Regla: cada valor lleva su justificación

**Estado:** aceptada el 28 nov 2026, pedida por Camilo · **Fecha:** 2026-11-28 · **Nivel:** todo el repo · **Auditoría:** [valores.md](../valores.md)

## Contexto
El [ADR 0017](0017-n0-v2-estado-del-arte.md) sacó de normas y referencias los requisitos que deciden: la detección, las falsas y la latencia. Pero el repo usa muchos otros números que también pesan en el resultado: el α del umbral, el tope de horas de ajuste, el empate de 2 puntos, las constantes de la cadena clásica… Varios se fijaron por costumbre y no dicen de dónde salen. Un portafolio para la industria tiene que poder responder «¿por qué ese número?» para cada uno.

## Decisión
**Cada parámetro, umbral o valor de medición o comparación lleva su justificación**, en el ADR que lo fija o en el comentario del código que lo usa. Esta es la escala, de mejor a peor:

| Nivel | La justificación | Ejemplo en el repo |
|---|---|---|
| 1 | Sale de una norma, o es más exigente que ella | Falsas ≤ 5 % (criterio «aceptable» del AIAG MSA) |
| 2 | No hay norma, y se dice. Se cita una referencia: un fabricante o un paper | 100 ms por pieza (clasificadora de 600 piezas/min); McNemar exacto (Dietterich, 1998) |
| 3 | No hay norma ni referencia. Hay una razón estadística o un cálculo mostrado | α = 3 % con 289 que calibran (Bates et al., Prop. 1); 66 buenas para α/2 |
| 4 | Valor por defecto de la librería, dicho explícitamente y sin buscar otro | Lote de 32 y 5.000 pasos de Dinomaly (anomalib 2.7.1) |
| — | Sin justificación | Queda marcado «para la v6» en [valores.md](../valores.md) |

Reglas:
- **Un ADR nuevo trae su tabla «valor · de dónde sale · tipo de fuente»**, como la sección «De dónde salen los números» del ADR 0017.
- **Una fuente secundaria se dice**, por ejemplo una guía que cita una norma pagada que no se leyó.
- **Un valor sin justificación no se cambia en una corrida ya fijada.** Se marca y se resuelve en la versión siguiente, antes de ver su prueba. Así, los valores de e13 que no tienen justificación siguen como están, y la v6 los justifica o los cambia antes de su prueba sellada.

## Auditoría del 28 nov 2026
[valores.md](../valores.md) revisa 48 valores (o grupos de constantes que comparten razón) del código y de los ADR, incluidos los 3 del [ADR 0019](0019-v6-metodo-de-seleccion.md). 15 están justificados, 17 tienen una razón débil (sin norma ni referencia) y 16 no tienen justificación. Los que más pesan en el resultado, y que la v6 tiene que justificar antes de su prueba:

1. El empate de 2 puntos de detección (con 72–74 malas, 1 pieza son ≈ 1,4 puntos).
2. Las 50 piezas y 3 de calentamiento con que se mide la latencia.
3. «La clásica aplica si mide el 90 % o más de las buenas».
4. La regla de tamaño de Roboflow de 20 buenas y 20 malas de prueba, que no calza con el cálculo del propio ADR 0017 (hacen falta 149 malas para demostrar el 2 % con 95 % de confianza). Además, el margen de recorte del 10 % quedó sin uso cuando la pieza pasó a ser la foto entera.
5. El tope de 2 h de ajuste.
6. Las constantes de la cadena clásica (P1, P2 y P5 sin razón; P3 y P4 con una razón débil).

Volver al [árbol](../arbol.md).
