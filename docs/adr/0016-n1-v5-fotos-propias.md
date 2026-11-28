# ADR 0016 · N1 v5: un candidato fijado y una sesión de fotos propias

**Estado:** rechazada el 27 nov 2026, sin que se tomara ninguna foto. Camilo decidió que la v5 no usa fotos propias: va solo con datasets públicos de golillas buenas y malas, y el proyecto se adapta a varios tipos de golilla. El plan nuevo va en el [ADR 0017](0017-n0-v2-estado-del-arte.md). Este queda como registro, con su lista y su código del sello sin usar · **Fecha:** 2026-11-27 · **Nivel:** N1 · **Código:** `golillas/sesion.py` y `scripts/sesion_v5.py` (lista de tomas y sello); la prueba ciega va en `experimentos/e13_n1_v5.py`, por escribir

## Contexto
En la v4 (e12, [ADR 0015](0015-n1-v4-opciones.md)) ningún candidato cumplió. El umbral conformal arregló a PatchCore en Kaggle: pasó de 55 a 86 malas de 100, sin falsas. Pero el requisito del montaje propio se mide con 15 buenas, y con 15 no existe un umbral conformal del 3 % (hacen falta 33). Lo que bloquea ya no es el método: faltan buenas propias para calibrar.

Después de e12, un análisis post hoc de las 12 fusiones posibles encontró una sola que cumple los dos requisitos: **clasica_4 + PatchCore l2+l3 fijo**, y al límite (1/15 buenas propias y 5/100 en Kaggle). Como salió de mirar 12 combinaciones, sus números son optimistas, y solo datos nuevos pueden confirmarlos.

El 27 nov, Camilo decidió (D1) que la v5 va con una sola sesión de fotos propias y ese candidato fijado de antemano.

## Decisión
1. **Un solo candidato, sin elegir.** La v5 no compara candidatos: la elección ya se hizo con los datos de la v4. Las fotos nuevas solo ajustan, calibran y prueban.
2. **Una sola sesión de 180 fotos propias**, en un día y con un montaje: 20 buenas ajustan, 100 calibran, y 30 buenas y 30 malas prueban.
3. **El criterio queda escrito aquí antes de la primera foto**, y la prueba ciega se corre una sola vez (e13).

## El candidato (fijo)
Es una fusión: la golilla se retira si la clásica **o** PatchCore la retiran. Cada parte usa su umbral conformal con α/2 = 1,5 %, para que juntas no pasen del 3 %. Es la fusión del análisis post hoc del ADR 0015, con un solo cambio: en el montaje propio también usa umbral conformal, porque ahora habrá buenas para calibrarlo.

| Parte | Qué es | Ajusta con | Umbral |
|---|---|---|---|
| clasica_4 | La cadena P1–P5 ([ADR 0006](0006-p1-preproceso.md) a [0010](0010-p5-rasgos.md)) con sus 4 rasgos: relleno del anillo, excentricidad, Fourier exterior y desviación interior p95. El puntaje es la distancia de Mahalanobis (escalado y Ledoit-Wolf) | La media y la covarianza de las 20 que ajustan | Conformal con α/2, calibrado con las 100 que calibran |
| PatchCore l2+l3 fijo | ResNet-18 de ImageNet, capas layer2 y layer3 y coreset del 1 %. El recorte es de 160 px alrededor del centro de la segmentación, llevado a 224 px (el de e08 a e12) | El banco de las 20 que ajustan, con sus 8 simetrías (regla de e11: menos de 50 buenas) | Conformal con α/2, calibrado con las 100 que calibran |

Lo demás no cambia. Todas las fotos pasan por la regla de polaridad P1 de e10. Si la cadena clásica no logra medir una golilla, la retira siempre, y esa golilla no calibra. Ninguna mala ajusta ni calibra nada.

**Qué promete el umbral.** Con 100 buenas que calibran y α/2 = 1,5 %, cada umbral es el **mayor** puntaje de las 100, porque ⌈101 · 0,985⌉ = 100. Si una buena nueva sale del mismo montaje, supera ese umbral con probabilidad 1/101 ≈ 1 %, así que la fusión retira 2 % o menos de las buenas. En una sesión dada, la tasa real de cada parte sigue una Beta(1, 100): su percentil 95 es 3,0 %, y la probabilidad de pasar del 5 % es 0,6 % ([Bates et al., Prop. 1](https://arxiv.org/abs/2104.08279)). Si la cadena clásica no lograra medir más de 34 de las 100 que calibran, su umbral no existiría, porque hacen falta 66 o más. En ese caso la clásica solo retiraría las golillas que no mide, y eso se informaría.

## Qué foto sirve para qué
| Grupo | Fotos | Para qué | Por qué ese número |
|---|---|---|---|
| Ajustan | 20 buenas | La media y la covarianza de la clásica, y el banco de PatchCore | Es el diseño aprobado en D1; con las 8 simetrías, el banco tiene 160 vistas |
| Calibran | 100 buenas | Los dos umbrales de α/2 | Con α/2 hacen falta 66 o más. Con 100, la probabilidad de que la fusión pase del 5 % baja de 15 % a 4 % ([ADR 0015](0015-n1-v4-opciones.md), «Tamaño de la calibración propia») |
| Prueba | 30 buenas y 30 malas | Medir, una sola vez (e13) | 6 malas por tipo de defecto |
| Separadores | 18 papelitos | Una foto a un papelito (B01 a B18) al empezar cada fila de 10 tomas | Para calzar cada foto con su toma sin mirarla |

Las malas son 6 cortadas (corte radial), 6 dobladas, 6 con muesca (3 en el borde exterior y 3 en el del agujero), 6 de otra medida y 6 rayadas. Cada foto es una golilla distinta: ninguna se fotografía dos veces.

Las 21 fotos viejas no se mezclan con las nuevas. Son de otro día y otra cámara, y mezclarlas rompe la garantía del umbral. Solo sirven de control del código (ver «Sellado y orden de lectura»).

## Criterio (fijado antes de tomar las fotos)
La fusión se adopta si, en la prueba ciega (e13, una sola corrida), cumple los tres requisitos:

1. **Detección:** retira las **18 malas de forma**, es decir, las cortadas, las dobladas y las con muesca. Si retira las 18, el IC95 exacto de la detección va de 81,5 % a 100 %.
2. **Falsas alarmas:** retira **2 o menos de las 30 buenas** de prueba. Es la misma tasa del N0, 1 de cada 15. Con 2 de 30, el IC95 exacto va de 0,8 % a 22 %.
3. **Latencia:** menos de 100 ms por pieza en la CPU de norea, medida con una de las fotos que ajustan.

Las 6 de otra medida y las 6 rayadas **se informan, pero no deciden**. Las rayadas no cambian la forma, igual que en el [ADR 0014](0014-n1-v3-varios-montajes.md). Las de otra medida tampoco cambian la forma: solo el tamaño. El candidato no tiene un rasgo de tamaño (clasica_4 no usa el radio exterior), y el N0 nunca midió tamaño. Se fotografían para medir esos dos límites.

**Por qué estos umbrales.** El N0 ([ADR 0001](0001-n0-objetivo.md)) pide todas las malas (6/6) y a lo más 1 de 15 buenas retiradas. Aquí se pide lo mismo, con el triple de piezas.

**Qué tan probable es pasar.** Con la binomial, según la tasa real del detector:
- Si detecta de verdad el 99 % de las malas de forma, pasa el requisito 1 con probabilidad 84 %. Con 97 % pasa con 58 %; con 95 %, con 40 %; y con 90 %, con 15 %.
- Si retira de verdad el 2 % de las buenas, que es lo que promete el umbral, pasa el requisito 2 con probabilidad 98 %. Con 5 % pasa con 81 %, y con 10 %, con 41 %.

La prueba es exigente con la detección a propósito: el N0 y la línea simulada piden que salgan 99 % o más de las malas. Un detector del 95 % pasa solo 4 de cada 10 veces.

**Qué se hace con el resultado:**
- **Si cumple los tres,** se adopta. `decisiones.json` guarda `enfoque_v5 = fusion` (clasica_4 + patchcore_l2+l3_fijo) con `adoptado = true`, y el [ADR 0004](0004-n1-enfoque.md) sale de revisión con esta fusión.
- **Si falla alguno,** no se adopta, y el ADR 0004 sigue en revisión. Las 60 fotos de prueba quedan gastadas: una v6 necesita un ADR nuevo y fotos de prueba nuevas. Las 120 buenas que ajustan y calibran sí pueden reusarse para desarrollar.
- **No hay regla de respaldo:** con un solo candidato, el resultado es «cumple» o «no cumple».

**Solo para informar**, después de guardar la decisión: la detección por tipo de defecto, con su IC95; el AUC de la fusión; cada parte sola, con umbral conformal de α = 3 %; y cuántas de las que calibran no logró medir la clásica. Los números del candidato en Kaggle y en Edge Impulse ya están en el ADR 0015 (post hoc), y no se repiten porque no son datos nuevos.

## Protocolo de la sesión
La guía paso a paso para Camilo, con la lista impresa, está en [sesion-fotos-v5.md](../sesion-fotos-v5.md). El protocolo fija esto:
- **Un día y un montaje.** La golilla va sobre papel blanco mate, con el celular fijo mirando hacia abajo. La luz es la misma lámpara todo el rato, sin luz de ventana ni flash. El enfoque y la exposición van bloqueados, y las fotos salen en JPG.
- **La escala.** El ancho de la foto abarca 5 golillas. Cada foto se reduce a 480 px de lado corto (480 × 640, como las 21 originales; `golillas.sesion.reducir`), y así la golilla queda con un radio de unos 50 px. Antes de empezar, 3 fotos de ensayo confirman el montaje; esas golillas después no se usan.
- **La lista al azar.** `scripts/sesion_v5.py lista` arma el orden de las 180 tomas con semilla 0 (`np.random.default_rng(0)`), con los grupos y las malas intercalados. Así, si la luz cambia durante la tarde, el cambio cae parejo en ajustar, calibrar y probar, y no se confunde con un defecto. La lista queda en `data/propio_v5/lista_tomas.csv`, en el mismo commit que este ADR.
- **Las etiquetas se deciden antes de fotografiar.** Las malas se hacen a mano antes de la sesión, y las buenas se revisan a ojo. Las medidas mínimas: el corte va de lado a lado del anillo; el doblez levanta un lado 1 mm o más; la muesca tiene 1 mm o más de hondo; el rayón se ve, mide 3 mm o más y va hacia arriba. La otra medida es 5/16" (3/16" si no hay), y se anota en el sello.
- **Repetir una toma** solo si la foto salió movida, cortada, o con la mano o una sombra. Nunca por cómo se ve la golilla.

## Sellado y orden de lectura
1. **Antes de la sesión** va este commit, con el ADR, la lista y el código del sello.
2. **La sesión.** Las fotos originales quedan fuera de git, en `data/propio_v5/originales/`, en norea y con una copia en Drive.
3. **El sello.** `scripts/sesion_v5.py sellar` ordena las fotos por la hora de la cámara, las calza con la lista y escribe un CSV por grupo, `data/propio_v5/sello_{ajusta,calibra,prueba,separadores}.csv`, con la huella SHA-256 de cada foto. Se detiene si falta o sobra una foto, si hay fotos de más de un día o si hay tamaños distintos. La hoja de control muestra solo los 18 papelitos, que tienen que leerse B01 a B18 en orden. Los CSV van en un commit propio. Hasta ahí, ningún programa ha mirado los píxeles de una golilla.
4. **e13 queda en código antes de leer las fotos**, con una prueba de humo que no usa fotos de la v5. El control es el siguiente: en las 21 fotos viejas, con las reglas de e12, el candidato tiene que repetir el 6/6 y el 1/15 del análisis post hoc.
5. **La corrida**, una sola vez y en este orden:
   1. **Revisión del montaje, solo con las 20 que ajustan:** la cadena clásica tiene que medir 18 o más, y la mediana del radio equivalente tiene que quedar entre 40 y 60 px. Si falla, la sesión se repite entera, y las fotos que calibran y que prueban no se leen nunca.
   2. Ajustar con las 20 y calibrar con las 100.
   3. Medir las 60 de prueba y guardar la decisión.

   e13 solo lee fotos cuya huella calza con el sello (`golillas.sesion.cargar_grupo`).

## Qué se publica
El repo es público. Las fotos originales traen la hora y pueden traer la ubicación en sus metadatos (EXIF), así que no se suben. Después de e13 se puede subir la versión de 480 × 640 sin EXIF, como las 21 originales; eso se decide entonces. El sello solo guarda nombres de archivo, huellas y la hora de cada toma.

## Riesgos
- **Las de otra medida pueden pasar todas.** clasica_4 no mide tamaño, y PatchCore compara parches chicos, que se parecen aunque la golilla sea un 20 % más grande (**no verificado**). Por eso no deciden.
- **Las malas están hechas a mano.** No son defectos de fábrica, y pueden ser más fáciles o más difíciles de ver que los reales. Se informa así.
- **Un solo día.** La garantía del umbral vale para ese montaje y esa luz. Otro día es otro lote, como el set de prueba de Edge Impulse en e12: para usar el modelo en otra sesión, hay que recalibrar con buenas de esa sesión.
- **Un error al calzar fotos y tomas** cambiaría etiquetas. Para evitarlo están los papelitos separadores, el conteo de cada fila durante la sesión y la hoja de control.
- **El requisito 1 es exigente:** un detector del 95 % lo pasa solo 4 de cada 10 veces. Se fija así porque el N0 pide todas las malas.

## Lo que no se hace
- Cambiar los criterios de los ADR 0014 y 0015, o el de este ADR después de tomar las fotos.
- Elegir algo con el set de prueba de Edge Impulse o con las 60 fotos de prueba.
- Mezclar las 21 fotos viejas con las nuevas, o fotos de dos días.
- Subir al repo las fotos originales o imágenes de datasets públicos.
- Validar contra MVTec `metal_nut`: según D2, va después de las fotos.

Volver al [árbol](../arbol.md).
