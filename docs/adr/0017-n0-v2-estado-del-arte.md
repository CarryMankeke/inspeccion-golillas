# ADR 0017 · N0 v2 y N1 v5: estado del arte con datasets públicos de varios tipos de golilla

**Estado:** propuesta. Falta el OK de Camilo al criterio y a las descargas. Ningún set de prueba se ha bajado ni mirado · **Fecha:** 2026-11-28 · **Nivel:** N0 y N1 · **Script:** `experimentos/e13_estado_del_arte.py`, por escribir · **Catastro:** [estado-del-arte.md](../estado-del-arte.md)

## Contexto
En las versiones 1 a 4, el enfoque (N1) se decidió con las 21 fotos propias. En otros montajes la clásica no se traslada. PatchCore ordena bien, y con el umbral conformal detecta 81–87 de 100 malas en Kaggle, pero el requisito del montaje propio se mide con 15 buenas y bloquea todo ([ADR 0015](0015-n1-v4-opciones.md)).

El 27 nov Camilo rechazó la sesión de fotos propias ([ADR 0016](0016-n1-v5-fotos-propias.md)). La v5 va solo con datasets públicos con golillas buenas y malas, y si las golillas difieren, el proyecto se adapta a varios tipos. Además, el repo pasa a mostrar el estado del arte: visión clásica contra visión con IA, con todo lo que exista y las licencias permitan.

Eso cambia la pregunta. Antes era: ¿qué enfoque inspecciona mejor esta golilla en este montaje? Ahora es: **¿qué métodos, de la visión clásica a la IA de 2025, logran inspeccionar un tipo nuevo de golilla calibrándose solo con sus buenas?**

## Qué cambia en el N0 (N0 v2)
| | N0 v1 ([ADR 0001](0001-n0-objetivo.md)) | N0 v2 (este ADR) |
|---|---|---|
| Objetivo | Retirar de la cinta las golillas malas de un montaje | Un inspector que **se calibra por tipo de golilla**: con buenas de un tipo, fotografiadas en su montaje, aprende ese tipo y retira las malas sin haber visto ninguna |
| Datos que deciden | 21 fotos propias | Datasets públicos de varios tipos (tabla de abajo). Las 21 propias solo sirven para desarrollar |
| Qué es una mala | 4 defectos de forma | Lo que el dataset marca como mala, de forma o de superficie. Se informa por tipo de defecto |
| Detección | 6/6 | ≥ 95 % de las malas de prueba de cada tipo |
| Falsas alarmas | ≤ 1/15 | ≤ 5 % de las buenas de prueba de cada tipo |
| Umbral | Hotelling | Conformal partido con α = 3 %, calibrado solo con buenas de entrenamiento |
| Latencia | < 100 ms en CPU | < 100 ms por pieza en la CPU del servidor (Intel i3-9100, 4 núcleos), contando todas sus vistas |
| Licencias | — | Código, pesos y datos auxiliares con licencia verificada en la fuente que deje publicar resultados en este repo |

Se mantienen estos límites: una golilla por foto (o por juego de vistas), y la línea procesa un tipo a la vez y sabe cuál es. El ajuste corre en la GTX 1060 (6 GB) con 8 GB de RAM, y la línea sigue simulada ([ADR 0012](0012-p7-linea.md)).

**Por qué estos números.** El ≤ 5 % de falsas es el requisito 2 de los ADR 0014 y 0015. El 95 % es el de la prueba R del ADR 0014. El 6/6 del N0 v1 no se puede pedir con 72 a 74 malas por tipo: un detector que de verdad detecta el 99 % las vería todas solo 48 de cada 100 veces (0,99⁷⁴).

Si este ADR se acepta, el ADR 0001 lleva una línea que remite aquí. El ADR 0004 se cierra con la decisión de e13.

## Datasets (fijados)
| Dataset | Golilla | Buenas de entrenamiento | Prueba | Licencia | Papel |
|---|---|---|---|---|---|
| MANTA Tiny `red_washer` | Roja. 5 vistas por pieza en una imagen de 256 × 1.280 px (según la fuente) | 578 | 148 buenas y 74 malas | CC BY-NC-SA 4.0 (ver abajo) | **Decide** |
| MANTA Tiny `yellow_green_washer` | Amarillo-verde, mismo montaje | 584 | 144 buenas y 72 malas | CC BY-NC-SA 4.0 | **Decide** |
| Roboflow saksit «metal washer» | 313 fotos (train 194, valid 60, test 59) | Por contar | Por contar | Dominio público | **Decide** si pasa la regla de tamaño; si no, informa |
| Roboflow renuka-p | Golillas planas de bronce, 151 fotos (60, 45 y 46) | Por contar | Por contar | CC BY 4.0 | Igual |
| Roboflow «Washer Classifier» | `washer_ok` y `washer_ng`, 443 fotos | Por contar | Por contar | CC BY 4.0 | Igual |
| Edge Impulse | Planas, dos fondos | 273 | 88 buenas y 103 malas, ya vistas en e10 | BSD 3-Clause Clear | Desarrolla |
| Kaggle «Simplified washers» | Dibujos | 1.000 | 100 y 100 | CC BY-SA 4.0 | Desarrolla |
| 21 fotos propias | Planas, oscuras sobre papel | 15 | 6 malas | MIT | Desarrolla y sirve de control |
| Sintéticas ([ADR 0005](0005-datos-sinteticos.md)) | Generadas | 300 | 300 y 300 | MIT | Desarrolla |

Los conteos de MANTA vienen de `distribution.md`, en el mismo depósito de figshare. Los de Roboflow vienen de sus fichas.

**Licencia de MANTA.** figshare marca CC BY 4.0, y la [página del proyecto](https://grainnet.github.io/MANTA) dice CC BY-NC-SA 4.0, sin uso comercial. Los autores no publican un LICENSE que lo resuelva, así que se respeta la más restrictiva. Las imágenes no se suben al repo, y una figura que muestre alguna queda bajo CC BY-NC-SA 4.0.

**Por qué MANTA Tiny y no la completa.** La Tiny trae las dos golillas en un solo archivo de 3,2 GB (`mechanics.tar.gz`, con otras 9 piezas que se borran), con la partición de entrenamiento y prueba y sus conteos publicados. La completa pesa 4,6 GB, tiene más resolución y no publica conteos por pieza.

**Regla de tamaño para Roboflow (fijada antes de contar).** Un set de Roboflow decide solo si, contando las etiquetas y sin abrir ninguna imagen, sus particiones train y valid traen 66 piezas buenas o más, y su partición test trae 20 buenas o más y 20 malas o más. Con 66 buenas, 33 ajustan y 33 calibran, el mínimo para α = 3 %. Si no pasa la regla, el set solo informa su AUC.

**Qué es una pieza en Roboflow.** Cada caja anotada es una pieza, y se recorta con un margen del 10 %. Las copias aumentadas de una misma foto (el mismo nombre de origen antes de `.rf.`) van siempre al mismo grupo. Si existe una versión sin aumentos, se exporta esa.

**Lo que ya se vio.** El 27 nov se vieron unas 6 miniaturas de cada set de Roboflow (renuka-p, saksit y project-1) en sus páginas. No se ajustó nada con ellas. De MANTA solo se leyeron la ficha, la tabla de conteos y los metadatos de figshare.

Los descartados y sus razones están en el [catastro](../estado-del-arte.md#1-datasets-con-golillas-buenas-y-malas).

## Métodos (lista fija, con su configuración por defecto y sin búsqueda)
| # | Método | Familia | Qué aprende de las buenas | Implementación | Licencia del código · de los pesos |
|---|---|---|---|---|---|
| 1 | clasica_4 | Clásica | Media y covarianza de 4 rasgos de forma ([ADR 0010](0010-p5-rasgos.md)) | Propia, P1–P6 | MIT · sin pesos |
| 2 | clasica_v3_6 | Clásica | Lo mismo, más la razón de radios y el radio exterior: mide tamaño | Propia | MIT · sin pesos |
| 3 | PatchCore propio | IA sin entrenamiento | Banco de parches de ResNet-18 (layer2+3, coreset del 1 %), el de e09 a e12 | Propia | MIT · BSD-3 (torchvision) |
| 4 | PatchCore | IA sin entrenamiento | Banco de parches de WideResNet-50 (layer2+3, coreset del 10 %) | anomalib | Apache-2.0 · BSD-3 |
| 5 | PaDiM | IA sin entrenamiento | Una gaussiana por posición, sobre ResNet-18 | anomalib | Apache-2.0 · BSD-3 |
| 6 | AnomalyDINO | IA de pocas muestras | Vecinos más cercanos sobre parches de DINOv2 ViT-S/14 | anomalib | Apache-2.0 · Apache-2.0 |
| 7 | Dinomaly | IA con entrenamiento | Reconstruye los rasgos de DINOv2 ViT-B/14 con registros | anomalib | Apache-2.0 · Apache-2.0 |
| 8 | EfficientAD-S | IA con entrenamiento | Un alumno imita a un maestro, más un autoencoder | anomalib | Apache-2.0 · **no verificado** (maestro e Imagenette) |
| 9 | WinCLIP | IA de cero o pocas muestras, con texto | CLIP compara ventanas de la foto con frases de «buena» y «dañada» | anomalib | Apache-2.0 · **no verificado** (OpenCLIP LAION-400M) |

Los métodos 8 y 9 entran solo si la licencia de sus pesos y datos auxiliares queda verificada en la fuente antes del commit que sella los datos. Si no, quedan fuera y se informa así.

**Por qué estos.** Hay uno por cada familia del estado del arte: banco de parches (PatchCore), estadística por posición (PaDiM), vecinos sobre un modelo base (AnomalyDINO, el mejor con pocas buenas), reconstrucción con un modelo base (Dinomaly, el mejor en MVTec AD en 2025), alumno-maestro (EfficientAD, el más rápido) y visión con texto sin ejemplos (WinCLIP). Además, comparar el PatchCore propio con el de anomalib valida la implementación propia.

**Versiones fijas.** Se usa anomalib 2.7.x, con la versión exacta fijada en el commit del código, y torch 2.14 con CUDA 12.6, la última que trae kernels para la GTX 1060. Se instala en un entorno aparte del actual.

**Quedan fuera, con su razón:**
- **Supervisados** (la ResNet-18 propia, YOLO y RF-DETR): el N0 v2 calibra solo con buenas, y MANTA no trae malas de entrenamiento. Además, YOLO es AGPL-3.0.
- **MuSc y AdaCLIP:** no caben en los 100 ms. MuSc tarda 270 ms por imagen en una RTX 3090, según su repo, y AdaCLIP unos 50 s en CPU, según una demo.
- **UniVAD:** código con licencia CC BY-NC-SA, y su ViT-H no cabe en 6 GB.
- **DINOv3 y SAM 3:** tienen licencia propia de Meta.
- **Servicios comerciales:** LandingLens prohíbe usarse para comparar productos (cláusula 5(b)(viii) de sus términos). Edge Impulse (FOMO-AD, de una clase) y HALCON (edición de estudiante) piden cuenta o licencia, y sus términos sobre publicar resultados no se leyeron. Quedan como opción para una v6, si Camilo crea la cuenta y los términos lo permiten. AWS Lookout for Vision cerró, Azure Custom Vision se retira en 2028 e Intel Geti usa anomalib por dentro.
- **MATLAB** (Automated Visual Inspection): repite PatchCore, FastFlow y EfficientAD, y no está instalado en el servidor.

## Protocolo
Por cada tipo de golilla:

1. **Perfil del tipo**, mirando solo las buenas que ajustan. Fija qué vista es la de arriba (en MANTA), la polaridad y la escala que lleva el radio exterior mediano a 50 px, la del N0 v1, para la clásica. Queda en `golillas/tipos/<tipo>.json`, en un commit antes de la corrida.
2. **Revisión del montaje**, también solo con las que ajustan. Si la cadena clásica no logra medir el 90 % o más de ellas, la clásica «no aplica» a ese tipo: cuenta como que no cumple, y se informa. Es el caso de las golillas que no se ven como un anillo completo desde arriba, como las con orejas.
3. **Partición.** Las buenas de entrenamiento se barajan con `np.random.default_rng(0)` y se parten en dos mitades: la primera ajusta y la segunda calibra. En MANTA quedan 289 y 289 en la roja, y 292 y 292 en la amarillo-verde. La prueba es la partición de prueba del dataset.
4. **Ajuste.** Cada método se ajusta con las que ajustan, con su configuración por defecto. Tres reglas por si no alcanzan los recursos:
   - Si un método con entrenamiento no cabe en memoria, su lote baja a la mitad hasta que quepa.
   - Si un método sin entrenamiento no cabe, se ajusta con una submuestra de piezas al azar (semilla 0), que baja a la mitad hasta que quepa. Es lo esperable en PatchCore con WideResNet-50: con 1.445 vistas, su banco antes del coreset ocuparía unos 7 GB (1.445 × 784 parches × 1.536 números × 4 bytes; estimación propia).
   - Si el ajuste pasa de 2 horas por tipo en la GTX 1060, se corta a las 2 horas.

   Cada caso se informa en la tabla.
5. **Puntaje por pieza.** En MANTA, los métodos de IA miran las 5 vistas por separado: las vistas de las piezas que ajustan entran todas juntas al ajuste, y el puntaje de una pieza es el máximo de sus 5 vistas. La clásica mide solo la vista de arriba. Si la clásica no logra medir una pieza, la retira siempre, y esa pieza no calibra, como en e11 y e12.
6. **Umbral:** conformal partido con α = 3 %, sobre los puntajes por pieza de las que calibran (`golillas.decision.umbral_conformal`). Con 289 que calibran, la tasa real de falsas de una calibración sigue una Beta(8, 282): su media es 2,8 % y pasa del 5 % con probabilidad 2 % ([Bates et al., Prop. 1](https://arxiv.org/abs/2104.08279)).
7. **Prueba:** se corre una sola vez.

## Criterio (fijado antes de bajar los sets de prueba)
Deciden los dos tipos de MANTA y los sets de Roboflow que pasen la regla de tamaño.

1. **Detección y falsas, en cada tipo que decide:** retira el 95 % o más de las malas de prueba y el 5 % o menos de las buenas de prueba. En MANTA eso es 71 o más de 74 y 7 o menos de 148 en la roja, y 69 o más de 72 y 7 o menos de 144 en la amarillo-verde.
2. **Latencia:** menos de 100 ms por pieza en la CPU del servidor, con sus 5 vistas en MANTA. Es la mediana de 50 piezas que ajustan, con los 4 núcleos.
3. **Licencia:** ya está filtrada en la lista de métodos.

**Elección.** Entre los que cumplen 1 y 2, gana la mayor detección media de los tipos que deciden. Si hay empate (2 puntos o menos), gana el más fácil de explicar: primero la clásica, luego los sin entrenamiento, luego los con entrenamiento y al final los de visión con texto. Si siguen empatados, gana el más rápido.

**Si uno gana,** `decisiones.json` guarda `enfoque_v5` con `adoptado = true`. El [ADR 0004](0004-n1-enfoque.md) sale de revisión, y ese método pasa a ser el del inspector, con un perfil por tipo.

**Si ninguno cumple,** se informa «ninguno cumple» y no se adopta nada. La tabla igual se publica, ordenada por detección media entre los que cumplen las falsas en cada tipo, y el ADR 0004 sigue en revisión. Los requisitos no se relajan.

**Qué tan probable es pasar** (binomial, en cada tipo de MANTA):
- Detección: con una tasa real del 98 %, el requisito se cumple con probabilidad 94 %. Con 97 % baja a 82 %, con 95 % a 50 % y con 90 % a 5 %.
- Falsas: con una tasa real del 3 %, que es lo que apunta el umbral, se cumple con probabilidad 92 %. Con 5 % baja a 55 %, y con 7 % a 19 %.
- Un método que detecta el 98 % y retira el 3 % pasa los dos tipos de MANTA con probabilidad 0,94² × 0,92² ≈ 0,75.

**Cómo se informa (no decide):**
- El IC95 exacto (Clopper-Pearson) de cada tasa, y el AUC por pieza con su IC95 por remuestreo (2.000, semilla 0).
- La detección por tipo de defecto, y el AUC de cada vista en MANTA.
- La memoria máxima de la GPU y el tiempo de ajuste de cada método.
- Una curva de pocas muestras, después de guardar la decisión: el AUC con 1, 4, 16 y 64 piezas que ajustan (5 semillas), solo para los métodos sin entrenamiento y para WinCLIP sin ninguna, si entra. Responde cuántas buenas hacen falta para calibrar un tipo nuevo.
- Los mismos métodos en los sets de desarrollo: Kaggle, Edge Impulse (no ciego), las propias y las sintéticas.

## Orden y sellado
1. **Este ADR, aceptado por Camilo, va en un commit** junto con el código de e13 y una prueba de humo que usa solo los sets de desarrollo. El control: con la configuración de e12, el PatchCore propio tiene que repetir en Kaggle sus números de e12.
2. **Descarga** al servidor: MANTA Tiny (`mechanics.tar.gz`, 3.197.059.395 bytes, con el MD5 que publica figshare) y los sets de Roboflow que Camilo exporte. Las huellas quedan fijadas en `scripts/descargar_datos.py`.
3. **Sello.** Una lista por tipo con la huella SHA-256 de cada archivo que ajusta, calibra y prueba, armada con las rutas y los bytes, sin decodificar imágenes. Va en un commit propio. Se reusa la idea de `golillas/sesion.py`, que quedó sin usar en el ADR 0016.
4. **Perfil de cada tipo**, con las que ajustan (commit).
5. **Corrida única de e13** en el servidor, en segundo plano y mirando `nvidia-smi` antes.
6. Después de guardar la decisión, lo que solo informa.

Hasta el paso 5, ningún programa decodifica una imagen de prueba.

## Qué cambia en el código
| Dónde | Cambio |
|---|---|
| `golillas/tipos/` (nuevo) | Un perfil JSON por tipo: dataset, vistas, vista de arriba, polaridad, escala y si aplica la clásica |
| `golillas/conjuntos.py` (nuevo) | Cargadores que entregan piezas (id, vistas, mala, defecto, grupo) de MANTA (corta las 5 vistas), Roboflow (recorta las cajas COCO), Edge Impulse, Kaggle y las propias |
| `golillas/metodos/` (nuevo) | Una interfaz común, `ajustar(piezas)` y `puntajes(piezas)`, con adaptadores para la clásica (normaliza la escala), el PatchCore propio (sale de e09) y los modelos de anomalib |
| `golillas/decision.py` | Suma el IC exacto, que hoy vive en e12 |
| `golillas/sello.py` (nuevo) | Sella y verifica las particiones de los datasets, con la idea de `sesion.py` |
| `experimentos/e13_estado_del_arte.py` (nuevo) | El protocolo de este ADR, con `--humo` |
| `scripts/descargar_datos.py` | La entrada de MANTA Tiny (ya agregada) y las de Roboflow cuando se exporten |
| `pyproject.toml` | Un extra `sota` con anomalib y torch 2.14 fijados |
| `tests/` | Perfiles, cargadores (con imágenes falsas de 256 × 1.280), sello e interfaz de métodos |
| `data/ATRIBUCIONES.md` | MANTA y Roboflow, cuando se bajen |

**El README** cambia después de e13. La portada pasa a ser la tabla «método × tipo de golilla», con la clásica contra la IA, y la historia de las versiones 1 a 4 queda como el camino que llevó hasta ahí. Mientras tanto, solo cambian los próximos pasos.

## Riesgos
- **Las golillas de MANTA pueden no ser metálicas ni planas:** «roja» y «amarillo-verde» hacen pensar en plástico (**no verificado**). Se aceptan igual como tipos de golilla, y se informa qué son.
- **Los defectos de MANTA se hicieron a mano,** según su paper. Pueden ser más fáciles o más difíciles que los de fábrica.
- **Si ningún set de Roboflow pasa la regla, deciden solo los dos tipos de MANTA,** que comparten montaje. Los otros montajes (Edge Impulse, Kaggle) solo informan.
- **Cinco vistas en 100 ms son 20 ms por vista en CPU.** Los modelos grandes, como Dinomaly con ViT-B, probablemente no cumplan la latencia (**no verificado**). Ese resultado también es parte del estado del arte.
- **La licencia de MANTA no está resuelta.** Mientras los autores no aclaren, se trata como no comercial.

## Lo que no se hace
- Bajar o mirar los sets de prueba antes de que este ADR esté aceptado y en un commit, o elegir algo con ellos.
- Cambiar los criterios de los ADR 0014, 0015 y 0016.
- Subir al repo imágenes de los datasets.
- Usar métodos, pesos o datos con licencia sin verificar, o servicios cuyos términos prohíben publicar comparaciones.

Volver al [árbol](../arbol.md).
