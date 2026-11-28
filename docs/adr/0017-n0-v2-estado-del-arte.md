# ADR 0017 · N0 v2 y N1 v5: estado del arte con datasets públicos de varios tipos de golilla

**Estado:** aceptada el 28 nov 2026, con el OK de Camilo en el chat y con los requisitos tomados de normas y referencias citables. Ningún set de prueba se ha bajado ni mirado antes de este commit · **Fecha:** 2026-11-28 · **Nivel:** N0 y N1 · **Script:** `experimentos/e13_estado_del_arte.py`, por escribir · **Catastro:** [estado-del-arte.md](../estado-del-arte.md)

> **Cambio del 28 nov 2026, antes de bajar ningún dato:** la propuesta del commit 923c275 pedía ≥ 95 % de detección, con números propios. Camilo pidió que cada requisito saliera de una norma o referencia citable. Ahora la detección, las falsas alarmas y la latencia salen de la tabla «De dónde salen los números». EfficientAD queda fuera (sus pesos no declaran licencia) y WinCLIP entra (pesos MIT).

## Contexto
En las versiones 1 a 4, el enfoque (N1) se decidió con las 21 fotos propias. En otros montajes la clásica no se traslada. PatchCore ordena bien, y con el umbral conformal detecta 81–87 de 100 malas en Kaggle, pero el requisito del montaje propio se mide con 15 buenas y bloquea todo ([ADR 0015](0015-n1-v4-opciones.md)).

El 27 nov Camilo rechazó la sesión de fotos propias ([ADR 0016](0016-n1-v5-fotos-propias.md)). La v5 va solo con datasets públicos con golillas buenas y malas, y si las golillas difieren, el proyecto se adapta a varios tipos. Además, el repo pasa a mostrar el estado del arte: visión clásica contra visión con IA, con todo lo que exista y las licencias permitan. El repo es un portafolio para trabajar en la industria, ojalá en minería, así que sus requisitos se toman de normas y referencias de la industria en vez de inventarse.

Eso cambia la pregunta. Antes era: ¿qué enfoque inspecciona mejor esta golilla en este montaje? Ahora es: **¿qué métodos, de la visión clásica a la IA de 2025, logran inspeccionar un tipo nuevo de golilla calibrándose solo con sus buenas, y con qué desempeño frente a los criterios que usa la industria?**

## De dónde salen los números
| Qué | Requisito | Fuente | Tipo de fuente |
|---|---|---|---|
| Cómo se acepta un sistema de visión que clasifica | Con muestras de referencia etiquetadas, contando la falsa aceptación (mala que pasa) y el falso rechazo (buena retirada), con un plan de muestreo y una prueba estadística | VDI/VDE/VDMA 2632-3:2017 y 2632-3.1:2020 ([VDI](https://vdi.de/en/home/vdi-standards/details/vdivdevdma-2632-blatt-3-machine-visionindustrial-image-processing-acceptance-test-of-classifying-machine-vision-systems), [índice en DIN](https://www.din.de/de/wdc-beuth:din21:325735543/toc-3177660/download)) | Primaria, pero solo la ficha y el índice: la norma es pagada y sus números no se leyeron |
| Detección | Tasa de no detección **≤ 2 %** en cada tipo, es decir, detectar el 98 % o más de las malas | Criterio «aceptable» del análisis de sistemas de medición por atributos, atribuido al AIAG MSA 4.ª ed. ([TechieQuality, 2024](https://www.techiequality.com/2024/03/28/attribute-msa-how-to-do-attribute-msa-example-acceptance-criteria/), y coincide en otras guías) | Secundaria: el manual es pagado y las guías no citan su página |
| Falsas alarmas | Tasa de falsa alarma **≤ 5 %** en cada tipo | El mismo criterio «aceptable» | Secundaria |
| Efectividad | Decisiones correctas **≥ 90 %** | El mismo criterio | Secundaria |
| Latencia | **≤ 100 ms por pieza**, con todas sus vistas | El ritmo de una clasificadora óptica de disco de vidrio para golillas y tuercas: hasta 600 piezas por minuto ([Fastener+Fixing, 2021](https://fastenerandfixing.com/manufacturing-technology/new-optical-sorting-machines-drive-efficiency)), o sea, 100 ms por pieza. Calza con la guía de Keyence, que da de 20 a 100 ms para una inspección simple ([Keyence](https://keyence.com/ss/products/vision/visionbasics/basic/practice)) | Ficha de fabricante: ninguna norma fija la latencia |
| Qué se informa | Matriz de confusión, sensibilidad, especificidad, precisión, F1, intervalos de confianza, latencia y rendimiento | ISO/IEC TS 4213:2022, sobre la evaluación del desempeño de clasificadores de aprendizaje automático ([ISO](https://www.iso.org/standard/79799.html), índice) | Primaria, solo el índice |
| Contexto: aceptar lotes | Los lotes de fijaciones, golillas incluidas, se aceptan por muestreo con número de aceptación cero | ISO 3269:2019 (fijaciones) e ISO 2859-1, que en Chile es la [NCh44:2007](https://ecommerce.inn.cl/nch44200744722) | Primaria, solo la ficha |

**Qué significa «aceptable» y «marginal».** Según el mismo criterio, un sistema es **aceptable** con no detección ≤ 2 % y falsas ≤ 5 %. Es **marginal** con no detección ≤ 5 % y falsas ≤ 10 %, y en ese caso sirve solo si el cliente lo aprueba. Este proyecto adopta solo lo aceptable, y lo marginal se informa.

**Por qué no se prueba con confianza estadística.** La VDI 2632-3.1 decide con una prueba de hipótesis. Para demostrar una no detección ≤ 2 % con 95 % de confianza, sin ninguna falla, hacen falta 149 malas o más (1 − 0,05^(1/n) ≤ 0,02). MANTA trae 72 y 74 por tipo. Por eso se decide con el criterio puntual del AIAG, y cada tasa se informa con su intervalo exacto. Por ejemplo, 1 falla en 74 deja la no detección por debajo de 6,3 % con 95 % de confianza.

**La norma del producto.** Las golillas planas métricas tienen dimensiones y tolerancias en ISO 7089 e ISO 4759-3. Por ejemplo, una M8 tiene 8,40 mm de agujero (máx. 8,62), 16 mm de diámetro exterior (mín. 15,57) y 1,6 mm de espesor (de 1,4 a 1,8), según la tabla de [Westfield Fasteners](https://westfieldfasteners.co.uk/Standards/Washer_ISO7089_M.html), una fuente secundaria. Esta versión no mide en milímetros, porque MANTA no trae la calibración de su cámara. Las tolerancias sirven después: para el paso de medir el diámetro en mm y para detectar golillas encimadas ([ADR 0018](0018-p0-alimentacion-encimadas.md)).

## Qué cambia en el N0 (N0 v2)
| | N0 v1 ([ADR 0001](0001-n0-objetivo.md)) | N0 v2 (este ADR) |
|---|---|---|
| Objetivo | Retirar de la cinta las golillas malas de un montaje | Un inspector que **se calibra por tipo de golilla**: con buenas de un tipo, fotografiadas en su montaje, aprende ese tipo y retira las malas sin haber visto ninguna |
| Datos que deciden | 21 fotos propias | Datasets públicos de varios tipos (tabla de abajo). Las 21 propias solo sirven para desarrollar |
| Qué es una mala | 4 defectos de forma | Lo que el dataset marca como mala, de forma o de superficie. Se informa por tipo de defecto |
| Detección | 6/6 | No detección ≤ 2 % en cada tipo (AIAG, aceptable) |
| Falsas alarmas | ≤ 1/15 | ≤ 5 % en cada tipo (AIAG, aceptable) |
| Umbral | Hotelling | Conformal partido con α = 3 %, calibrado solo con buenas de entrenamiento |
| Latencia | < 100 ms en CPU | ≤ 100 ms por pieza en la CPU del servidor (Intel i3-9100, 4 núcleos), contando todas sus vistas: 600 piezas por minuto |
| Licencias | — | Código, pesos y datos auxiliares con licencia leída en la fuente que deje publicar resultados en este repo |

Se mantienen estos límites: una golilla por foto (o por juego de vistas), y la línea procesa un tipo a la vez y sabe cuál es. Las golillas encimadas van en el [ADR 0018](0018-p0-alimentacion-encimadas.md). El ajuste corre en la GTX 1060 (6 GB) con 8 GB de RAM, y la línea sigue simulada ([ADR 0012](0012-p7-linea.md)). El computador de la línea no tiene por qué tener GPU, así que la latencia se exige en CPU, y la de GPU se informa.

Este ADR remite al ADR 0001 y cierra el ADR 0004 con la decisión de e13.

## Datasets (fijados)
| Dataset | Golilla | Buenas de entrenamiento | Prueba | Licencia | Papel |
|---|---|---|---|---|---|
| MANTA Tiny `red_washer` | Anillo plano moldeado, no metálico, delgado y de agujero grande; cabe en 3 × 11 × 11 mm | 578 | 148 buenas y 74 malas | CC BY-NC-SA 4.0 (ver abajo) | **Decide** |
| MANTA Tiny `yellow_green_washer` | Anillo plano moldeado, no metálico, grueso y de agujero chico; cabe en 3 × 8 × 8 mm | 584 | 144 buenas y 72 malas | CC BY-NC-SA 4.0 | **Decide** |
| Roboflow saksit «metal washer» (v4) | Golillas de metal según su nombre, 313 fotos (train 194, valid 60, test 59) | 181 | 41 buenas y 17 malas | Dominio público | Informa: no pasa la regla de tamaño |
| Roboflow renuka-p (v3) | Golillas planas de bronce, 151 fotos (60, 45 y 46) | 9 | 3 buenas y 43 malas | CC BY 4.0 | Informa: no pasa la regla |
| Roboflow «Washer Classifier» (v3) | `washer_ok` y `washer_ng`, 443 fotos (377, 54 y 12) | 271 | 9 buenas y 3 malas | CC BY 4.0 | Informa: no pasa la regla |
| Edge Impulse | Planas, dos fondos | 273 | 88 buenas y 103 malas, ya vistas en e10 | BSD 3-Clause Clear | Desarrolla |
| Kaggle «Simplified washers» | Dibujos | 1.000 | 100 y 100 | CC BY-SA 4.0 | Desarrolla |
| 21 fotos propias | Planas, metálicas, oscuras sobre papel | 15 | 6 malas | MIT | Desarrolla y sirve de control |
| Sintéticas ([ADR 0005](0005-datos-sinteticos.md)) | Generadas | 300 | 300 y 300 | MIT | Desarrolla |

**Qué son las golillas de MANTA.** Que son anillos planos se ve en la figura II del [suplementario del paper](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Fan_MANTA_A_Large-Scale_CVPR_2025_supplemental.pdf) y en la [figura de ejemplos](https://grainnet.github.io/static/images/manta_examples2.png) de la página del proyecto. Sus tamaños (la caja que las contiene) salen de la tabla I del suplementario. Que no son metálicas es una inferencia, porque ninguna fuente nombra el material: en las figuras se ven opacas y mate, la amarillo-verde se deforma como goma, y la descripción que publican los autores de cada defecto (`mechanics_visual_description.json`, [figshare](https://figshare.com/articles/dataset/manta-Declarative_Knowledge/27951513)) explica los defectos de las dos por el moldeo, la resina y los pigmentos. Son golillas de plástico o elastómero, como las de nylon, que también se usan en la industria. El inspector no depende del material, pero se informa.

Los conteos de MANTA Tiny vienen de `distribution.md`, en su depósito de figshare. Los de Roboflow vienen de sus fichas.

**Licencia de MANTA.** figshare marca CC BY 4.0, y la [página del proyecto](https://grainnet.github.io/MANTA) dice CC BY-NC-SA 4.0, sin uso comercial. Los autores no publican un LICENSE que lo resuelva, así que se respeta la más restrictiva. Las imágenes no se suben al repo, y una figura que muestre alguna queda bajo CC BY-NC-SA 4.0.

**Por qué MANTA Tiny y no la completa.** La Tiny trae las dos golillas en un solo archivo de 3,2 GB (`mechanics.tar.gz`, con otras 9 piezas que se borran), con la partición de entrenamiento y prueba y sus conteos publicados. La completa pesa 4,6 GB, tiene más resolución y no publica conteos por pieza en figshare.

**Regla de tamaño para Roboflow (fijada antes de contar).** Un set de Roboflow decide solo si, contando las etiquetas y sin abrir ninguna imagen, sus particiones train y valid traen 66 piezas buenas o más, y su partición test trae 20 buenas o más y 20 malas o más. Con 66 buenas, 33 ajustan y 33 calibran, el mínimo para α = 3 %. Si no pasa la regla, el set solo informa su AUC.

**Qué es una pieza en Roboflow.** Cada caja anotada es una pieza, y se recorta con un margen del 10 %. Las copias aumentadas de una misma foto (el mismo nombre de origen antes de `.rf.`) van siempre al mismo grupo. Si existe una versión sin aumentos, se exporta esa.

> **Cambio del 28 nov 2026, al contar las etiquetas y sin abrir ninguna imagen:** la pieza pasa a ser **la foto entera**. En los JSON, cada foto trae una sola golilla (en saksit, 193 de 194 fotos de train tienen una caja), y las cajas de algunos defectos marcan solo el defecto: en saksit, `bent` y `color` cubren una mediana del 6 % de la foto, contra el 76 % de `good`; en renuka-p, `DENT` cubre el 0,3 %. Recortar esas cajas habría juzgado un trozo de golilla. Una foto es mala si tiene alguna caja de defecto, buena si todas sus cajas son de la clase buena, y queda fuera si no tiene cajas (1 en el test de saksit). Las malas de train y valid no se usan. Hubo varias versiones sin aumentos, y se exportó la más reciente. El cambio no toca la decisión, porque ningún set de Roboflow pasa la regla de tamaño (abajo).

**Resultado de la regla de tamaño (28 nov 2026, contando etiquetas, sin abrir imágenes):**

| Set (versión) | Buenas en train + valid (≥ 66) | Buenas de prueba (≥ 20) | Malas de prueba (≥ 20) | Papel |
|---|---|---|---|---|
| saksit «metal washer» (v4) | 181 | 41 | **17** | Informa |
| renuka-p «Washer Defect Detection» (v3) | **9** | **3** | 43 | Informa |
| «Washer Classifier» (v3) | 271 | **9** | **3** | Informa |

**Ningún set de Roboflow decide.** Deciden solo los dos tipos de MANTA, como preveía el primer riesgo de la lista. Los tres quedan sellados (`data/sellos/roboflow_*`) e informan su AUC. Los nombres de origen se repiten entre particiones (por ejemplo, `000_png` en train, valid y test de saksit), pero como ninguna versión tiene aumentos, son fotos distintas con el mismo nombre de archivo; la partición los junta igual, por prudencia.

**Lo que ya se vio antes de bajar nada:**
- Roboflow: el 27 nov, unas 6 miniaturas de cada set (renuka-p, saksit y project-1), en sus páginas.
- MANTA: el 28 nov, la figura de ejemplos de su página (10 piezas normales y 10 anómalas de cada golilla), la figura II del suplementario y las descripciones de defectos de los autores, para saber qué golillas son. También la ficha, la tabla de conteos y los metadatos de figshare.

No se ajustó nada con ellas, y el criterio no cambió después de verlas: los requisitos salen de las normas.

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
| 8 | WinCLIP | IA de cero o pocas muestras, con texto | CLIP compara ventanas de la foto con frases de «buena» y «dañada» | anomalib | Apache-2.0 · MIT ([ficha de los pesos](https://huggingface.co/timm/vit_base_patch16_plus_clip_240.laion400m_e31)) |

Las licencias de los pesos se leyeron en sus fichas de Hugging Face y en el LICENSE de cada repo; el detalle está en el [catastro](../estado-del-arte.md#2-métodos-con-ia). Los pesos de WinCLIP son MIT, pero se entrenaron con LAION-400M, cuyas imágenes siguen bajo el copyright de sus dueños: el repo no las usa ni las redistribuye.

**Por qué estos.** Hay uno por cada familia del estado del arte: banco de parches (PatchCore), estadística por posición (PaDiM), vecinos sobre un modelo base (AnomalyDINO, el mejor con pocas buenas), reconstrucción con un modelo base (Dinomaly, el mejor en MVTec AD en 2025) y visión con texto sin ejemplos (WinCLIP). Además, comparar el PatchCore propio con el de anomalib valida la implementación propia.

**Versiones fijas.** anomalib 2.7.1, torch 2.14.1+cu126 y torchvision 0.29.1+cu126, ya instalados en el servidor en un entorno aparte. torch 2.14 con CUDA 12.6 es la última que trae kernels para la GTX 1060.

**Quedan fuera, con su razón:**
- **EfficientAD:** los pesos de su maestro, un zip de 38 MiB en un release de anomalib, no declaran licencia en el release, en anomalib ni en el repo de referencia. Además, su entrenamiento usa Imagenette, cuyas imágenes vienen de ImageNet y heredan su uso solo no comercial.
- **Supervisados** (la ResNet-18 propia, YOLO y RF-DETR): el N0 v2 calibra solo con buenas, y MANTA no trae malas de entrenamiento. Además, YOLO es AGPL-3.0.
- **MuSc y AdaCLIP:** no caben en los 100 ms. MuSc tarda 270 ms por imagen en una RTX 3090, según su repo, y AdaCLIP unos 50 s en CPU, según una demo.
- **UniVAD:** código con licencia CC BY-NC-SA, y su ViT-H no cabe en 6 GB.
- **DINOv3 y SAM 3:** tienen licencia propia de Meta.
- **Servicios comerciales:**
  - LandingLens prohíbe usarse para comparar productos (cláusula 5(b)(viii) de sus términos).
  - Edge Impulse (FOMO-AD, de una clase) y HALCON (edición de estudiante) piden cuenta o licencia, y sus términos sobre publicar resultados no cargaron. Quedan como opción para una v6, si Camilo crea la cuenta y los términos lo permiten.
  - AWS Lookout for Vision cerró el 31 oct 2025, y Azure Custom Vision se retira el 25 sep 2028.
  - Intel Geti usa anomalib por dentro.
- **MATLAB** (Automated Visual Inspection): repite PatchCore, FastFlow y EfficientAD, y no está instalado en el servidor.

## Medición previa de latencia (28 nov 2026, no decide)
Antes de bajar los datos se midió, en la CPU del servidor con 4 hilos, cuánto tarda solo la red base de cada método, con pesos al azar (el tiempo no depende de los pesos). Es una **cota inferior**: el método completo tarda eso más su búsqueda o su decodificador. Script: `scripts/latencia_cota.py`.

| Método | Red base y tamaño de entrada | 1 vista | 5 vistas (una pieza de MANTA) |
|---|---|---|---|
| PatchCore propio | ResNet-18 hasta layer3, 224 px | 13 ms | 65 ms |
| PaDiM | ResNet-18, layer1 a layer3, 256 px | 17 ms | 79 ms |
| AnomalyDINO | DINOv2 ViT-S/14, 252 px | 61 ms | 288 ms |
| PatchCore (anomalib) | WideResNet-50, layer2+3, 256 px | 103 ms | 541 ms |
| WinCLIP | Una pasada de CLIP ViT-B-16-plus, 240 px | 182 ms | 886 ms |
| Dinomaly | Solo el codificador DINOv2-reg ViT-B/14, 392 px | 519 ms | 2.694 ms |

Con 5 vistas, solo las clásicas, el PatchCore propio y PaDiM pueden cumplir los 100 ms en CPU. Los otros se corren igual, porque el estudio muestra cuánto se gana en detección y cuánto cuesta en tiempo. Su latencia en la GTX 1060 se informa, porque con una GPU en la línea el resultado cambia.

## Protocolo
Por cada tipo de golilla:

1. **Perfil del tipo**, mirando solo las buenas que ajustan. Fija qué vista es la de arriba (en MANTA), la polaridad y la escala que lleva el radio exterior mediano a 50 px, la del N0 v1, para la clásica. Queda en `golillas/tipos/<tipo>.json`, en un commit antes de la corrida.
2. **Revisión del montaje**, también solo con las que ajustan. Si la cadena clásica no logra medir el 90 % o más de ellas, la clásica «no aplica» a ese tipo: cuenta como que no cumple, y se informa.
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

1. **Aceptable en cada tipo que decide** (AIAG): no detección ≤ 2 % y falsas ≤ 5 %. En MANTA eso es:
   - en la roja, 73 o más de 74 malas, y 7 o menos de 148 buenas;
   - en la amarillo-verde, 71 o más de 72 malas, y 7 o menos de 144 buenas.

   Con eso, la efectividad queda sobre el 90 % sola.
2. **Latencia:** 100 ms o menos por pieza en la CPU del servidor, con sus 5 vistas en MANTA. Es la mediana de 50 piezas que ajustan, con los 4 núcleos.
3. **Licencia:** ya está filtrada en la lista de métodos.

**Elección.** Entre los que cumplen 1 y 2, gana la mayor detección media de los tipos que deciden. Si hay empate (2 puntos o menos), gana el más fácil de explicar: primero la clásica, luego los sin entrenamiento, luego los con entrenamiento y al final los de visión con texto. Si siguen empatados, gana el más rápido.

**Si uno gana,** `decisiones.json` guarda `enfoque_v5` con `adoptado = true`. El [ADR 0004](0004-n1-enfoque.md) sale de revisión, y ese método pasa a ser el del inspector, con un perfil por tipo.

**Si ninguno cumple,** se informa «ninguno cumple» y no se adopta nada. La tabla igual se publica, con la clase AIAG de cada método en cada tipo (aceptable, marginal o inaceptable), y el ADR 0004 sigue en revisión. Los requisitos no se relajan: un método marginal no se adopta.

**Qué tan probable es pasar** (binomial, en cada tipo de MANTA):
- Detección: con una tasa real del 99,5 %, el requisito se cumple con probabilidad 95 %. Con 99 % baja a 83 %, con 98 % a 56 % y con 97 % a 35 %.
- Falsas: con una tasa real del 3 %, que es lo que apunta el umbral, se cumple con probabilidad 92 %. Con 5 % baja a 55 %, y con 7 % a 19 %.
- Es una prueba exigente, como la norma: con 72–74 malas, un detector del 98 % la pasa solo la mitad de las veces en cada tipo.

**Cómo se informa (no decide), según ISO/IEC TS 4213:**
- La matriz de confusión de cada método en cada tipo, con la sensibilidad (1 − no detección), la especificidad (1 − falsas), la precisión y el F1.
- El IC95 exacto (Clopper-Pearson) de cada tasa, y el AUC por pieza con su IC95 por remuestreo (2.000, semilla 0).
- La clase AIAG (aceptable, marginal o inaceptable), la detección por tipo de defecto y el AUC de cada vista en MANTA.
- La latencia en CPU y en la GTX 1060, el rendimiento en piezas por minuto, la memoria máxima de la GPU y el tiempo de ajuste.
- Una curva de pocas muestras, después de guardar la decisión: el AUC con 1, 4, 16 y 64 piezas que ajustan (5 semillas), para los métodos sin entrenamiento y para WinCLIP sin ninguna. Responde cuántas buenas hacen falta para calibrar un tipo nuevo.
- Los mismos métodos en los sets de desarrollo: Kaggle, Edge Impulse (no ciego), las propias y las sintéticas.

## Orden y sellado
1. **Este ADR va en un commit antes de bajar nada.**
2. **Descarga** al servidor: MANTA Tiny (`mechanics.tar.gz`, 3.197.059.395 bytes, con el MD5 que publica figshare) y los sets de Roboflow que se exporten. Las huellas quedan fijadas en `scripts/descargar_datos.py`.
3. **Sello.** Una lista por tipo con la huella SHA-256 de cada archivo que ajusta, calibra y prueba, armada con las rutas y los bytes, sin decodificar imágenes. Va en un commit propio.
4. **Código de e13 y prueba de humo**, solo con los sets de desarrollo. El control: con la configuración de e12, el PatchCore propio tiene que repetir en Kaggle sus números de e12.
5. **Perfil de cada tipo**, con las que ajustan (commit).
6. **Corrida única de e13** en el servidor, en segundo plano y mirando `nvidia-smi` antes.
7. Después de guardar la decisión, lo que solo informa.

Hasta el paso 6, ningún programa decodifica una imagen de prueba.

## Qué cambia en el código
| Dónde | Cambio |
|---|---|
| `golillas/tipos/` (nuevo) | Un perfil JSON por tipo: dataset, vistas, vista de arriba, polaridad, escala y si aplica la clásica |
| `golillas/conjuntos.py` (nuevo) | Cargadores que entregan piezas (id, vistas, mala, defecto, grupo) de MANTA (corta las 5 vistas), Roboflow (recorta las cajas COCO), Edge Impulse, Kaggle y las propias |
| `golillas/metodos/` (nuevo) | Una interfaz común, `ajustar(piezas)` y `puntajes(piezas)`, con adaptadores para la clásica (normaliza la escala), el PatchCore propio (sale de e09) y los modelos de anomalib |
| `golillas/decision.py` | Suma el IC exacto, que hoy vive en e12, y la clase AIAG |
| `golillas/sello.py` (nuevo) | Sella y verifica las particiones de los datasets, con la idea de `sesion.py` |
| `experimentos/e13_estado_del_arte.py` (nuevo) | El protocolo de este ADR, con `--humo` |
| `scripts/descargar_datos.py` | La entrada de MANTA Tiny (ya agregada) y las de Roboflow cuando se exporten |
| `pyproject.toml` | Un extra `sota` con anomalib 2.7.1 y torch 2.14.1+cu126 fijados |
| `tests/` | Perfiles, cargadores (con imágenes falsas de 256 × 1.280), sello e interfaz de métodos |
| `data/ATRIBUCIONES.md` | MANTA y Roboflow, cuando se bajen |

**El README** cambia después de e13. La portada pasa a ser la tabla «método × tipo de golilla», con la clásica contra la IA y la clase AIAG de cada uno. La historia de las versiones 1 a 4 queda como el camino que llevó hasta ahí. Mientras tanto, solo cambian los próximos pasos.

## Riesgos
- **Las golillas de MANTA no son metálicas.** Se aceptan igual como tipos de golilla. Los sets metálicos de Roboflow pueden compensarlo si pasan la regla de tamaño.
- **Los defectos de MANTA se hicieron a mano,** según su paper. Pueden ser más fáciles o más difíciles que los de fábrica.
- **Si ningún set de Roboflow pasa la regla, deciden solo los dos tipos de MANTA,** que comparten montaje. Los otros montajes (Edge Impulse, Kaggle) solo informan.
- **La latencia en CPU deja fuera a casi toda la IA reciente** (medición previa). Eso no es un defecto del estudio: es lo que cuesta llevar esos modelos a una línea sin GPU.
- **La licencia de MANTA no está resuelta por sus autores.** Mientras no aclaren, se trata como no comercial.
- **Los umbrales del AIAG vienen de guías secundarias.** El manual es pagado; si Camilo consigue una copia (biblioteca de la USACH), se cita con su página.

## Lo que no se hace
- Bajar o mirar los sets de prueba antes de que este ADR esté en un commit, o elegir algo con ellos.
- Cambiar los criterios de los ADR 0014, 0015 y 0016.
- Subir al repo imágenes de los datasets.
- Usar métodos, pesos o datos con licencia sin leer en la fuente, o servicios cuyos términos prohíben publicar comparaciones.

Volver al [árbol](../arbol.md).
