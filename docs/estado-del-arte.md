# Estado del arte: qué existe para inspeccionar golillas

Catastro del 28 nov 2026 de las normas, datasets, métodos, librerías y servicios que sirven para inspeccionar golillas en una cinta. Cada licencia se leyó en su fuente: el archivo LICENSE, la ficha del dataset o del modelo, o los términos del servicio. Lo que no se pudo abrir lo dice, y no se usa para decidir. Este catastro alimenta el plan de comparación del [ADR 0017](adr/0017-n0-v2-estado-del-arte.md).

Las licencias se resumen para decidir qué usar en este repo público (MIT). Este resumen no es asesoría legal.

## 1. Datasets con golillas buenas y malas
Solo sirven los datasets con fotos reales de golillas que traigan piezas buenas y malas. Sin buenas no se puede calibrar ni medir falsas alarmas.

| Dataset | Golilla | Buenas · malas | Licencia | Acceso | Veredicto |
|---|---|---|---|---|---|
| [MANTA Tiny](https://figshare.com/articles/dataset/manta-tiny/27934938) `red_washer` (CVPR 2025) | Anillo plano moldeado, no metálico (inferencia de las figuras y de las descripciones de sus autores; ver el [ADR 0017](adr/0017-n0-v2-estado-del-arte.md)); cabe en 3 × 11 × 11 mm. 5 vistas por pieza | 578 buenas de entrenamiento; prueba con 148 buenas y 74 malas (`distribution.md` del mismo depósito) | CC BY 4.0 en figshare; CC BY-NC-SA 4.0 en la [página del proyecto](https://grainnet.github.io/MANTA). Se respeta la más restrictiva | Directo, sin cuenta: `mechanics.tar.gz`, 3,2 GB, con 11 piezas mecánicas | **Sirve** |
| MANTA Tiny `yellow_green_washer` | Anillo plano moldeado, grueso, no metálico; cabe en 3 × 8 × 8 mm. Mismo montaje | 584 de entrenamiento; prueba con 144 y 72 | Igual | En el mismo archivo | **Sirve** |
| MANTA completo ([red](https://figshare.com/articles/dataset/manta-mechanics-button_gear_nut_nut_cap_red_washer/27937518), [yellow-green](https://figshare.com/articles/dataset/manta-mechanics-terminal_screw_wrie_cap_square_button_cap_yellow_green_washer/27937521)) | Las mismas, a resolución nativa | 2.298 y 2.288 imágenes en total (tabla I del suplementario); figshare no publica el conteo por partición | Igual | Directo: 3,19 GB y 1,40 GB | Sirve, pero sin conteos publicados |
| [Roboflow «Washer Classifier»](https://universe.roboflow.com/cap-inspection-cioib/washer-classifier-fohnr) | Clases `washer_ok` y `washer_ng`; anillo metálico oscuro (según la miniatura) | 443 en total; el reparto pide cuenta | CC BY 4.0 | Ver sin cuenta; bajar pide cuenta | Sirve con reparos: falta el conteo |
| [Roboflow saksit «metal washer»](https://universe.roboflow.com/saksit-zdivw/metal-washer) | 313 fotos a 640 × 640 (train 194, valid 60, test 59) | Clases `good`, `scratch`, `bent`, `color` y `flip`; el reparto pide cuenta | Dominio público | Pide cuenta | Sirve con reparos: falta el conteo |
| [Roboflow renuka-p «Washer Defect Detection»](https://universe.roboflow.com/renuka-p/washer-defect-detection-ap9hk) | Golillas planas de bronce; 151 fotos (train 60, valid 45, test 46) | Clases `No Defect`, `DENT`, `BULGE`, `CUTOUT`, `DULL` y `DULLCUTOUT`; el reparto pide cuenta | CC BY 4.0 | Pide cuenta | Sirve con reparos: pocas fotos |
| [Edge Impulse «Flat washers»](https://docs.edgeimpulse.com/datasets/image/flat-washers) | Golillas planas, dos fondos | 273 + 88 buenas, 103 malas | BSD 3-Clause Clear | Directo | Ya en uso: desarrolla (su prueba se vio en e10) |
| [Kaggle «Simplified washers»](https://www.kaggle.com/datasets/thibdx/simplified-washers-for-anomaly-detection) | Dibujos, no fotos | 1.000 + 100 buenas, 100 malas | CC BY-SA 4.0 | Pide cuenta | Ya en uso: desarrolla |

**No sirven** (revisados el 27 y 28 nov):
- Sin buenas: Roboflow MassMEP, Roboflow «washer-defect-detection-z3abq» (project-1), CSEM-MISD (además, CC BY-NC-ND), Roboflow «oring».
- No son golillas sueltas: Real-IAD y Real-IAD Variety (solo `retaining_ring`), MVTec LOCO `screw_bag` (la unidad es la bolsa), NKUST, VISION Ring, HSS-IAD, MPDD, MVTec AD, MVTec AD 2, VisA, BTAD, Eyecandies, GoodsAD, MAD, GC10-DET, NEU y MSDD; y tampoco [3CAD](https://github.com/ShuCvlab/3CAD), [Real-IAD D3](https://arxiv.org/abs/2504.14221) ni los dos RAD (revisados el 28 nov).
- Una sola clase: Roboflow «washers» de hardee (sirve para ubicar golillas, no para juzgarlas).
- En Hugging Face, «washer» solo devuelve lavadoras. Los datasets en chino de «gasket» y «washer» no se revisaron.

## 2. Métodos con IA
### La librería: anomalib
[anomalib](https://github.com/open-edge-platform/anomalib) (Apache-2.0) es la librería abierta de referencia para detectar anomalías. La versión vigente es la 2.7.1 (17 nov 2026). Pide Python ≥ 3.10 y torch ≥ 2.10, y trae 34 modelos, entre ellos PatchCore, PaDiM, EfficientAD, FastFlow, Reverse Distillation, DRAEM, SuperSimpleNet, Dinomaly, INP-Former, AnomalyDINO y WinCLIP. No publica una tabla de resultados por pieza de MVTec, y su descarga automática de MVTec está rota.

### De una clase: se ajustan solo con buenas
| Método | Idea | Licencia del código | Pesos | ¿Cabe en la GTX 1060? |
|---|---|---|---|---|
| PatchCore ([Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265)) | Banco de parches de una red de ImageNet | Apache-2.0 (anomalib) | WideResNet-50 y ResNet-18 de torchvision: BSD-3 ([ficha](https://huggingface.co/timm/wide_resnet50_2.tv_in1k)) | Sí, con submuestra si el banco no cabe en 8 GB de RAM |
| PaDiM | Una gaussiana por posición | Apache-2.0 (anomalib) | ResNet-18: BSD-3 | Sí |
| AnomalyDINO ([WACV 2025](https://github.com/dammsi/AnomalyDINO)) | Vecinos más cercanos sobre parches de DINOv2 ViT-S/14 | Apache-2.0 (su repo y anomalib) | DINOv2: Apache-2.0 ([ficha](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)) | Sí. En MVTec AD, AUROC de imagen 96,6 con 1 buena y 98,4 con 16 (su README) |
| Dinomaly ([CVPR 2025](https://github.com/guojiajeremy/Dinomaly)) | Reconstruye los rasgos de DINOv2 ViT-B/14 | Apache-2.0 | DINOv2 con registros: Apache-2.0 ([ficha](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m)) | Justo, al entrenar. AUROC de imagen 99,7 en MVTec AD (su paper) |
| INP-Former (CVPR 2025) | Prototipos de la propia imagen sobre DINOv2 | MIT | Sus pesos en Google Drive no declaran licencia | Justo |
| EfficientAD | Alumno-maestro más autoencoder; el más rápido | Apache-2.0 (anomalib) | El maestro (zip de 38 MiB en un release de anomalib) no declara licencia, y el entrenamiento usa Imagenette, cuyas imágenes heredan el uso solo no comercial de ImageNet: **queda fuera** | Sí (tamaño S) |

### De cero o pocas muestras
| Método | Idea | Licencia del código | Pesos | Notas |
|---|---|---|---|---|
| WinCLIP (CVPR 2023) | CLIP compara ventanas de la foto con frases como «golilla buena» o «golilla dañada» | Apache-2.0 (anomalib) | OpenCLIP ViT-B-16-plus-240 `laion400m_e31`: MIT según su [ficha](https://huggingface.co/timm/vit_base_patch16_plus_clip_240.laion400m_e31); las imágenes de LAION-400M siguen bajo el copyright de sus dueños | Funciona sin ninguna buena |
| MuSc (ICLR 2024) | Compara las fotos de prueba entre sí, sin entrenar | MIT | CLIP | 270 ms por imagen en una RTX 3090, según su repo: no cumple los 100 ms |
| AnomalyCLIP, AdaCLIP, APRIL-GAN | CLIP afinado en otro dataset | MIT | Sin licencia declarada | AdaCLIP: 50 s por imagen en CPU, según una demo |
| UniVAD (CVPR 2025) | SAM-HQ ViT-H más GroundingDINO | **CC BY-NC-SA 4.0** | De terceros | No cabe en 6 GB |

### Modelos base
| Modelo | Licencia de los pesos | Nota |
|---|---|---|
| DINOv2 | Apache-2.0 | La base de AnomalyDINO y Dinomaly |
| DINOv3 | Licencia propia de Meta: obliga a incluir el acuerdo y a reconocer el uso, y tiene acceso restringido | Se evita en este repo |
| SigLIP 2 | Apache-2.0 | |
| SAM 2 | Apache-2.0 | |
| SAM 3 | Licencia propia de Meta, casi igual a la de DINOv3 | Se evita |
| OpenCLIP | Código MIT; cada peso tiene su ficha (el de WinCLIP es MIT) | |

### Detectores supervisados (necesitan malas etiquetadas)
| Detector | Licencia | Nota |
|---|---|---|
| Ultralytics YOLO (YOLO26) | **AGPL-3.0**, o licencia de pago | Usarlo desde un repo MIT obliga a revisar la AGPL: se evita |
| RF-DETR | Apache-2.0 en Nano, Small, Medium y Large; **PML 1.0** en Atto, Femto, Pico, XL y 2XL | Nano o Small, si algún día se usan malas para entrenar |
| YOLOX, RT-DETR, D-FINE, DEIM | Código Apache-2.0; pesos sin licencia declarada | |

### MATLAB
La librería Automated Visual Inspection de MATLAB trae FCDD (R2022b), FastFlow y PatchCore (R2023a) y EfficientAD (R2024b) ([documentación](https://www.mathworks.com/help/vision/automated-visual-inspection.html)). Repite métodos que anomalib ya trae, y MATLAB no está instalado en el servidor.

## 3. Visión clásica: librerías
| Librería | Licencia | Versión |
|---|---|---|
| OpenCV y opencv-contrib | Apache-2.0 | 5.1.0 (el paquete omite SURF por su patente) |
| scikit-image | BSD-3 | 0.26.0 |
| SciPy (`ndimage`) | BSD-3 | 1.18.1 |
| Mahotas | MIT | 1.4.19 |
| SimpleITK | Apache-2.0 | 2.5.8 |
| MVTec HALCON | Edición de estudiante gratis por un semestre, solo para docencia o investigación; su cláusula sobre publicar resultados no cargó | — |

Este repo usa OpenCV, scikit-image y SciPy.

## 4. Servicios y software comercial
El dato clave es si sus términos dejan publicar una comparación. Muchos contratos de software (EULA) lo prohíben.

| Producto | Acceso gratis | ¿Deja publicar resultados? | Estado |
|---|---|---|---|
| LandingAI LandingLens | 1.000 créditos sin tarjeta | **No**: sus [términos](https://landing.ai/terms-of-service) (5(b)(viii)) prohíben usarlo «for benchmarking or competitive analysis» | Vigente |
| Roboflow (entrenar e inferir) | Plan gratis con 10 créditos al mes | No se vio una cláusula en contra | Vigente; entrena modelos supervisados |
| Edge Impulse (Qualcomm) | Plan Developer gratis para estudiantes; trae anomalías de una clase (FOMO-AD) | Sus términos no cargaron: queda fuera | Vigente |
| Intel Geti | Abierto, Apache-2.0 | Sí | Vigente; usa anomalib por dentro |
| NVIDIA TAO | Gratis con licencia de NVIDIA | Contrato no leído: queda fuera | Vigente |
| MVTec HALCON y MERLIC | Edición de estudiante de HALCON | Contrato no leído: queda fuera | Vigente |
| Zebra Aurora (Vision Studio Lite, MIL, Design Assistant) | Lite gratis para uso no comercial; MIL con prueba de 30 días | Contrato no leído: queda fuera | Vigente |
| Euresys Open eVision, Teledyne Astrocyte | Pruebas de 30 a 60 días | Contratos no leídos: quedan fuera | Vigentes |
| Cognex, Keyence, SICK, Siemens | Sin prueba pública que se haya visto | Quedan fuera | — |
| Azure AI Custom Vision | — | — | Se retira el 25 sep 2028 ([Microsoft](https://learn.microsoft.com/azure/ai-services/custom-vision-service/whats-new)) |
| AWS Lookout for Vision | — | — | Cerrado el 31 oct 2025 ([AWS](https://docs.aws.amazon.com/general/latest/gr/full_shutdown_services.html)) |
| Google AutoML Vision (legado) | — | — | Apagado el 31 jul 2024 ([Google](https://docs.cloud.google.com/vertex-ai/docs/deprecations)) |

## 5. La GPU: qué versión de torch sirve
La GTX 1060 es Pascal (sm_61). Las ruedas de PyTorch con CUDA 12.8 y 12.9 dejaron de traer Pascal desde la 2.8 ([aviso](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)), y las de CUDA 13 empiezan en Turing. Desde la 2.15 ya no habrá ruedas con CUDA 12.6 ([aviso](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)). El servidor usa torch 2.14.1 con CUDA 12.6, que trae kernels sm_60 y corre en la 1060 (e08 a e12 corrieron ahí). Hay que fijar torch 2.14, y anomalib 2.7 lo acepta.

## 6. Normas y referencias para los requisitos
| Norma o referencia | Qué aporta | Acceso |
|---|---|---|
| VDI/VDE/VDMA 2632-3:2017 y 2632-3.1:2020 | Cómo aceptar un sistema de visión que clasifica: muestras de referencia, falsa aceptación y falso rechazo, plan de muestreo y prueba estadística | Pagadas; se leyó la ficha y el índice |
| AIAG MSA, 4.ª ed. (análisis por atributos) | Aceptable: efectividad ≥ 90 %, no detección ≤ 2 %, falsa alarma ≤ 5 %. Marginal: ≥ 80 %, ≤ 5 %, ≤ 10 % | Pagado; los números vienen de guías secundarias ([TechieQuality](https://www.techiequality.com/2024/03/28/attribute-msa-how-to-do-attribute-msa-example-acceptance-criteria/)) |
| ISO/IEC TS 4213:2022 | Qué informar de un clasificador: matriz de confusión, sensibilidad, especificidad, intervalos, latencia y rendimiento | Pagada; se leyó el índice |
| ISO 3269:2019 e ISO 2859-1 ([NCh44:2007](https://ecommerce.inn.cl/nch44200744722)) | Aceptación de lotes de fijaciones, golillas incluidas | Pagadas |
| ISO 7089 e ISO 4759-3 | Dimensiones y tolerancias de golillas planas | Pagadas; la tabla de M8 y M10 viene de [Westfield](https://westfieldfasteners.co.uk/Standards/Washer_ISO7089_M.html) |
| EMVA 1288 | Caracterizar cámaras (ruido, sensibilidad) | Gratis |
| DS 132 (Reglamento de Seguridad Minera) | Pide inspeccionar las correas y guardar los registros; nada sobre visión | Gratis |

Ninguna norma fija la latencia. La referencia es el ritmo de las clasificadoras ópticas de golillas: hasta 600 piezas por minuto ([Fastener+Fixing](https://fastenerandfixing.com/manufacturing-technology/new-optical-sorting-machines-drive-efficiency)), o sea, 100 ms por pieza. En CPU, con 5 vistas, la red base de PatchCore con WideResNet-50 ya tarda 541 ms, la de AnomalyDINO 288 ms, la de WinCLIP 886 ms y la de Dinomaly 2,7 s; la de PaDiM, 79 ms, y la del PatchCore propio, 65 ms (`scripts/latencia_cota.py`).

## 7. Qué entra en la comparación
Entra lo que se puede ajustar solo con buenas, corre en este servidor y tiene licencias leídas que dejan publicar resultados: la clásica propia, PatchCore (propio y de anomalib), PaDiM, AnomalyDINO, Dinomaly y WinCLIP. EfficientAD queda fuera porque los pesos de su maestro no declaran licencia. Los servicios comerciales quedan fuera de esta versión. Las golillas encimadas van en el [ADR 0018](adr/0018-p0-alimentacion-encimadas.md). La lista, el protocolo y el criterio están en el [ADR 0017](adr/0017-n0-v2-estado-del-arte.md).

## 8. Lo que e13 no compara: catastro de la v6 por familia (28 nov 2026)
**¿e13 compara todo lo que la clásica y la IA permiten bajo estas reglas? No.** e13 es una muestra fija de 8 métodos, elegida antes de bajar los datos: uno por familia de IA que entonces se conocía, y la clásica de forma. Esta sección revisa todo lo que trae anomalib 2.7.1 (33 modelos de imagen, etiqueta `lib/v2.7.1`) y la clásica que falta, y pasa cada familia por el embudo del [ADR 0019](adr/0019-v6-metodo-de-seleccion.md). Fuentes: el código de anomalib en esa etiqueta, el LICENSE de cada repo original y la ficha de cada peso, leídos el 28 nov 2026. Lo que no cargó va marcado.

### Regla pareja de licencias (desde la v6)
Antes, el criterio fue disparejo. A EfficientAD se le restó que Imagenette viene de ImageNet, que es de uso no comercial, pero PatchCore y PaDiM también usan pesos entrenados en ImageNet, y DINOv2 y CLIP se entrenaron en LVD-142M y LAION-400M. La regla nueva es una sola:

> **Un método entra si su código y sus pesos declaran, cada uno, una licencia que permita redistribuir y publicar resultados (permisiva o copyleft compatible). Un peso sin licencia declarada no entra, aunque el código sea permisivo. Los datos de preentrenamiento no excluyen por sí solos: se anota su licencia, y una nota marca lo que impediría el uso por un cliente comercial.**

Con la regla pareja, **ninguno de los 8 de e13 sale**, y **EfficientAD sigue fuera**, ahora por una sola razón: el zip de su maestro en el release de anomalib no declara licencia. Imagenette es Apache-2.0, pero sus imágenes son de ImageNet, y eso queda solo como nota. **Nota para un cliente comercial, que vale para casi todos:** los pesos de timm y torchvision declaran Apache-2.0 o BSD-3, pero se entrenaron en ImageNet-1k, cuyos términos son de uso no comercial. Ninguna fuente primaria resuelve si esos términos alcanzan al modelo (**no verificado**), y lo tiene que revisar el cliente. DINOv2 se entrenó en LVD-142M, cuya licencia no está declarada. CLIP se entrenó en LAION-400M: sus metadatos son CC BY 4.0, pero cada imagen conserva el copyright de su dueño.

**MANTA** es CC BY 4.0 en figshare y CC BY-NC-SA 4.0 en la página del proyecto (las dos confirmadas). Se respeta la más restrictiva: sirve para este repo abierto y sin fines comerciales, pero **no para un cliente comercial**.

### IA: familias que faltan
Etapas del embudo: **2** = licencias, **3** = cota de latencia. La cota en la CPU del servidor viene de la [medición previa del ADR 0017](adr/0017-n0-v2-estado-del-arte.md#medición-previa-de-latencia-28-nov-2026-no-decide) cuando el método usa la misma red base. Por ejemplo, la Wide ResNet-50 hasta layer3 a 256 px tarda sola 103 ms por vista, o sea, 541 ms por pieza de 5 vistas. En la v6 cada método pasará por `scripts/latencia_cota.py`.

| Familia | Métodos (clase en anomalib 2.7.1) | Código · pesos · datos | ¿Malas? | v6 | Etapa que lo saca y por qué |
|---|---|---|---|---|---|
| Flujos normalizantes | FastFlow (`Fastflow`) | Apache-2.0 ([original](https://github.com/gathierry/FastFlow/blob/master/LICENSE)) · ResNet-18 de timm, Apache-2.0 · ImageNet-1k | No | **Entra** | Pasa la 2. Su red base, la ResNet-18, cabe (13 ms por vista); el flujo, por medir |
| | CFlow (`Cflow`) | BSD-3 ([original](https://github.com/gudovskiyd/cflow-ad/blob/master/LICENSE)) · WRN-50 de timm, Apache-2.0 · ImageNet-1k | No | Sale | 3: su red base es la WRN-50 (≥ 541 ms por pieza) |
| | CS-Flow (`Csflow`) | El [original](https://github.com/marco-rudolph/cs-flow) no trae LICENSE; anomalib dice MIT · EfficientNet-B5 de torchvision | No | Sale | 2: la licencia del original no está declarada (**no verificado** si anomalib copia su código) |
| | U-Flow (`Uflow`) | El [original](https://github.com/mtailanian/uflow) es AGPL-3.0; el port de anomalib dice Apache-2.0 sin citarlo · dos CaiT de timm, Apache-2.0 | No | Sale | 2: procedencia dudosa (**no verificado** si el port deriva del código AGPL); además, 448 px fijos con dos transformers (3) |
| Maestro-alumno | STFPM (`Stfpm`) | Apache-2.0 (anomalib); el código de los autores sería GPL-3.0 (**no verificado** que anomalib derive de él) · ResNet-18 de timm, Apache-2.0 | No | **Entra** | Pasa la 2 con nota. Corre el maestro y el alumno, dos ResNet-18 (≈ 2 × 13 ms por vista): por medir |
| | Reverse Distillation (`ReverseDistillation`) | MIT ([original](https://github.com/hq-deng/RD4AD/blob/main/LICENSE)) · WRN-50 · ImageNet-1k | No | Sale | 3: WRN-50. Su paper mide 0,31 s por imagen en una CPU i7 |
| | EfficientAD (`EfficientAd`) | Apache-2.0 · el maestro, sin licencia declarada · ImageNet + Imagenette | No | Sale | 2: pesos sin licencia |
| | UniNet (`UniNet`), L2BT (`L2BT`) | MIT o Apache-2.0 · WRN-50 de torchvision o DINOv2 ViT-B de timm | No (UniNet admite malas) | Sale | 3: WRN-50 (UniNet); ViT-B (L2BT, por medir, pero el ViT-S ya tarda 61 ms por vista) |
| Densidad | DFKDE (`Dfkde`) | Apache-2.0 (anomalib; sin repo del paper) · ResNet-18 de timm | No | **Entra** | Pasa la 2. No entrena: ResNet-18 hasta layer4 + PCA + KDE; por medir |
| | DFM (`Dfm`) | Apache-2.0 · ResNet-50 de timm | No | **Entra** | Pasa la 2. No entrena; la ResNet-50 hasta layer3, por medir |
| | CFA (`Cfa`) | El [original](https://github.com/sungwool/CFA_for_anomaly_localization) no trae LICENSE · WRN-50 de torchvision | No | Sale | 2 (licencia del original) y 3 (WRN-50) |
| Anomalías sintéticas | DRAEM (`Draem`) | MIT ([original](https://github.com/VitjanZ/DRAEM/blob/main/LICENSE)) · U-Net propia, sin pesos externos · texturas DTD, que Oxford ofrece «for research purposes», sin licencia | No | **Entra** | Pasa la 2 con nota: DTD no se sube al repo y no sirve para un cliente comercial. Dos U-Net: por medir |
| | DSR (`Dsr`) | Apache-2.0 · el VQ-VAE preentrenado no declara licencia | No | Sale | 2: pesos sin licencia |
| | SuperSimpleNet (`Supersimplenet`), GLASS (`Glass`), SimpleNet (no está en 2.7.1) | MIT · WRN-50 de timm o torchvision · GLASS corre sin DTD | No (SuperSimpleNet admite malas) | Sale | 3: WRN-50 |
| Reconstrucción | GANomaly (`Ganomaly`) | MIT ([original](https://github.com/samet-akcay/ganomaly/blob/master/LICENSE)) · DCGAN propio, sin pesos externos | No | **Entra** | Pasa la 2. Red chica sin preentrenamiento: por medir |
| | FRE (`Fre`) | Apache-2.0 · ResNet-50 de timm | No | **Entra** | Pasa la 2. ResNet-50 hasta layer3 + un autocodificador lineal: por medir |
| Memoria y DINO | MH-PatchCore (`MHPatchcore`) | MIT · WRN-50 | No | Sale | 3: WRN-50; además, repite PatchCore |
| | INP-Former, FoundAD, GeneralAD | MIT · DINOv2 ViT-B o ViT-L de timm, Apache-2.0 · LVD-142M | No | Sale | 3: ViT-B a 392 px, la red base de Dinomaly, ya tarda 519 ms por vista; FoundAD y GeneralAD van a 518 px |
| | RAD (`Rad`), SuperADD (`SuperADD`) | Apache-2.0 · DINOv3, con licencia propia de Meta (no cargó) | No | Sale | 2: licencia de DINOv3, como en el ADR 0017 |
| Cero disparos y lenguaje | AnomalyVFM (`AnomalyVFM`) | El original no trae LICENSE; los pesos en Hugging Face no declaran licencia | No | Sale | 2 |
| | VLM-AD (`VlmAd`) | Apache-2.0 · por defecto `gpt-4o-mini` por la API de OpenAI (manda las fotos a un tercero), o LLaVA local | No | Sale | 2 y 3: un servicio externo no se puede publicar ni auditar, y un LLaVA de 7.000 millones de parámetros no cabe en 100 ms |

**Lo que queda para la v6 por el lado de la IA:** FastFlow, STFPM, DFKDE, DFM, DRAEM, GANomaly y FRE. Todos se ajustan solo con buenas. Su latencia está **por medir** en la CPU del servidor con `scripts/latencia_cota.py`, después de e13.

Ojo con un dato de e13 que estas cotas no ven: el PatchCore propio y PaDiM tardaron 265 y 195 ms por pieza, aunque su red base sola tarda 65 y 79 ms. La búsqueda o el modelo que va después de la red base pesa tanto como ella. Por eso la v6 también mide los modelos optimizados (OpenVINO u ONNX, con INT8), para saber si los 100 ms son un problema del hardware o del modelo.

### Clásica que falta
Todo esto se puede hacer con OpenCV ([Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE)), scikit-image ([BSD-3](https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt)), scikit-learn ([BSD-3](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)) y NumPy (BSD-3). Ninguna técnica depende de los módulos «nonfree» de opencv-contrib. Ninguna fuente publica tiempos para estas funciones, así que su latencia queda **por medir** en el i3-9100.

| Técnica | Qué detecta | Implementación | Referencia | v6 |
|---|---|---|---|---|
| Color en HSV o Lab | Manchas, decoloración, pigmento fuera de tono | `cv2.cvtColor` + estadísticos por anillo | CIE (**no verificado**) | **Entra** |
| Textura: GLCM (Haralick), LBP, Gabor | Rayas, rugosidad, poros | `skimage.feature.graycomatrix` y `local_binary_pattern`; `cv2.getGaborKernel` | Haralick et al. 1973; Ojala et al. 2002; Daugman 1985 (DOIs comprobados en Crossref) | **Entra** |
| Morfología: top-hat y black-hat | Manchas claras u oscuras y rayas finas | `cv2.morphologyEx` | Serra 1982 (**no verificado**) | **Entra** |
| Plantilla dorada y modelo de variación | Cualquier diferencia con la buena promedio, píxel a píxel | Propia: alinear con `cv2.findTransformECC` o `cv2.phaseCorrelate`, luego media y desviación por píxel | Evangelidis y Psarakis 2008 (ECC); el «variation model» de HALCON (solo se leyeron resúmenes de su documentación) | **Entra** |
| Clasificadores de una clase | Combinan los rasgos de forma y de superficie | `OneClassSVM`, `IsolationForest`, `LocalOutlierFactor(novelty=True)`, `GaussianMixture`, `EllipticEnvelope` de scikit-learn | Schölkopf et al. 2001; Liu et al. 2008; Breunig et al. 2000 (DOIs comprobados) | **Entra** |
| Medición en mm con tolerancias ISO 7089 / ISO 4759-3 | Golillas fuera de medida | `cv2.calibrateCamera` con un patrón | Zhang 2000 (DOI comprobado) | No aplica a MANTA: su página no menciona calibración ni escala (figshare no revisado). Queda para fotos propias o una planta |

### Combinaciones
Las combinaciones (clásica con clásica, clásica con IA en fusión o en cascada, e IA con IA) entran al mismo embudo. Desde la v6, su latencia se exige con la media y el p99, no con la mediana ([ADR 0019](adr/0019-v6-metodo-de-seleccion.md)).

### Lo que no se verificó
- Si los términos de ImageNet alcanzan a los pesos derivados.
- Si los ports de U-Flow, STFPM y CS-Flow derivan del código original.
- La licencia de DINOv3, RADIO y FLUX.
- Los tiempos de casi todos los métodos nuevos.
- La documentación de OpenCV (dio 403).
- Si MANTA trae calibración en figshare.

Volver al [índice](README.md).
