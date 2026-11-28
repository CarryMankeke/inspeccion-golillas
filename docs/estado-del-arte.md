# Estado del arte: qué existe para inspeccionar golillas

Catastro del 28 nov 2026 de los datasets, métodos, librerías y servicios que podrían inspeccionar golillas en una cinta. Cada licencia se leyó en su fuente: el archivo LICENSE, la ficha del dataset o del modelo, o los términos del servicio. Lo que no se pudo abrir va marcado **no verificado**. Este catastro alimenta el plan de comparación del [ADR 0017](adr/0017-n0-v2-estado-del-arte.md).

Las licencias se resumen para decidir qué usar en este repo público (MIT). Este resumen no es asesoría legal.

## 1. Datasets con golillas buenas y malas
Solo sirven los datasets con fotos reales de golillas que traigan piezas buenas y malas. Sin buenas no se puede calibrar ni medir falsas alarmas.

| Dataset | Golilla | Buenas · malas | Licencia | Acceso | Veredicto |
|---|---|---|---|---|---|
| [MANTA Tiny](https://figshare.com/articles/dataset/manta-tiny/27934938) `red_washer` (CVPR 2025) | Golilla roja. 5 vistas por pieza, pegadas en una imagen de 256 × 1.280 px (según la fuente). Forma y material **no verificados** | 578 buenas de entrenamiento; prueba con 148 buenas y 74 malas (`distribution.md` del mismo depósito) | CC BY 4.0 en figshare; CC BY-NC-SA 4.0 en la [página del proyecto](https://grainnet.github.io/MANTA). Se respeta la más restrictiva | Directo, sin cuenta: `mechanics.tar.gz`, 3,2 GB, con 11 piezas mecánicas | **Sirve** |
| MANTA Tiny `yellow_green_washer` | Golilla amarillo-verde, mismo montaje | 584 de entrenamiento; prueba con 144 y 72 | Igual | En el mismo archivo | **Sirve** |
| MANTA completo ([red](https://figshare.com/articles/dataset/manta-mechanics-button_gear_nut_nut_cap_red_washer/27937518), [yellow-green](https://figshare.com/articles/dataset/manta-mechanics-terminal_screw_wrie_cap_square_button_cap_yellow_green_washer/27937521)) | Las mismas, a resolución nativa | Conteo por pieza **no verificado** (el paper da el total: 128.721 buenas y 8.617 malas en 38 piezas) | Igual | Directo: 3,19 GB y 1,40 GB | Sirve, pero sin conteos publicados |
| [Roboflow «Washer Classifier»](https://universe.roboflow.com/cap-inspection-cioib/washer-classifier-fohnr) | Clases `washer_ok` y `washer_ng`; anillo metálico oscuro (según la miniatura) | 443 en total; reparto **no verificado** | CC BY 4.0 | Ver sin cuenta; bajar pide cuenta | Sirve con reparos: falta el conteo |
| [Roboflow saksit «metal washer»](https://universe.roboflow.com/saksit-zdivw/metal-washer) | 313 fotos a 640 × 640 (train 194, valid 60, test 59) | Clases `good`, `scratch`, `bent`, `color` y `flip`; reparto **no verificado** | Dominio público | Pide cuenta | Sirve con reparos: falta el conteo |
| [Roboflow renuka-p «Washer Defect Detection»](https://universe.roboflow.com/renuka-p/washer-defect-detection-ap9hk) | Golillas planas de bronce; 151 fotos (train 60, valid 45, test 46) | Clases `No Defect`, `DENT`, `BULGE`, `CUTOUT`, `DULL` y `DULLCUTOUT`; reparto **no verificado** | CC BY 4.0 | Pide cuenta | Sirve con reparos: pocas fotos |
| [Edge Impulse «Flat washers»](https://docs.edgeimpulse.com/datasets/image/flat-washers) | Golillas planas, dos fondos | 273 + 88 buenas, 103 malas | BSD 3-Clause Clear | Directo | Ya en uso: desarrolla (su prueba se vio en e10) |
| [Kaggle «Simplified washers»](https://www.kaggle.com/datasets/thibdx/simplified-washers-for-anomaly-detection) | Dibujos, no fotos | 1.000 + 100 buenas, 100 malas | CC BY-SA 4.0 | Pide cuenta | Ya en uso: desarrolla |

**No sirven** (revisados el 27 y 28 nov):
- Sin buenas: Roboflow MassMEP, Roboflow «washer-defect-detection-z3abq» (project-1), CSEM-MISD (además, CC BY-NC-ND), Roboflow «oring».
- No son golillas sueltas: Real-IAD y Real-IAD Variety (solo `retaining_ring`), MVTec LOCO `screw_bag` (la unidad es la bolsa), NKUST, VISION Ring, HSS-IAD, MPDD, MVTec AD, MVTec AD 2, VisA, BTAD, Eyecandies, GoodsAD, MAD, GC10-DET, NEU y MSDD (de memoria, sin volver a abrirlos el 28 nov).
- Una sola clase: Roboflow «washers» de hardee (sirve para ubicar golillas, no para juzgarlas).
- Sin revisar: 3CAD, Real-IAD D3 y RAD (no cargó su lista de piezas), y los datasets en chino de «gasket» y «washer». En Hugging Face, «washer» solo devuelve lavadoras.

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
| EfficientAD | Alumno-maestro más autoencoder; el más rápido | Apache-2.0 (anomalib) | El maestro se baja de un release de anomalib sin licencia declarada, y el entrenamiento baja Imagenette (repo Apache-2.0; imágenes de ImageNet): **no verificado** | Sí (tamaño S) |

### De cero o pocas muestras
| Método | Idea | Licencia del código | Pesos | Notas |
|---|---|---|---|---|
| WinCLIP (CVPR 2023) | CLIP compara ventanas de la foto con frases como «golilla buena» o «golilla dañada» | Apache-2.0 (anomalib) | OpenCLIP ViT-B-16-plus-240 `laion400m_e31`: la licencia de esos pesos es **no verificada** (OpenCLIP no la declara) | Funciona sin ninguna buena |
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
| CLIP de OpenAI y OpenCLIP | Código MIT; pesos **no verificados** | |

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
| MVTec HALCON | Edición de estudiante gratis por un semestre, solo para docencia o investigación; su cláusula sobre publicar resultados **no verificada** | **no verificada** |

Este repo usa OpenCV, scikit-image y SciPy.

## 4. Servicios y software comercial
El dato clave es si sus términos dejan publicar una comparación. Muchos contratos de software (EULA) lo prohíben.

| Producto | Acceso gratis | ¿Deja publicar resultados? | Estado |
|---|---|---|---|
| LandingAI LandingLens | 1.000 créditos sin tarjeta | **No**: sus [términos](https://landing.ai/terms-of-service) (5(b)(viii)) prohíben usarlo «for benchmarking or competitive analysis» | Vigente |
| Roboflow (entrenar e inferir) | Plan gratis con 10 créditos al mes | No se vio una cláusula en contra | Vigente; entrena modelos supervisados |
| Edge Impulse (Qualcomm) | Plan Developer gratis para estudiantes; trae anomalías de una clase (FOMO-AD) | Términos del plan **no leídos** (no cargaron) | Vigente |
| Intel Geti | Abierto, Apache-2.0 | Sí | Vigente; usa anomalib por dentro |
| NVIDIA TAO | Gratis con licencia de NVIDIA | **No verificado** | Vigente |
| MVTec HALCON y MERLIC | Edición de estudiante de HALCON | **No verificado** | Vigente |
| Zebra Aurora (Vision Studio Lite, MIL, Design Assistant) | Lite gratis para uso no comercial; MIL con prueba de 30 días | **No verificado** | Vigente |
| Euresys Open eVision, Teledyne Astrocyte | Pruebas de 30 a 60 días | **No verificado** | Vigentes |
| Cognex, Keyence, SICK, Siemens | Sin prueba pública que se haya visto | **No verificado** | — |
| Azure AI Custom Vision | — | — | Se retira el 25 sep 2028 ([Microsoft](https://learn.microsoft.com/azure/ai-services/custom-vision-service/whats-new)) |
| AWS Lookout for Vision | — | — | Descontinuado el 31 oct 2025 (según la documentación de AWS; la página no se leyó) |
| Google AutoML Vision (legado) | — | — | Apagado el 31 jul 2024 ([Google](https://docs.cloud.google.com/vertex-ai/docs/deprecations)) |

## 5. La GPU: qué versión de torch sirve
La GTX 1060 es Pascal (sm_61). Las ruedas de PyTorch con CUDA 12.8 y 12.9 dejaron de traer Pascal desde la 2.8 ([aviso](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)), y las de CUDA 13 empiezan en Turing. Desde la 2.15 ya no habrá ruedas con CUDA 12.6 ([aviso](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)). El servidor usa torch 2.14.1 con CUDA 12.6, que trae kernels sm_60 y corre en la 1060 (e08 a e12 corrieron ahí). Hay que fijar torch 2.14, y anomalib 2.7 lo acepta.

## 6. Qué entra en la comparación
Entra lo que se puede ajustar solo con buenas, corre en este servidor y tiene licencias verificadas que dejan publicar resultados: la clásica propia, PatchCore (propio y de anomalib), PaDiM, AnomalyDINO y Dinomaly. EfficientAD y WinCLIP entran solo si la licencia de sus pesos se verifica antes de sellar los datos. Los servicios comerciales quedan fuera de esta versión. La lista, el protocolo y el criterio están en el [ADR 0017](adr/0017-n0-v2-estado-del-arte.md).

Volver al [índice](README.md).
