# Créditos y citas

Este repo se apoya en el trabajo de otros. Aquí está, por cada método, dataset y herramienta, el paper que lo describe, el código que se usa y su licencia. Para citar este repo, usa [CITATION.cff](../CITATION.cff).

**Cómo se verificó (28 nov 2026).** Los títulos, autores y años de los papers se comprobaron en la API de arXiv (`export.arxiv.org`) y, para el de 2018 o antes sin arXiv, en Crossref. Las licencias vienen del archivo LICENSE de cada repo y de la ficha de cada peso, leídos para el [ADR 0017](adr/0017-n0-v2-estado-del-arte.md) y el [catastro](estado-del-arte.md). Lo que no se pudo leer en la fuente va marcado **no verificado**.

## Métodos de IA (e13)
| Método | Paper | Código que usa el repo · licencia | Pesos · licencia | Datos con que se entrenaron los pesos |
|---|---|---|---|---|
| PatchCore | Roth et al., «Towards Total Recall in Industrial Anomaly Detection», CVPR 2022, [arXiv 2106.08265](https://arxiv.org/abs/2106.08265) | [anomalib](https://github.com/open-edge-platform/anomalib) 2.7.1 · Apache-2.0 | Wide ResNet-50-2 de [timm](https://github.com/huggingface/pytorch-image-models) · Apache-2.0 o BSD-3, según la variante; cuál baja anomalib por defecto: **no verificado** | ImageNet-1k (términos de uso no comercial) |
| PatchCore propio | El mismo paper | Propio (`golillas/metodos/patchcore_propio.py`) · MIT | ResNet-18 de [torchvision](https://github.com/pytorch/vision) · BSD-3 | ImageNet-1k |
| PaDiM | Defard et al., «PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization», ICPR 2020, [arXiv 2011.08785](https://arxiv.org/abs/2011.08785) | anomalib 2.7.1 · Apache-2.0 | ResNet-18 · BSD-3 | ImageNet-1k |
| AnomalyDINO | Damm et al., «AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2», WACV 2025, [arXiv 2405.14529](https://arxiv.org/abs/2405.14529) | anomalib 2.7.1 · Apache-2.0 | DINOv2 ViT-S/14 de [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) · Apache-2.0 | LVD-142M (su licencia no está declarada) |
| Dinomaly | Guo et al., «Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection», CVPR 2025, [arXiv 2405.14325](https://arxiv.org/abs/2405.14325) | anomalib 2.7.1 · Apache-2.0 | DINOv2-reg ViT-B/14 · Apache-2.0 | LVD-142M |
| WinCLIP | Jeong et al., «WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation», CVPR 2023, [arXiv 2303.14814](https://arxiv.org/abs/2303.14814) | anomalib 2.7.1 · Apache-2.0 | CLIP ViT-B-16-plus-240 de [open_clip](https://github.com/mlfoundations/open_clip) ([ficha](https://huggingface.co/timm/vit_base_patch16_plus_clip_240.laion400m_e31)) · MIT | LAION-400M: metadatos CC BY 4.0; cada imagen tiene el copyright de su dueño |

Redes base y modelos de fondo: He et al., «Deep Residual Learning for Image Recognition», CVPR 2016, [arXiv 1512.03385](https://arxiv.org/abs/1512.03385). Zagoruyko y Komodakis, «Wide Residual Networks», BMVC 2016, [arXiv 1605.07146](https://arxiv.org/abs/1605.07146). Oquab et al., «DINOv2: Learning Robust Visual Features without Supervision», [arXiv 2304.07193](https://arxiv.org/abs/2304.07193). Radford et al., «Learning Transferable Visual Models From Natural Language Supervision», ICML 2021, [arXiv 2103.00020](https://arxiv.org/abs/2103.00020). Cherti et al., «Reproducible scaling laws for contrastive language-image learning», CVPR 2023, [arXiv 2212.07143](https://arxiv.org/abs/2212.07143). Schuhmann et al., «LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs», [arXiv 2111.02114](https://arxiv.org/abs/2111.02114).

Librería: Akcay et al., «Anomalib: A Deep Learning Library for Anomaly Detection», ICIP 2022, [arXiv 2202.08341](https://arxiv.org/abs/2202.08341).

## Visión clásica
Cadena propia (P1–P6, MIT) sobre [OpenCV](https://github.com/opencv/opencv) (Apache-2.0), [NumPy](https://github.com/numpy/numpy) (BSD-3) y [scikit-learn](https://github.com/scikit-learn/scikit-learn) (BSD-3). Ningún rasgo usa módulos con patente (los «nonfree» de opencv-contrib).

## Estadística
- **Umbral conformal:** Angelopoulos y Bates, «A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification», [arXiv 2107.07511](https://arxiv.org/abs/2107.07511). Bates et al., «Testing for Outliers with Conformal p-values», *Annals of Statistics*, [arXiv 2104.08279](https://arxiv.org/abs/2104.08279).
- **Comparación de dos clasificadores (McNemar exacto, ADR 0019):** Dietterich, «Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms», *Neural Computation* 10(7), 1998, [doi:10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197).

## Datasets
| Dataset | Cita | Licencia | Papel |
|---|---|---|---|
| MANTA Tiny (`red_washer`, `yellow_green_washer`) | Fan et al., «MANTA: A Large-Scale Multi-View and Visual-Text Anomaly Detection Dataset for Tiny Objects», CVPR 2025, [arXiv 2412.04867](https://arxiv.org/abs/2412.04867); datos en [figshare](https://figshare.com/articles/dataset/manta-tiny/27934938) | CC BY-NC-SA 4.0 según su [página](https://grainnet.github.io/MANTA) (figshare marca CC BY 4.0; se respeta la más restrictiva). **Sirve para este repo abierto, pero no para un cliente comercial** | Decide (e13) |
| Edge Impulse «Visual Anomaly Detection - Flat washers» | Edge Impulse (2024), [studio.edgeimpulse.com/public/497653](https://studio.edgeimpulse.com/public/497653/latest) | BSD 3-Clause Clear | Desarrolla |
| Kaggle «Simplified washers for anomaly detection» | Thibdx (2022), [kaggle.com/datasets/thibdx/…](https://www.kaggle.com/datasets/thibdx/simplified-washers-for-anomaly-detection) | CC BY-SA 4.0 | Desarrolla |
| Roboflow «metal washer» v4 | Saksit, [Roboflow Universe](https://universe.roboflow.com/saksit-zdivw/metal-washer) | Dominio público | Informa |
| Roboflow «Washer Defect Detection» v3 | renuka-p, [Roboflow Universe](https://universe.roboflow.com/renuka-p/washer-defect-detection-ap9hk) | CC BY 4.0 | Informa |
| Roboflow «Washer Classifier» v3 | Cap inspection, [Roboflow Universe](https://universe.roboflow.com/cap-inspection-cioib/washer-classifier-fohnr) | CC BY 4.0 | Informa |
| Roboflow «Washer Defects» v1 | MassMEP, [Roboflow Universe](https://universe.roboflow.com/massmep/washer-defects-3usoj) | CC BY 4.0 | Descartado: sin buenas |
| CSEM-MISD | Honzátko et al. (2021), [Zenodo 5513769](https://zenodo.org/records/5513769) | CC BY-NC-ND 4.0 | Descartado: sin buenas |

Los avisos completos que piden estas licencias están en [data/ATRIBUCIONES.md](../data/ATRIBUCIONES.md). Ninguna imagen de estos datasets está en el repo.

## Normas y referencias de los requisitos
AIAG MSA 4.ª ed. (criterio por atributos, vía guías secundarias), VDI/VDE/VDMA 2632-3 y 2632-3.1 (solo su índice), ISO/IEC TS 4213:2022 (solo su índice), ISO 7089 e ISO 4759-3 (tabla secundaria), ISO 2859-1 / NCh44:2007 e ISO 3269:2019 (solo sus fichas). El detalle y los enlaces están en «De dónde salen los números» del [ADR 0017](adr/0017-n0-v2-estado-del-arte.md#de-dónde-salen-los-números), y cada valor del repo, en [valores.md](valores.md).
