# Atribuciones de datos

| Datos | Origen | Licencia | En el repo |
|---|---|---|---|
| `data/real/` (21 fotos) | Proyecto MATLAB original del autor | MIT, como el resto del repo | Sí |
| `data/publico/edge_impulse_flat_washers/` | Edge Impulse, «Visual Anomaly Detection - Flat washers» (2024), https://studio.edgeimpulse.com/public/497653/latest | BSD 3-Clause Clear | No: se baja con `python scripts/descargar_datos.py edge_impulse_flat_washers` |
| `data/publico/csem_misd_washers/` | Honzátko et al., CSEM-MISD (2021), https://zenodo.org/records/5513769 | CC BY-NC-ND 4.0: se usa sin modificar ni redistribuir | No: `python scripts/descargar_datos.py csem_misd_washers` |
| `data/publico/roboflow_massmep_washers/` | MassMEP, «Washer Defects» v1 en Roboflow Universe, https://universe.roboflow.com/massmep/washer-defects-3usoj (exportado en COCO el 27 nov 2026) | CC BY 4.0 | No: pide cuenta; se exporta a mano y se comprueba con `python scripts/descargar_datos.py --verificar roboflow_massmep_washers` |
| `data/publico/kaggle_simplified_washers/` | Thibdx, «Simplified washers for anomaly detection», versión 5 (2022), https://www.kaggle.com/datasets/thibdx/simplified-washers-for-anomaly-detection | CC BY-SA 4.0 | No: pide cuenta; se baja con el conector o la CLI de Kaggle y se comprueba con `python scripts/descargar_datos.py --verificar kaggle_simplified_washers` |
| `data/publico/manta_tiny_mechanics/` (solo `red_washer` y `yellow_green_washer`) | Fan et al., MANTA Tiny (CVPR 2025), https://figshare.com/articles/dataset/manta-tiny/27934938 (bajado el 28 nov 2026) | CC BY-NC-SA 4.0 según la página del proyecto (figshare marca CC BY 4.0; se respeta la más restrictiva) | No: `python scripts/descargar_datos.py manta_tiny_mechanics` |
| `data/publico/roboflow_saksit_metal_washer/` | Saksit, «metal washer» v4 en Roboflow Universe, https://universe.roboflow.com/saksit-zdivw/metal-washer (exportado en COCO el 28 nov 2026) | Dominio público | No: pide cuenta; se exporta a mano y se descomprime con `python scripts/descargar_datos.py roboflow_saksit_metal_washer --zip <zip>` |
| `data/publico/roboflow_renuka_washer_defect/` | renuka-p, «Washer Defect Detection» v3 en Roboflow Universe, https://universe.roboflow.com/renuka-p/washer-defect-detection-ap9hk (exportado en COCO el 28 nov 2026) | CC BY 4.0 | Igual, con `roboflow_renuka_washer_defect` |
| `data/publico/roboflow_washer_classifier/` | Cap inspection, «Washer Classifier» v3 en Roboflow Universe, https://universe.roboflow.com/cap-inspection-cioib/washer-classifier-fohnr (exportado en carpetas por clase el 28 nov 2026) | CC BY 4.0 | Igual, con `roboflow_washer_classifier` |

## Aviso de Edge Impulse

`docs/figuras/e10_ejemplos.jpg` muestra 8 fotos reducidas del dataset de Edge Impulse. La licencia BSD 3-Clause Clear permite redistribuirlas si se reproducen el aviso de copyright y el texto de la licencia.

Edge Impulse no publica una línea de copyright propia del dataset (páginas leídas el 27 nov 2026). La [ficha del dataset](https://docs.edgeimpulse.com/datasets/image/flat-washers) da como autor a Edge Impulse (2024), y la [página del proyecto](https://studio.edgeimpulse.com/public/497653/latest) lleva el pie «© 2026 EdgeImpulse Inc. All rights reserved». El aviso de abajo se armó con esos dos datos. Las dos páginas enlazan la licencia al texto de SPDX, que se copia tal cual desde https://spdx.org/licenses/BSD-3-Clause-Clear.html (los corchetes son parte del texto de SPDX).

```text
"Visual Anomaly Detection - Flat washers" (2024), Edge Impulse (EdgeImpulse Inc.).
https://studio.edgeimpulse.com/public/497653/latest
Licensed under the BSD 3-Clause Clear License:

The Clear BSD License

Copyright (c) [xxxx]-[xxxx] [Owner Organization]
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

     * Neither the name of [Owner Organization] nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Aviso de Roboflow MassMEP

«Washer Defects», de MassMEP, en Roboflow Universe (https://universe.roboflow.com/massmep/washer-defects-3usoj), con licencia CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Ninguna de sus imágenes está en el repo.

## Aviso de Kaggle «Simplified washers»

«Simplified washers for anomaly detection», de Thibdx, en Kaggle (https://www.kaggle.com/datasets/thibdx/simplified-washers-for-anomaly-detection), con licencia CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Son dibujos hechos en Affinity Publisher, no fotos. Ninguna de sus imágenes está en el repo; si alguna figura llega a mostrar una, esa figura queda bajo CC BY-SA 4.0.

## Aviso de MANTA

«MANTA Tiny», de Fan et al. (CVPR 2025), en figshare (https://figshare.com/articles/dataset/manta-tiny/27934938). figshare lo publica con CC BY 4.0 y la página del proyecto (https://grainnet.github.io/MANTA) con CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/); mientras los autores no lo aclaren, se usa como CC BY-NC-SA 4.0. Ninguna de sus imágenes está en el repo; si alguna figura llega a mostrar una, esa figura queda bajo CC BY-NC-SA 4.0 y sin uso comercial. Los CSV de `data/sellos/manta_*` solo traen nombres de archivo y huellas.

## Avisos de Roboflow (ADR 0017)

- «metal washer», de Saksit, en Roboflow Universe (https://universe.roboflow.com/saksit-zdivw/metal-washer), versión 4, en dominio público.
- «Washer Defect Detection», de renuka-p, en Roboflow Universe (https://universe.roboflow.com/renuka-p/washer-defect-detection-ap9hk), versión 3, con licencia CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- «Washer Classifier», de Cap inspection, en Roboflow Universe (https://universe.roboflow.com/cap-inspection-cioib/washer-classifier-fohnr), versión 3, con licencia CC BY 4.0.

Ninguna de sus imágenes está en el repo. Los CSV de `data/sellos/roboflow_*` solo traen nombres de archivo y huellas.

## Cita de Edge Impulse

La que sugiere su ficha:

```bibtex
@misc{edgeimpulse_dataset_497653,
    title = {Visual Anomaly Detection - Flat washers},
    author = {Edge Impulse},
    year = {2024},
    url = {https://studio.edgeimpulse.com/public/497653/latest},
    note = {BSD 3-Clause Clear}
}
```
