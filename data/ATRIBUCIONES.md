# Atribuciones de datos

| Datos | Origen | Licencia | En el repo |
|---|---|---|---|
| `data/real/` (21 fotos) | Proyecto MATLAB original del autor | MIT, como el resto del repo | Sí |
| `data/publico/edge_impulse_flat_washers/` | Edge Impulse, «Visual Anomaly Detection - Flat washers» (2024), https://studio.edgeimpulse.com/public/497653/latest | BSD 3-Clause Clear | No: se baja con `python scripts/descargar_datos.py edge_impulse_flat_washers` |
| `data/publico/csem_misd_washers/` | Honzátko et al., CSEM-MISD (2021), https://zenodo.org/records/5513769 | CC BY-NC-ND 4.0: se usa sin modificar ni redistribuir | No: `python scripts/descargar_datos.py csem_misd_washers` |
| Roboflow «Washer Defects» (MassMEP) | https://universe.roboflow.com/massmep/washer-defects-3usoj | CC BY 4.0 | No: pide cuenta, se baja a mano |
| Kaggle «Simplified washers» (thibdx) | https://www.kaggle.com/datasets/thibdx/simplified-washers-for-anomaly-detection | CC BY-SA 4.0 | No: pide cuenta, se baja a mano |

Cita sugerida por Edge Impulse:

```bibtex
@misc{edgeimpulse_dataset_497653,
    title = {Visual Anomaly Detection - Flat washers},
    author = {Edge Impulse},
    year = {2024},
    url = {https://studio.edgeimpulse.com/public/497653/latest},
    note = {BSD 3-Clause Clear}
}
```
