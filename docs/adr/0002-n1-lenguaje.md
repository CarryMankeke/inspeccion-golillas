# ADR 0002 · N1: Python + OpenCV en vez de MATLAB

**Estado:** aceptada · **Fecha:** 2026-11-27 · **Nivel:** N1

## Contexto
El proyecto original estaba en MATLAB (Image Processing Toolbox). La licencia universitaria existe, pero:
- el servidor donde corren los experimentos no tiene MATLAB;
- un repositorio público debe poder ejecutarlo cualquiera, sin licencia;
- las pruebas automáticas (CI de GitHub) necesitan algo que se instale con `pip`.

## Opciones
1. Seguir en MATLAB.
2. Python con OpenCV, NumPy, scikit-learn y SciPy.
3. C++ con OpenCV.

## Decisión
**Python 3.12 + OpenCV.** Es gratis, corre en el servidor y en CI, y OpenCV tiene las mismas operaciones que el toolbox de MATLAB. C++ sería más rápido, pero la cadena ya cabe en ≈ 36 ms por pieza en Python, muy por debajo de la meta de 100 ms.

Esta decisión **no se midió**: la deciden las restricciones (licencia, servidor, CI). Para no perder la comparación con el trabajo original, las reglas del MATLAB se portaron a Python y se midieron como línea base ([ADR 0003](0003-n1-linea-base-matlab.md)).

## Consecuencias
- Dependencias fijadas en `pyproject.toml`. Las redes neuronales (PyTorch) van aparte, en el extra `red`, porque solo las usa el experimento e08.
- El port del MATLAB es de las reglas, no del binario: no se verificó contra MATLAB.

Volver al [árbol](../arbol.md).
