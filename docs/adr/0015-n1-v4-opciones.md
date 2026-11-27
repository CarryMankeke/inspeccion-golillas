# ADR 0015 · N1 v4: opciones después de e11, y el plan de la v4

**Estado:** propuesta, con el criterio fijado antes de correr · **Fecha:** 2026-11-27 · **Nivel:** N1 · **Script:** `experimentos/e12_n1_v4.py` (por escribir; la prueba ciega propia pasa a e13)

## Contexto
En la v3 (e11, [ADR 0014](0014-n1-v3-varios-montajes.md)) ningún candidato cumplió el criterio. Hay tres causas, y las tres están medidas:

1. **El umbral de PatchCore es demasiado prudente.** PatchCore ordena muy bien (AUC 0,96–0,98 en Kaggle), pero su umbral es el mayor puntaje de las buenas. Con ese umbral detecta solo 25–55 de 100 malas en Kaggle.
2. **La cadena clásica no se traslada.** En los dibujos de Kaggle no encuentra el agujero en 94 de 1.000 buenas. Además, con 6–7 rasgos y 14 buenas el umbral de Hotelling se afloja, y detecta 5/6 malas propias.
3. **La regla de respaldo ignoraba las falsas alarmas.** Por eso eligió una fusión que retira 19 de 100 buenas.

Una pista, vista en la v3 y por eso no decide nada: la clásica v3_7 deja pasar m01 (deformada) y PatchCore l1+l2+l3 proporcional deja pasar m04 (borde interior). Fallan en malas distintas.

## Opciones, ordenadas por impacto y costo
Fuentes revisadas el 27 nov 2026. Lo que no se pudo abrir en la fuente va marcado como **no verificado**.

| # | Opción | Qué problema de e11 ataca | Impacto | Costo |
|---|---|---|---|---|
| 1 | **Umbral conformal** con una tasa de falsas alarmas α fijada ([Angelopoulos y Bates, §4.4](https://arxiv.org/abs/2107.07511); [Bates et al., Ann. Stat. 2023](https://arxiv.org/abs/2104.08279)) | El umbral prudente de PatchCore, y el Hotelling que supone normalidad | Alto | Bajo: reusa e11, unos 15 min de GPU |
| 2 | **Regla de respaldo que respeta las falsas alarmas** | La regla débil de la v3 | Alto: evita elegir mal | Muy bajo |
| 3 | **Intervalos Beta-Binomial y modelo jerárquico de tasas por montaje** en PyMC ([ejemplo de PyMC](https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html); [Brown, Cai y DasGupta 2001](https://doi.org/10.1214/ss/1009213286), **no verificado** en la fuente) | Decir cuánto se espera en un montaje nuevo, con incertidumbre | Medio: no detecta más, pero informa mejor | Bajo a medio |
| 4 | **anomalib** ([Akcay et al., ICIP 2022](https://arxiv.org/abs/2202.08341)): PatchCore de referencia, PaDiM y EfficientAD, y validar con MVTec `metal_nut` | ¿Nuestro PatchCore está bien hecho? Además suma candidatos | Medio | Medio: instalar, caber en 6 GB, y el formulario de MVTec es de Camilo |
| 5 | **Segmentación robusta, desenrollado polar y plantilla dorada** para la clásica (HALCON `polar_trans_image_ext`, **no verificado**) | La clásica no mide en otros montajes, y los cortes radiales | Medio en el montaje propio; bajo en otros | Medio |
| 6 | **Mezcla gaussiana bayesiana** para la nube de buenas ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)) | Edge Impulse tiene dos fondos | Bajo a medio | Bajo |
| 7 | **Métodos para cambio de dominio**: ADShift ([Cao et al., ICCV 2023](https://arxiv.org/abs/2303.13845)) y RegAD ([Huang et al., ECCV 2022](https://arxiv.org/abs/2207.07361)) | Pasar a un montaje nuevo | Puede ser alto | Alto |
| 8 | **Cero o pocas muestras con CLIP**: WinCLIP ([CVPR 2023](https://arxiv.org/abs/2303.14814)), AnomalyCLIP ([ICLR 2024](https://arxiv.org/abs/2310.18961)) y MuSc ([ICLR 2024](https://arxiv.org/abs/2401.16753)) | Calibrar sin buenas del montaje nuevo | Puede ser alto, pero la latencia en CPU es dudosa | Alto: modelos grandes para una GTX 1060 |
| 9 | **DRAEM** ([ICCV 2021](https://arxiv.org/abs/2108.07610)): defectos sintéticos sobre buenas | Aprovechar el generador sintético | Incierto | Alto: se entrena por montaje |
| 10 | **Regresión logística bayesiana** para la fusión y **optimización bayesiana** de hiperparámetros ([Snoek et al.](https://arxiv.org/abs/1206.2944)) | Pesar la fusión y ajustar hiperparámetros | Bajo: necesita malas o sobreajusta la elección | Medio |

Lo que dice la literatura y cambia el plan:
- **El umbral conformal pide un mínimo de buenas para calibrar.** El umbral es el puntaje número ⌈(n+1)(1−α)⌉ de las n buenas de calibración, ordenadas de menor a mayor, y no existe si α < 1/(n+1) (Angelopoulos y Bates, apéndice D). Con α = 3 % hacen falta al menos 33 buenas, y con 15 el α más bajo posible es 6,25 %.
- **La garantía vale solo si las buenas nuevas vienen del mismo montaje y lote que las de calibración.** Un cambio de cámara o de lote la rompe. Un preprint de 2026 ([Thai y Thai](https://arxiv.org/abs/2610.00236), sin revisión por pares) midió 1,7 veces la tasa nominal al calibrar con pocas buenas. Por eso las falsas alarmas se siguen midiendo aparte.
- **Con n chico, la tasa real varía bastante entre sesiones de calibración.** La tasa condicionada a la calibración sigue una Beta(ℓ, n+1−ℓ), con ℓ = ⌊(n+1)α⌋ (Bates et al., Proposición 1).
- **Nuestro PatchCore tiene contra qué compararse.** El original da un AUROC de 99,7–100 en `metal_nut` ([Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265), tabla S1), y anomalib da 0,991 con ResNet-18. Eso permite validar nuestra implementación (opción 4).

## Decisión: la v4 ataca el umbral y la regla de respaldo
**La v4 hace las opciones 1 y 2, y suma los intervalos exactos de la opción 3 a cada tasa.** Son las que atacan la falla más grande que se midió: un método que ordena bien y queda mal por el umbral. Además son baratas, porque reusan todo e11. El modelo jerárquico en PyMC (el resto de la opción 3) y anomalib (opción 4) quedan para la v5. Las opciones 7–10 cuestan demasiado para lo que se sabe hoy.

## Criterio de la v4 (fijado antes de correr)
**Candidatos:** la clásica de 4 rasgos de hoy (la de los [ADR 0010](0010-p5-rasgos.md) y [0011](0011-p6-decision.md)), las clásicas v3_6 y v3_7, los 4 PatchCore de e11 y la fusión de la mejor clásica con el mejor PatchCore, elegidas dentro de cada familia con la misma regla. Las configuraciones son las de e11: no se busca nada.

**Umbral conformal partido, con α = 3 %:**
- **Sintéticas:** 150 buenas de entrenamiento ajustan (banco de PatchCore, media y covarianza de la clásica) y otras 150 calibran el umbral. Se mide en las 600 de validación.
- **Kaggle:** 500 buenas de entrenamiento ajustan y 500 calibran. Se mide en las 200 de prueba.
- **Edge Impulse:** las 273 buenas se parten en tres tercios: ajustar, calibrar y medir falsas alarmas. Ahora esa tasa mide algo de verdad, también en PatchCore.
- **Montaje propio:** 15 buenas no alcanzan para α = 3 %. Se usan las reglas de e11 (Hotelling y el máximo dejando una fuera), y se dice así.
- Las particiones se hacen al azar con semilla 0, antes de correr.
- En la fusión, cada parte usa α/2 = 1,5 %, para que las dos juntas no pasen del 3 %.
- Una pieza que la cadena clásica no logra medir se retira siempre y no calibra, como en e11.

**Requisitos:** los mismos del ADR 0014.
1. En el montaje propio: 6/6 malas en la prueba A, ≥ 95 % en la prueba R y < 100 ms en CPU.
2. Falsas alarmas: ≤ 1/15 en las buenas propias, ≤ 5 % en las buenas de prueba de Kaggle y ≤ 5 % en el tercio de Edge Impulse que mide.

**Elección:** entre los que cumplen 1 y 2, gana la mayor detección media (Kaggle y sintéticas). Si hay empate (≤ 2 puntos), gana la más fácil de explicar y después la más rápida.

**Respaldo nuevo:** si ninguno cumple 1 y 2, se toma, entre los que cumplen 2, la mayor detección media. Si ninguno cumple 2, se toma la menor tasa media de falsas alarmas. En los dos casos el resultado se informa como «ninguno cumple» y no se adopta.

**Cómo se informa:** cada tasa va con su IC95 exacto (Clopper-Pearson). El set de prueba de Edge Impulse se corre después de guardar la elección, solo para informar y marcado «no ciego».

## Qué cambia en el set ciego propio
Para que el umbral conformal funcione también en el montaje propio, la sesión de fotos del [ADR 0014](0014-n1-v3-varios-montajes.md) pasa de 15 a **50 buenas de calibración**: 16 ajustan (banco y covarianza) y 34 calibran el umbral, una más que el mínimo de 33 para α = 3 %. El total sube a 100 fotos: 50 que calibran, más 25 buenas y 25 malas de prueba. Esas fotos siguen sin tomarse, así que el cambio no compromete la prueba ciega.

## Lo que no se hace
- Cambiar el criterio del [ADR 0014](0014-n1-v3-varios-montajes.md) después de ver e11. La v4 es un ADR nuevo, con su propio criterio.
- Elegir con el set de prueba de Edge Impulse ni con el set ciego propio.
- Llenar el formulario de MVTec por Camilo. Si se valida contra `metal_nut`, ese paso es suyo.

Volver al [árbol](../arbol.md).
