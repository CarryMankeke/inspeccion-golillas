# ADR 0014 · N1 v3: reabrir el enfoque con varios montajes

**Estado:** propuesta, con el criterio fijado antes de correr · **Fecha:** 2026-11-27 · **Nivel:** N1 · **Scripts:** `experimentos/e11_n1_v3.py` (elección) y `experimentos/e12_prueba_ciega_propia.py` (prueba ciega), aún por escribir

## Contexto
En el montaje propio, la visión clásica detecta las 6 malas reales y gana a las redes ([ADR 0004](0004-n1-enfoque.md)). En la prueba ciega con fotos de otro laboratorio (e10, [ADR 0013](0013-datos-publicos.md)) eso no se sostiene: la clásica detecta 31 % de las malas y PatchCore 67 % (AUC 0,75 contra 0,95). La clásica no mide tamaño y no ve cortes radiales finos.

Hay que volver a elegir el enfoque, pero sin hacer trampa: el set de prueba de Edge Impulse ya se vio, así que no puede elegir nada. Por eso esta versión separa desde el principio qué datos eligen y qué datos prueban.

## Qué datos sirven para qué
| Datos | Buenas | Malas | Papel en v3 |
|---|---|---|---|
| 21 fotos propias | 15 | 6 | **Elegir**: prueba A (dejando una fuera) y prueba R (con giros), como en e06 y e09 |
| Sintéticas de validación de e09 (semilla 9) | 300 | 300 (50 de cada uno de los 6 defectos) | **Elegir** |
| Kaggle «Simplified washers» (dibujos) | 1.000 calibran; 100 de prueba | 100 | **Elegir**: es otro montaje con malas y nunca se ha usado. Falta bajarlo (OK de Camilo) |
| Edge Impulse, 273 buenas de entrenamiento | 273 | 0 | **Elegir, solo falsas alarmas**: cada buena se compara con las demás (dejando una fuera) |
| Edge Impulse, 191 de prueba | 88 | 103 | **No elige.** Ya se vio en e10. Se vuelve a correr al final y se informa marcado «no ciego» |
| Roboflow MassMEP | 0 | 187 | No sirve: no tiene buenas ([ADR 0013](0013-datos-publicos.md)) |
| CSEM-MISD | 0 | 50 | No sirve: no tiene buenas |
| **Set ciego nuevo: fotos propias** | 25 (+15 que calibran) | 25 | **Prueba ciega**: se corre una sola vez, al final |

Si Kaggle no se baja, se elige con las otras tres filas de «Elegir» y se dice en el resultado.

## Candidatos (configuraciones fijas, sin búsqueda)
1. **PatchCore**, con el coreset del 1 % de e09. Son 4 variantes:
   - capas layer2+3 (la de e09) o layer1+2+3 (la capa de detalle fino);
   - recorte de 160 px fijo (el de e08) o recorte proporcional al radio exterior medido. En el proporcional, el lado es k × radio, con k tal que en las 15 buenas propias dé los mismos 160 px. Así el montaje propio no cambia y en los otros la golilla ocupa la misma parte del recorte.
2. **Clásica v3**: los 4 rasgos de hoy ([ADR 0010](0010-p5-rasgos.md)) más la **razón de radios** y el **radio exterior**. Ambos ya se calculan en `golillas/rasgos.py`. El radio exterior queda en píxeles: con la cámara fija, calibrar con buenas del mismo montaje hace el papel de los milímetros. Son 2 variantes: con esos 6 rasgos, o con un séptimo para los cortes finos, el **relleno mínimo por sector** (el anillo se divide en 36 sectores de 10° y se toma el sector con menos material).
3. **Fusión**: se retira la pieza si la clásica v3 elegida **o** el PatchCore elegido la retiran, cada uno con su propio umbral. Cuando retira la clásica, la razón se explica con el nombre del rasgo.

La red supervisada queda fuera. Necesita malas del montaje para entrenar, y en e10 ningún umbral sirvió.

**Umbrales:** siempre con piezas buenas del montaje, como las piezas patrón en planta. La clásica usa el umbral de Hotelling y PatchCore el LOO interno de e09. Ninguna mala ajusta un umbral.

## Criterio (fijado antes de correr)
Cada candidato se calibra con las buenas de cada montaje y se mide ahí mismo.

1. **Requisito del montaje propio** (el N0 del [ADR 0001](0001-n0-objetivo.md), más la prueba R): 6/6 malas en la prueba A, ≥ 95 % de malas en la prueba R y < 100 ms por pieza en CPU.
2. **Requisito de falsas alarmas:** ≤ 1/15 en las buenas propias, ≤ 5 % en las 100 buenas de prueba de Kaggle y ≤ 5 % en las 273 buenas de Edge Impulse (dejando una fuera).
3. **Entre los que cumplen 1 y 2, gana la mayor detección media**: el promedio de la detección en las malas de Kaggle y en las sintéticas de validación.
4. **Empate** (≤ 2 puntos de diferencia): gana la más fácil de explicar (clásica, luego fusión, luego PatchCore). Si siguen empatadas, gana la más rápida en CPU.

Si ninguno cumple 1 y 2, gana el de mayor detección media entre los que cumplen la latencia, y se informa que ninguno cumple. Los requisitos no se relajan.

## Set ciego nuevo: 65 fotos propias
- **Montaje** del [ADR 0001](0001-n0-objetivo.md): golilla oscura sobre papel claro, la cámara a la altura de siempre, y la posición y el giro al azar. Todas las fotos se toman el mismo día.
- **15 buenas para calibrar**, como las piezas patrón al empezar un turno. No se cuentan en el resultado.
- **25 buenas y 25 malas de prueba**: 50 golillas distintas, con una foto cada una. Las malas son 5 de cada tipo: cortadas (corte radial), dobladas o deformadas, con muesca en un borde, de otra medida y con un rayón de superficie. Los rayones no cambian la forma: están para medir ese límite, no para que la clásica gane.
- **Por qué 25 y 25.** Si se detectan las 25 malas, el IC95 de la detección va de 86 % a 100 %; hoy, con 6 malas, va de 54 % a 100 %. Si no se retira ninguna de las 25 buenas, el IC95 de las falsas alarmas va de 0 % a 14 %.
- **Sellado.** Camilo etiqueta cada foto al tomarla en un CSV (archivo, clase, defecto). Antes de correr nada, ese CSV y la huella SHA-256 de cada foto se suben en un commit propio. Ningún script lee esas fotos antes de ese commit.
- **Una sola corrida** con lo elegido. Se informan los tres candidatos, como en e10, sea cual sea el resultado.

## Lo que no se hace
- Elegir algo mirando el set de prueba de Edge Impulse, Roboflow o el set ciego nuevo.
- Agregar candidatos o cambiar el criterio después de ver los resultados de e11. Si hace falta, va en un ADR nuevo, y el set ciego se vuelve a tomar.
- Subir fotos de datasets públicos al repo.

## Riesgos
- Los dibujos de Kaggle pueden premiar a la clásica, porque sus bordes son limpios. Por eso no deciden solos: la detección media también promedia las sintéticas.
- Las malas «de otra medida» dependen de que la cámara no se mueva entre la calibración y la prueba. Si se mueve, el radio exterior falla, y eso se informa.

Volver al [árbol](../arbol.md).
