# Glosario

| Término | Qué significa aquí |
|---|---|
| **Golilla** | Arandela plana: un anillo de metal con un agujero al centro. |
| **ADR** | *Architecture Decision Record*: una nota corta que dice qué se decidió, entre qué opciones, con qué criterio y con qué evidencia. |
| **Segmentar** | Separar la pieza del fondo: cada píxel queda marcado como "golilla" o "fondo". |
| **IoU** | Cuánto coincide una segmentación con la verdad: área común dividida por el área total de ambas. 1 = idénticas. |
| **Subpíxel** | Ubicar un borde con más precisión que el tamaño de un píxel, interpolando dónde cambia el brillo. |
| **RANSAC** | Ajuste que prueba muchos círculos con 3 puntos al azar y se queda con el que más puntos explica. Así ignora la muesca o el corte. |
| **Rasgo** | Un número que describe la pieza (por ejemplo, cuánto se desvía el contorno de un círculo). |
| **AUC** | Qué tan bien un número separa malas de buenas: 1 = perfecto, 0,5 = al azar. |
| **MAD** | Desviación mediana absoluta: una medida de dispersión que no se deja engañar por valores raros. |
| **Mahalanobis** | Distancia de una pieza al "centro" de las buenas, que tiene en cuenta cómo varían juntos los rasgos. |
| **Hotelling** | Umbral para Mahalanobis que corrige por haber calibrado con pocas piezas (15 buenas). |
| **Dejando una fuera (LOO)** | Se calibra con 20 fotos y se prueba en la que quedó fuera; se repite 21 veces. Mide cómo le iría con fotos nuevas. |
| **Resustitución** | Probar con las mismas fotos con que se calibró. Siempre sale optimista. |
| **Giros exactos (D4)** | Las 8 versiones de una foto girada en 90° y en espejo: no se interpola ningún píxel. |
| **Sintéticas** | Fotos de golillas generadas por computador, con defectos y verdad exacta conocida. |
| **Sim2real** | La diferencia entre lo que funciona en sintéticas y en fotos reales. |
| **Falsa alarma** | Una golilla buena que el sistema manda a retirar. |
| **Encoder** | Sensor en el rodillo de la cinta que cuenta cuánto avanzó, en décimas de milímetro. |
| **Resbalamiento** | La pieza avanza un poco menos que la cinta (vibración, roce). |
| **PatchCore** | Detección de anomalías con una red neuronal: guarda trozos de imágenes buenas y marca lo que no se parece a ninguno. |

Volver al [índice](README.md).
