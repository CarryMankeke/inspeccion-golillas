# ADR 0018 · P0: alimentación y golillas encimadas

**Estado:** propuesta. Va después de e13 ([ADR 0017](0017-n0-v2-estado-del-arte.md)), porque ningún dataset público trae golillas encimadas · **Fecha:** 2026-11-28 · **Nivel:** N2 (un proceso nuevo antes de P1) · **Código:** por escribir

## Contexto
Toda la cadena supone una golilla sola por foto ([ADR 0001](0001-n0-objetivo.md)). En planta no siempre es así: las golillas son planas y delgadas, así que pueden llegar una sobre otra (encimadas) o pegadas por el aceite. Se advirtió en el ramo de visión artificial de la USACH.

Hoy, una golilla encimada se vería como una pieza de forma rara. La cadena clásica la retiraría como mala, y con eso se perdería una golilla buena, o dos.

## Cómo lo resuelve la industria
Catastro del 28 nov 2026. La madurez es comercial cuando hay productos a la venta. Los TRL son estimaciones propias. Solo se dan ritmos que se leyeron en la página del fabricante.

| Solución | Cómo funciona | ¿Sirve para golillas planas? | Madurez | Ejemplos y ritmo publicado |
|---|---|---|---|---|
| Tazón vibratorio con herramental | Las piezas suben por una pista en espiral. Un barredor o un chorro de aire bota al tazón lo que va montado | Sí: es lo estándar | Comercial (TRL 9) | — |
| Alimentador centrífugo | Un disco giratorio lleva las piezas al borde, donde una guía las pone en fila | Sí: bueno para piezas planas y redondas, si no se sobrecarga | Comercial (TRL 9) | — |
| Disco de vidrio con varias cámaras | Cada pieza pasa sola sobre un disco transparente, y se inspecciona en 360° | Sí, pero necesita un alimentador que ya las separe | Comercial (TRL 9) | [RNA MK360](https://www.rnaautomation.com/products/bespoke-automation/mk-360-glass-disc-vision-inspection/): más de 2.000 piezas por minuto y 100 % de inspección; otra máquina para golillas y tuercas, hasta 600 por minuto ([Fastener+Fixing](https://fastenerandfixing.com/manufacturing-technology/new-optical-sorting-machines-drive-efficiency)) (fabricantes) |
| Alimentador de escalón | Unos escalones suben las piezas, y las mal puestas caen | Poco: dos golillas pegadas pueden subir juntas | Comercial (TRL 9) | — |
| Alimentador flexible con visión | Una plataforma vibra y voltea las piezas, y una cámara elige cuál tomar con un robot | Parcial: desde arriba, dos encimadas se ven como una | Comercial (TRL 9) | Asyril Asycube, Omron AnyFeeder |
| Cintas a distinta velocidad | Una cinta más rápida estira la fila y deja espacio entre piezas | Separa piezas en fila, pero no despega una encimada | Comercial (TRL 9) | — |
| Sensor de pieza doble | Ultrasonido, corrientes de Foucault o láser miden si hay dos espesores | Hecho para chapa; en golillas sueltas no se encontró un caso | En chapa, TRL 9; en golillas, ≈ 4 | Pepperl+Fuchs, Baumer (sensores de doble lámina) |
| Bin-picking con visión 3D | Una cámara 3D elige una pieza del montón y un robot la saca | En teoría sí, pero lento | Comercial (TRL 8); con golillas delgadas y brillantes, ≈ 6 | — |

Lo que se aprende:
- **La industria evita el problema antes de la cámara:** el alimentador separa las piezas y bota las encimadas de vuelta. La cámara recibe piezas sueltas.
- **Las máquinas de disco de vidrio inspeccionan la pieza en 360°**, no solo desde arriba. Una cámara lateral es la forma más simple de sumar esa vista: desde el lado, una golilla encimada mide el doble de espesor.
- **No se encontró una fuente que diga qué hace una clasificadora con una pieza doble.** Devolverla al alimentador en vez de botarla es una decisión de diseño de este proyecto, no un dato de la industria.

## Propuesta
1. **Un alimentador simulado: tazón vibratorio con barredor.** Es lo estándar para golillas planas. Se modela con un parámetro medible, la fracción de encimadas que pasan el barredor, y su ritmo de piezas por minuto.
2. **Una segunda cámara lateral** en la línea simulada ([ADR 0012](0012-p7-linea.md)), que mide el espesor de la pieza. La regla es clásica: una golilla M8 de ISO 7089 tiene de 1,4 a 1,8 mm de espesor, así que dos encimadas miden de 2,8 a 3,6 mm. Las dos franjas no se tocan, y la regla se puede escribir con la tolerancia de la norma. Desde arriba, la cadena clásica ya da una pista: el agujero descentrado o un contorno doble.
3. **Una tercera salida en P7:** las encimadas no se botan; un segundo soplador las devuelve al alimentador. Las malas siguen yendo al rechazo.
4. **Datos sintéticos de encimadas:** el generador ([ADR 0005](0005-datos-sinteticos.md)) suma dos golillas desplazadas en la vista de arriba, y un perfil de doble espesor en la vista lateral.

## Criterio (por fijar antes de correr)
Se fija aquí, en un commit, antes de generar las encimadas de prueba. Usa la misma fuente del ADR 0017 (AIAG MSA, aceptable):
- detectar el 98 % o más de las encimadas;
- marcar como encimadas el 5 % o menos de las golillas sueltas;
- tardar 100 ms o menos por pieza, con las dos cámaras.

Como solo hay datos sintéticos, el resultado se informa como «validado en simulación», no como «probado en planta».

## Lo que no se hace
- Mezclarlo con la v5: MANTA y los sets de Roboflow traen una golilla por foto.
- Afirmar cómo resuelve esto una planta concreta sin una fuente.

Volver al [árbol](../arbol.md).
