# Reparto de experimentos del modelo

Meta: entrar al top 5 del público, que pide unos +0.004 de validación sobre la referencia vigente. Congelamiento del modelo: 2 dic (T47).

**Referencia vigente:** mezcla de 10 modelos (ronda 1 + ronda 2, envío 58945688, público 0.49702), validación A 0.48626 y B 0.48677. Se actualiza aquí cuando Johan reporte su ensamble de 10+10.

## Cómo se trabaja
- Cada integrante corre en **su** cuenta de Kaggle y con **su** cuota de GPU. Ningún experimento se repite entre cuentas.
- Johan (johancg04) revisa los PRs y es el único que hace envíos a Kaggle.
- Entrada solo con las horas t-2, t-1 y t del GFS (`futuro = 0`) y solo la data oficial.
- Un cambio a la vez sobre r2 (ronda 2, dropout 0.1). La regla de aceptación se escribe antes de correr.
- Cribado: corte A con 3 semillas, pareado con r2; si la media sube >= 0.002, se completa B con 5 semillas. Se acepta si la media sube >= 0.003 sobre r2 y mejora en A y en B. Un miembro de ensamble se acepta si la mezcla mejora >= 0.001 en A y en B.
- Toda corrida deja su fila en `experimentos.md`, también si no funciona.

## Etiquetas del plan
| Etiqueta | Qué prueba |
|---|---|
| Q1 regularización | dropout 0.2 y 0.3, dropout espacial, ruido gaussiano en la entrada |
| Q2 supervisión auxiliar | la red también predice el WRF de t-2 y t-1; en inferencia solo se usa t |
| Q3 pesos de muestra | más peso en la pérdida a las horas que evalúa Kaggle (00, 03, ..., 21 UTC) y a mayo y junio |
| Q4 diversidad | U-Net u otra familia de red (E3) |
| Q5 combinación | pesos por variable con validación cruzada entre los cortes A y B, y especialistas de viento |
| Q6 final | ensamble final y reentrenamiento con todo el train (T24) |

## Reparto (26 nov)
| Experimento | Responsable | Cuenta de Kaggle | Estado | Versión | Resultado |
|---|---|---|---|---|---|
| F1 horas futuras (t+1, t+2) | Edu | eduquispeperez | descartado por regla | exp-f1 v1 (QUICK en CPU); v2 y v3 cancelados sin correr | sin métrica; no se usa |
| V1 variables derivadas del viento | Edu | eduquispeperez | descartado | exp-v1 v2 | A −0.0021, B −0.0038 frente a r2 |
| Ronda 3: r3 = r2 + weight decay 0.05 como tercer grupo del ensamble (mezcla de 15) | Johan | johancg04 | descartado | vicuna-t21-ronda3 v1 (#30) | mezcla de 15 frente a la de 10: A −0.00038, B −0.00022; no pasa la regla |
| Ensamble de 10+10 modelos | Johan | johancg04 | corriendo | su kernel | pendiente |
| Q1 regularización: dropout espacial 0.2 y 0.3, ruido gaussiano sigma 0.05 y 0.1 | Edu | eduquispeperez | corriendo en GPU T4 x2 desde el 26 nov 19:39 | exp-q1 v3 (scriptVersionId 374606881); v2 cancelada tras 5 h en cola | pendiente |
| Q2 supervisión auxiliar | Edu | eduquispeperez | en cola desde el 26 nov 19:41 | exp-q2 v3 (scriptVersionId 374607386); v2 cancelada tras 5 h en cola | pendiente |
| Q3 pesos de muestra | por asignar (lalo-dev8 o maxrodev) | la suya | sin empezar | | |
| Q4 diversidad (U-Net, E3) | por asignar (lalo-dev8 o maxrodev) | la suya | sin empezar | | |
| Q5 combinación por variable | por asignar | | 29 y 30 nov, con las predicciones de validación de lo aceptado | | |
| Q6 ensamble final y FULL (T24) | Johan (T24) | la de cada dueño de los pesos | 1 dic | | |

**GPU de Kaggle (26 nov):** Q1 v2 y Q2 v2 pasaron 5 horas en cola sin que Kaggle diera un motivo, con el cupo casi sin usar (0:29 de 30 h). Se cancelaron y se relanzaron desde la web. Q2 v3 quedó en cola mientras Q1 v3 corre: es un indicio de que la cuenta tiene un límite de una sesión de GPU a la vez. Por eso conviene lanzar de a un experimento por cuenta. La cuenta solo ofrece GPU T4 x2 y TPU v5e-8; no hay P100.

Dropout más weight decay ya se probó como r3 (ronda 3): la mezcla de 15 bajó −0.0003 frente a la de 10, así que en Q1 no se repite.
