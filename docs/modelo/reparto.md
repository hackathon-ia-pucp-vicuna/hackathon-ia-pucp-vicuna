# Reparto de experimentos del modelo

Cada integrante corre sus experimentos en **su** cuenta de Kaggle y con **su** cuota de GPU. Ningún experimento se repite entre cuentas. Johan (johancg04) solo recibe PRs y sube a Kaggle; no entrena salvo que lo pida.

Reglas comunes: base r2 (ronda 2, dropout 0.1, código de `vicuna-t21-ronda3`), entrada solo con horas t-2, t-1 y t del GFS (`futuro = 0`), cortes A y B, 5 semillas con `solas` y ensamble de 5, regla de aceptación escrita antes de correr, fila en `experimentos.md` siempre. Congelamiento del modelo: 2 dic.

Reparto propuesto (26 nov), pendiente de ajuste con el equipo:

| Experimento | Responsable | Cuenta de Kaggle | Estado | Versión del kernel | Resultado |
|---|---|---|---|---|---|
| F1 horas futuras (t+1, t+2) | Edu | eduquispeperez | descartado por regla | exp-f1 v1 (QUICK CPU), v2 y v3 cancelados sin correr | sin métrica; no se usa |
| V1 variables derivadas del viento (velocidad, dirección, divergencia, vorticidad) | Edu | eduquispeperez | plan en revisión | exp-v1 (por crear) | pendiente |
| EMA de pesos sobre r2 | Edu | eduquispeperez | en cola, después de V1 | exp-ema (por crear) | pendiente |
| E3 U-Net (arquitectura distinta) | por confirmar (propuesto: milerrodriguez) | la suya | sin empezar | | |
| E4 bloque estilo ConvNeXt y variantes de capacidad o dropout no probadas | por confirmar (propuesto: lalocura) | la suya | sin empezar | | |
| E6 ensamble de familias distintas | coordinar con lalo-dev8 (dueño de T21) | | espera componentes aceptados | | |
| FULL: reentrenar lo aceptado con todo el train | quien sea dueño de cada componente | la suya | antes del 2 dic | | |
| Subida a Kaggle de los candidatos | Johan | johancg04 | recibe PRs | | máximo 2 envíos por día, con aprobación de Edu |
