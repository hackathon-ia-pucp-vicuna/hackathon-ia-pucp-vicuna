# Reparto de experimentos del modelo

Meta: entrar al top 5 del público, que pide unos +0.004 de validación sobre la mezcla de 10 (A 0.48626, B 0.48677). Congelamiento del modelo: 2 dic.

Cada integrante corre en **su** cuenta de Kaggle y con **su** cuota de GPU. Ningún experimento se repite entre cuentas. Johan (johancg04) revisa los PRs y es el único que sube a Kaggle; no entrena salvo que lo pida.

Reglas comunes: base r2 (ronda 2, dropout 0.1, celdas de `vicuna-t21-ronda3`), entrada solo con horas t-2, t-1 y t del GFS (`futuro = 0`), un cambio a la vez, regla de aceptación escrita antes de correr. Cribado: corte A con 3 semillas, pareado con r2; si la media sube >= 0.002 se completa B con 5 semillas. Aceptación: media >= 0.003 sobre r2 y mejora en A y en B. Miembro de ensamble: la mezcla mejora >= 0.001 en A y en B. Fila en `experimentos.md` siempre.

Reparto propuesto (26 nov), pendiente de ajuste con el equipo:

| Experimento | Responsable | Cuenta de Kaggle | Estado | Versión del kernel | Resultado |
|---|---|---|---|---|---|
| F1 horas futuras (t+1, t+2) | Edu | eduquispeperez | descartado por regla | exp-f1 v1 (QUICK CPU); v2 y v3 cancelados sin correr | sin métrica; no se usa |
| V1 variables derivadas del viento | Edu | eduquispeperez | descartado | exp-v1 v2 (GPU) | A −0.0021, B −0.0038 frente a r2 |
| Q1 regularización: dropout 0.2 y 0.3, ruido gaussiano en la entrada (sigma 0.05 y 0.1) | Edu | eduquispeperez | plan en revisión | exp-q1 (por crear) | |
| Q2 supervisión auxiliar de las horas t-2 y t-1 | Edu | eduquispeperez | plan en revisión | exp-q2 (por crear) | |
| Q3 pesos de muestra en la pérdida (horas evaluadas; mayo y junio) | por confirmar (propuesto: lalocura) | la suya | notebook e instrucciones los prepara Edu | | |
| Q4 U-Net u otra familia (E3) | por confirmar (propuesto: milerrodriguez) | la suya | notebook e instrucciones los prepara Edu | | |
| Q5 combinación por variable y especialistas de viento | Edu, coordinando con lalo-dev8 (dueño de T21) | eduquispeperez | 29 y 30 nov, con las predicciones de validación de lo aceptado | | |
| Q6 ensamble final y FULL (T24) | Johan (T24) con los pesos de cada dueño | la de cada dueño | 1 dic | | |
| Subida a Kaggle de los candidatos | Johan | johancg04 | recibe PRs | | máximo 2 envíos por día, con aprobación de Edu |
