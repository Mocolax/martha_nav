# Escenarios de entrenamiento: respaldo en la literatura

Justifica las plantillas de `martha_nav/sim2d/scenarios.py` (spec §4.2). Solo se
anota lo que se leyó en cada fuente. Si un rango no aparece en el texto, se dice.

## Resumen por paper

| paper | escenarios | generación procedural | detalle relevante para martha_nav |
|---|---|---|---|
| Long et al. 2018 [1] | 7 escenarios en Stage, entre ellos un pasillo con obstáculos, un círculo de cruce y un escenario aleatorio | sí: el escenario 7 sortea robots, **obstáculos** y metas en cada episodio | LiDAR montado **al frente** del robot, no en el centro; CNN-1D de 32 filtros (k5 s2, luego k3 s2); actor y crítico **sin compartir** parámetros; +15 al llegar y −15 al chocar; **primera etapa sin obstáculos** |
| Arena-Bench / Arena-Rosnav [2] | mapas interiores y exteriores (15 × 15 m en la evaluación), más un almacén | sí: generador aleatorio de Heiden et al., con dificultad creciente por **más y mayores obstáculos** (exteriores) o **pasillos más estrechos** (interiores); modo "random" que cambia los obstáculos en cada episodio | entrenamiento con Gym + Stable-Baselines3 y PPO, la misma pila que martha_nav; generan el gemelo 3D de su laboratorio a partir de un mapa SLAM, como `lab.world` |
| BARN [3] | 300 entornos de 5 × 5 m generados con autómatas celulares, cada uno con inicio y meta | sí | ordena los entornos por métricas de dificultad, desde espacios abiertos hasta pasos donde el robot "se aprieta" entre obstáculos densos |
| DRL-VO [4] | Lobby, Autolab, Cumberland, Square, Freiburg (con peatones) | no (mundos fijos) | la política recibe LiDAR y un **sub-objetivo** de un planificador global: el mismo esquema de planificador local que martha_nav |

## Plantilla → respaldo

| plantilla | equivalente en la literatura | rangos citados |
|---|---|---|
| `corridor` | pasillo con obstáculos de Long et al. [1]; mapas interiores de Arena [2] | no se leyeron anchos numéricos; Arena varía el ancho del pasillo como su eje de dificultad interior |
| `doorway` | habitaciones unidas por pasos en el generador interior de Arena [2] | ancho de paso configurable en Arena, sin valor por defecto en el texto leído |
| `l_turn` | mapas interiores de Arena [2] (pasillos con giros) | — |
| `furniture_walls` | laboratorio y oficinas de Arena [2]; mundos de DRL-VO [4] | — |
| `furniture_center` | escenario aleatorio de Long et al. [1]; mapas exteriores de Arena [2] | Arena sube el número y tamaño de obstáculos para aumentar la dificultad |
| `narrow_passage` | entornos densos de BARN [3] | BARN incluye pasos donde el robot apenas cabe; martha_nav fija el mínimo en 0.85 m para Martha, de 0.41 m de ancho |

## Diferencias con martha_nav

- **Dificultad:** Arena y BARN escalan la dificultad (currículum o ranking). martha_nav mezcla todas las
  fuentes por decisión del usuario, con un plan de respaldo gradual si no converge (spec §4.2).
- **Etapa sin obstáculos:** Long et al. entrenan primero sin obstáculos y luego añaden la
  complejidad. Es la misma idea que la puerta de convergencia (`--preset gate`), pero usada solo como
  verificación, no como etapa de entrenamiento.
- **Memoria:** Long et al. y DRL-VO apilan varios scans. martha_nav usa uno solo (decisión de la spec, §5.1).
- **Rangos numéricos:** en el texto leído no aparecen anchos de pasillo ni densidades concretas. Los rangos
  de martha_nav vienen de la geometría de Martha y del laboratorio. **Propuesta:** si hace falta citar
  valores, leer la documentación de arena-tools y el código del generador de Heiden et al.

## Referencias

1. P. Long, T. Fan, X. Liao, W. Liu, H. Zhang, J. Pan. *Towards Optimally Decentralized Multi-Robot
   Collision Avoidance via Deep Reinforcement Learning.* ICRA 2018. arXiv:1709.10082.
2. L. Kästner et al. *Arena-Bench: A Benchmarking Suite for Obstacle Avoidance Approaches in Highly
   Dynamic Environments.* IEEE RA-L 2022. arXiv:2206.05728. Plataforma: https://github.com/Arena-Rosnav
3. D. Perille, A. Truong, X. Xiao, P. Stone. *Benchmarking Metric Ground Navigation.* SSRR 2020.
   arXiv:2008.13315. Dataset: https://www.cs.utexas.edu/~xiao/BARN/BARN.html
4. Z. Xie, P. Dames. *DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity
   Obstacles.* IEEE T-RO 2023. arXiv:2301.06512.
