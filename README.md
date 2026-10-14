# Martha Nav — navegación autónoma con PPO y ROS 2

`martha_nav` implementa un sistema de navegación para el robot móvil Martha y el
TurtleBot3 Burger. Combina planificación global sobre un mapa con un planificador
local aprendido mediante **PPO (Proximal Policy Optimization)**: la ruta indica por
dónde llegar y la política usa el LiDAR para ajustar las velocidades ante obstáculos.

El proyecto incluye entrenamiento en un simulador 2D propio, evaluación en Gazebo
Classic, visualización en RViz y componentes para desplegar la navegación en robots
reales. Se desarrolló como parte del trabajo de tesis sobre el planificador local
de Martha.

Este manual describe la rama `feat/sim2d-training`. Los resultados citados pertenecen
a los experimentos registrados; ejecutar hoy el código no garantiza reproducir
exactamente una corrida histórica.

## Contenido

- [Trabajo realizado](#trabajo-realizado)
- [Cómo funciona](#cómo-funciona)
- [Organización del repositorio](#organización-del-repositorio)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Manual de simulación](#manual-de-simulación)
- [Entrenamiento y evaluación](#entrenamiento-y-evaluación)
- [Uso en robots reales](#uso-en-robots-reales)
- [Resultados y limitaciones](#resultados-y-limitaciones)
- [Pruebas y solución de problemas](#pruebas-y-solución-de-problemas)
- [Documentación complementaria](#documentación-complementaria)

## Trabajo realizado

| Componente | Desarrollo incluido |
|---|---|
| Simulador 2D | Entorno de aprendizaje con cinemática, LiDAR, colisiones, mapas rasterizados desde archivos `.world`, escenarios paramétricos y obstáculos que no aparecen en el mapa global. |
| Planificación global | Rutas mediante Dijkstra sobre una rejilla de ocupación inflada, seguimiento del avance y selección de un punto objetivo intermedio (*carrot* o zanahoria). la implementación utiliza Dijkstra. |
| Aprendizaje | PPO con extractor CNN para el LiDAR, registro de recompensas, evaluación periódica y selección del mejor modelo; variantes con movimiento lateral, memoria LSTM y recompensa geodésica. |
| Transferencia a Gazebo | Modelos de robot, mundos, sensores, control mecanum o movimiento planar, transformaciones TF y evaluación por episodios. |
| Navegación ROS 2 | Nodos de mapa, planificación global, inferencia de la política, publicación de velocidades, cancelación de metas y guarda direccional ante obstáculos. |
| Localización | Integración con `slam_toolbox` para mapear y localizarse; en simulación también se puede utilizar la pose verdadera de Gazebo. |
| Martha real | Firmware ESP32, control de motores, encoders, giroscopio, puente serial, RPLIDAR y fusión de odometría mediante EKF. |
| TurtleBot3 Burger | Perfil propio de cuerpo y LiDAR, simulación, exportación de políticas a NumPy y herramientas de despliegue en Raspberry Pi. |
| Análisis experimental | Comparación de configuraciones, transferencia 2D–Gazebo, trayectorias, éxito, colisiones, estancamientos, SPL y bitácoras de resultados. |
| Verificación | Pruebas del simulador, recompensas, políticas, geometría, nodos, launch, URDF e interfaces de hardware. |

La existencia de los componentes de hardware no equivale a una validación experimental
completa del robot real. El alcance de los resultados se detalla más adelante.

## Cómo funciona

```text
Mapa + pose + meta
        │
        ▼
Planificador global (Dijkstra) ──► ruta y objetivo intermedio
                                              │
LiDAR + velocidad medida + acción anterior ────┤
                                              ▼
                                       Política PPO
                                              │
                                   Guarda direccional
                                              │
                                              ▼
                                          /cmd_vel
                                              │
                                  Gazebo o robot real
```

El LiDAR se reduce a **90 sectores** y sus distancias se codifican como `d/(d+1)`.
La observación reúne esos sectores, distancia y ángulo al objetivo intermedio,
velocidad medida y acción anterior: 96 valores con dos acciones o 98 con tres.
El nodo local calcula comandos a **10 Hz** y detiene el movimiento si faltan datos
recientes de los sensores.

Hay tres perfiles en [`martha_nav/robots.py`](martha_nav/robots.py):

| Perfil | Plataforma | Particularidad |
|---|---|---|
| `martha` | Base con ruedas mecanum | Permite acciones `(v, w)` o `(vx, vy, w)`, según el modelo entrenado. |
| `martha_tower` | Martha con torre de aluminio | Modela los postes que atraviesan el plano del LiDAR. Requiere una política compatible con ese perfil. |
| `burger` | TurtleBot3 Burger diferencial | Perfil configurado para LiDAR LDS-01 de 3,5 m y acciones `(v, w)`. |

Se incluyen dos políticas listas para inferencia:

| Archivo | Entrenamiento de origen | Acciones |
|---|---|---|
| [`policies/martha.npz`](policies/martha.npz) | `wide_dyn_s0` | Avance/retroceso y giro; **no ordena movimiento lateral**. |
| [`policies/burger.npz`](policies/burger.npz) | `burger_s1` | Avance/retroceso y giro. |

Los `.npz` ejecutan la inferencia con NumPy, sin PyTorch. Los checkpoints `.zip`
necesitan las dependencias de aprendizaje y su `config.yaml` asociado. No se incluye
una política para `martha_tower`. El cargador comprueba la compatibilidad del perfil;
no basta con cambiar el nombre del robot para reutilizar una política.

## Organización del repositorio

```text
martha_nav/
├── martha_nav/
│   ├── sim2d/       # Simulador, observaciones, recompensa y planificación
│   ├── learning/    # Entrenamiento, evaluación y exportación
│   ├── ros/         # Nodos e integración ROS 2
│   └── robots.py    # Perfiles de robot
├── launch/          # sim.launch.py, real.launch.py y burger.launch.py
├── policies/        # Políticas NumPy incluidas
├── config/          # Controladores, EKF, LiDAR y puntos de evaluación
├── urdf/            # Modelos de Martha y Burger
├── worlds/          # Escenarios de Gazebo
├── rviz/            # Configuración de visualización
├── firmware/        # Programa de la ESP32 y control de hardware
├── tools/           # Evaluación, mapas, gráficas y despliegue
├── test/            # Pruebas automatizadas
└── docs/            # Diseño, comandos y resultados
```

`runs/`, `maps/` y `entrega_tesis/` son salidas locales excluidas de Git. Un clon
nuevo no contiene los checkpoints históricos, mapas guardados ni todos los CSV
mencionados en las bitácoras. `docs/entrega/` describe el paquete de evidencias,
pero no contiene ese paquete completo.

## Requisitos e instalación

### Entorno ROS 2

El entorno documentado del proyecto utiliza **ROS 2 Humble** y **Gazebo Classic**.
Para una instalación nativa, la base habitual es Ubuntu 22.04 con ROS 2 Humble ya
instalado. Se necesitan `colcon`, `rosdep`, NumPy, SciPy, PyYAML y las dependencias
de [`package.xml`](package.xml). RViz requiere además `rviz2` y una sesión gráfica.

Los siguientes comandos preparan un workspace nativo. No crean el contenedor del
autor ni instalan ROS desde cero:

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone --branch feat/sim2d-training https://github.com/Mocolax/martha_nav.git
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src/martha_nav --ignore-src -r -y
colcon build --packages-select martha_nav --symlink-install
source install/setup.bash
cd src/martha_nav
```

`rosdep` debe estar inicializado previamente; si indica lo contrario, ejecutar una
vez `sudo rosdep init` y repetir `rosdep update`. Revisar que la resolución de
dependencias termine correctamente antes de compilar. La simulación mecanum exige
`mecanum_drive_controller` compatible con Humble, además de `gazebo_ros2_control`;
este repositorio no incluye el código de ese controlador. Si no está disponible,
puede utilizarse `drive:=planar` para la demostración sin dinámica de ruedas.

En **cada terminal nueva** cargar ROS y el workspace:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/martha_nav
```

### Dependencias de entrenamiento

La navegación con las políticas `.npz` incluidas no necesita instalar el conjunto
de aprendizaje. Para entrenar, evaluar checkpoints `.zip` o exportarlos, instalar
[`requirements.txt`](requirements.txt) en un entorno compatible con ROS. Por ejemplo,
desde la raíz del repositorio, con el soporte `python3-venv` instalado:

```bash
python3 -m venv --system-site-packages ~/venvs/martha-nav
source ~/venvs/martha-nav/bin/activate
python3 -m pip install -r requirements.txt scipy pyyaml pytest
```

Las versiones declaradas incluyen Stable-Baselines3 2.3.2 y sb3-contrib 2.3.0,
con NumPy anterior a 2. PyTorch y Gymnasium se instalan como dependencias del
aprendizaje. Los ejemplos 2D de este manual usan `python3 -m` desde la raíz del
repositorio para seleccionar el intérprete del entorno activado. No es necesario
tener Gazebo abierto para entrenar.

### Si ya existe el contenedor del proyecto

Las notas originales usan un contenedor llamado `ros2_humble`, con el repositorio
en `/home/ros/ros2_ws/src/martha_nav`. Los scripts `tools/ct` y `tools/ct_ros`
dependen de ese nombre y esa ruta; **no crean el contenedor**. El Dockerfile citado
en `requirements.txt` está fuera de este repositorio.

```bash
docker start ros2_humble
./tools/ct_ros ros2 pkg prefix martha_nav
```

`tools/ct` ejecuta comandos dentro del contenedor y `tools/ct_ros` carga además ROS
y el workspace. En ese entorno, anteponer `./tools/ct_ros` a los comandos ROS y
usar rutas visibles dentro del contenedor para modelos y mapas.

## Manual de simulación

### 1. Iniciar Martha con la política incluida

Desde la raíz del repositorio, con ROS y el workspace cargados:

```bash
ros2 launch martha_nav sim.launch.py \
  world:=lab drive:=planar rviz:=true \
  checkpoint:="$(pwd)/policies/martha.npz"
```

Debe abrirse Gazebo y RViz; el planificador local debe registrar `policy loaded`.
`planar` permite probar navegación sin simular la dinámica de las ruedas. Para
evaluar el accionamiento mecanum, cambiar a `drive:=mecanum` con sus controladores
instalados. Las métricas históricas dependen de la configuración usada en cada prueba.

### 2. Indicar una meta

En RViz, usar **2D Goal Pose** sobre una zona libre del mapa. Como alternativa,
en otra terminal preparada:

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 2.0, y: 2.0}, orientation: {w: 1.0}}}"
ros2 topic echo /nav_status
```

Las coordenadas son un ejemplo: elegir una posición alcanzable en el mundo activo.
RViz permite ver mapa, robot, LiDAR, ruta y objetivo intermedio.

### 3. Cancelar y detener

```bash
ros2 topic pub --once /cancel_goal std_msgs/msg/Empty '{}'
```

El estado debe pasar a `idle`. Para cerrar la simulación, usar `Ctrl+C` en la
terminal del launch.

### 4. Cambiar de escenario o de robot

Los mundos disponibles son `lab`, `room`, `hall`, `multi`, `tube`, `four_rooms` y
`roblab`. Para simular Burger, cerrar primero la simulación anterior y ejecutar:

```bash
ros2 launch martha_nav sim.launch.py \
  world:=lab robot:=burger rviz:=true \
  checkpoint:="$(pwd)/policies/burger.npz"
```

| Argumento de `sim.launch.py` | Uso |
|---|---|
| `checkpoint` | Ruta absoluta al `.npz` o `.zip`; se necesita para la navegación. |
| `world` | Nombre del mundo, sin extensión; por defecto `lab`. |
| `robot` | Si se omite, se toma del checkpoint. Debe coincidir con él. |
| `drive` | `mecanum` por defecto o `planar`; Burger utiliza su accionamiento diferencial. |
| `gui`, `rviz` | Activar la ventana de Gazebo y RViz; valores por defecto `true` y `false`. |
| `x`, `y` | Posición inicial del robot. |
| `slam` | `off`, `mapping` o `localization`; por defecto `off`. |
| `slam_map` | Prefijo del mapa serializado para localizarse, sin extensión. |
| `sim_speed_factor`, `physics_step_size` | Ajustes de la física y velocidad de simulación. |
| `action_delay` | Retardo del control en períodos; permite estudiar transferencia. |

Con `slam:=off` se usa la pose verdadera de Gazebo. Las pruebas con localización
estimada requieren generar un mapa y lanzar `slam:=localization`; el procedimiento
completo está en [comandos, sección 6.2](docs/comandos.md#62-prueba-de-localización-con-slam_toolbox).

## Entrenamiento y evaluación

### 1. Comprobar que el aprendizaje converge

Con el entorno Python de aprendizaje activado y desde la raíz del repositorio:

```bash
python3 -m martha_nav.learning.train --preset gate --seed 0 --name gate_check
```

`gate` entrena 500 000 pasos en una sala abierta sin obstáculos. La comprobación
propuesta en el proyecto es alcanzar al menos 80 % de éxito. No representa una
evaluación completa de evasión de obstáculos.

### 2. Entrenar con escenarios variados

```bash
python3 -m martha_nav.learning.train \
  --preset full --seed 0 --robot martha --wide-dynamics --name martha_manual_s0
```

`full` utiliza 5 millones de pasos y mezcla escenarios y obstáculos. `lab` queda
reservado para evaluar generalización. Este ejemplo crea un entrenamiento nuevo;
no reconstruye por sí solo el checkpoint histórico `wide_dyn_s0`.

| Opción | Efecto |
|---|---|
| `--steps N` | Cambia el presupuesto de pasos. |
| `--n-envs N` | Cambia los entornos paralelos; por defecto 16. |
| `--action-dim 3` | Añade velocidad lateral para Martha; por defecto se usan dos acciones. |
| `--robot burger` | Entrena para el perfil Burger. |
| `--robot martha_tower` | Entrena considerando la torre y sus retornos LiDAR. |
| `--wide-dynamics` | Amplía la variación de la dinámica durante el entrenamiento. |
| `--recurrent --device cpu` | Variante con memoria LSTM. |
| `--target goal` | Usa la meta final en la observación en lugar del objetivo intermedio. |
| `--reward-progress geodesic` | Variante de recompensa por progreso geodésico. |

La recompensa predeterminada suma progreso sobre la ruta, **+20 por alcanzar la
meta**, **−20 por colisión** y **−0,005 por paso**. Su implementación está en
[`reward.py`](martha_nav/sim2d/reward.py).

Cada ejecución guarda en `runs/<nombre>/` su configuración, `episodes.csv`,
`evals.csv`, `best_model.zip`, `last_model.zip`, `vecnormalize.pkl` y registros de
TensorBoard. Conservar el `config.yaml` junto al checkpoint.

### 3. Evaluar el checkpoint en 2D

```bash
python3 -m martha_nav.learning.evaluate \
  --model runs/martha_manual_s0/best_model.zip \
  --episodes 500 --condition obstacles

python3 -m martha_nav.learning.evaluate \
  --model runs/martha_manual_s0/best_model.zip \
  --episodes 200 --condition obstacles --sources lab

python3 -m martha_nav.learning.evaluate \
  --model runs/martha_manual_s0/best_model.zip \
  --condition obstacles --points lab
```

También se puede usar `--condition clean`. La evaluación con puntos de `lab`
utiliza los 90 pares de inicio y meta de `config/training_points.yaml`. Estos
comandos esperan un checkpoint de entrenamiento `.zip`, no los `.npz` incluidos.

### 4. Exportar para navegar sin PyTorch

```bash
python3 -m martha_nav.learning.export \
  --model runs/martha_manual_s0/best_model.zip
```

Genera `runs/martha_manual_s0/policy.npz`, que puede pasarse como `checkpoint`
al launch. **La exportación actual no admite políticas LSTM.**

### 5. Evaluar en Gazebo

Iniciar `sim.launch.py` con la política que se desea evaluar. Para comparar con las
corridas mecanum, usar `drive:=mecanum`. En otra terminal con ROS cargado:

```bash
mkdir -p runs/evaluacion_manual
ros2 run martha_nav evaluate_gazebo --ros-args \
  -p robot:=martha -p episodes:=100 -p mode:=seeds -p condition:=obstacles \
  -p out:="$(pwd)/runs/evaluacion_manual/gazebo_lab.csv"
```

El evaluador mueve el robot y genera episodios: se utiliza en **simulación**.
Para puntos fijos, cambiar a `-p mode:=points`. Con Burger, cambiar a
`-p robot:=burger` y lanzar su política correspondiente. No mezclar perfiles,
mundos ni condiciones al comparar resultados.

Las métricas incluyen éxito, colisión, estancamiento, timeout, SPL, tiempo y
distancia. La evaluación pareada compara los mismos inicios, metas y obstáculos
en 2D y Gazebo. Los scripts `tools/evaluate_run.sh` y
`tools/evaluate_run_gazebo.sh` automatizan baterías usando el contenedor original.

Para generar gráficas de un entrenamiento:

```bash
python3 tools/plot_report.py runs/martha_manual_s0
tensorboard --logdir runs
```

## Uso en robots reales

### Martha: mapeo y navegación

Antes de operar el hardware, seguir la puesta en marcha y calibración de
[comandos, sección 9](docs/comandos.md#9-robot-real). Allí se describen cableado,
firmware, sentidos de motores y encoders, protecciones y comprobación de TF.
Identificar los puertos con `ls -l /dev/serial/by-id/`; usar `by-path` si los
adaptadores comparten identificador. Los valores `REEMPLAZAR_...` deben sustituirse.

Mapear el lugar:

```bash
ros2 launch martha_nav real.launch.py rviz:=true \
  esp32_port:=/dev/serial/by-id/REEMPLAZAR_ESP32 \
  lidar_port:=/dev/serial/by-id/REEMPLAZAR_RPLIDAR
```

En otra terminal, conducir con `ros2 run teleop_twist_keyboard teleop_twist_keyboard`.
Mostrar `/slam_map` en RViz. Con el mapeo todavía activo, guardar desde la raíz del
repositorio:

```bash
./tools/save_map.sh "$(pwd)/maps/sala_real"
```

Se crean `.pgm` y `.yaml` para el mapa de ocupación, y `.posegraph` y `.data` para
la localización. Cerrar mapeo y teleoperación, colocar el robot en la pose de inicio
del mapa y lanzar navegación:

```bash
ros2 launch martha_nav real.launch.py rviz:=true \
  map:="$(pwd)/maps/sala_real" \
  checkpoint:="$(pwd)/policies/martha.npz" \
  esp32_port:=/dev/serial/by-id/REEMPLAZAR_ESP32 \
  lidar_port:=/dev/serial/by-id/REEMPLAZAR_RPLIDAR
```

`map` es el prefijo **sin extensión**. Corregir la pose con **2D Pose Estimate** y
enviar metas con **2D Goal Pose**. La política incluida corresponde al perfil
`martha` sin postes en su observación de entrenamiento; evaluar esa diferencia
antes de usarla con la torre. Para el perfil `martha_tower`, entrenar y exportar
su propia política.

Cancelar con `/cancel_goal` y cerrar el launch con `Ctrl+C`. El paro físico se
realiza con el interruptor de alimentación. Después de una protección enclavada,
resolver su causa y cancelar la meta antes de solicitar el rearme:

```bash
ros2 topic pub --once /cancel_goal std_msgs/msg/Empty '{}'
ros2 service call /esp32_bridge/reset std_srvs/srv/Trigger
```

### TurtleBot3 Burger

El bringup oficial del TurtleBot corre aparte en la Raspberry Pi. Robot y PC deben
compartir `ROS_DOMAIN_ID`; las instrucciones del proyecto usan `30`. Consultar
[comandos, sección 10](docs/comandos.md#10-turtlebot3-burger) para despliegue,
dependencias de la Pi, RViz remoto y medición de latencia.

En terminales separadas de la Pi, después de cargar ROS y el workspace:

```bash
# Terminal 1: hardware del TurtleBot, mantener activo
ros2 launch turtlebot3_bringup robot.launch.py

# Terminal 2: mapeo; conducir con teleop en otra terminal
ros2 launch martha_nav burger.launch.py
```

Guardar con `tools/save_map.sh`, cerrar el mapeo y la teleoperación, y navegar con
las rutas del workspace de la Pi:

```bash
ros2 launch martha_nav burger.launch.py \
  map:="$HOME/martha_ws/src/martha_nav/maps/sala" \
  checkpoint:="$HOME/martha_ws/src/martha_nav/policies/burger.npz" \
  speed_scale:=0.5
```

El launch real usa inflado de **0,15 m** y tolerancia de meta de **0,20 m**, mientras
que el perfil de evaluación usa inflado de **0,40 m** y tolerancia de **0,30 m**.
No atribuir los resultados de simulación a esa configuración real. La guarda
permanece activa por defecto (`guard:=true`).

## Resultados y limitaciones

La [bitácora de resultados](docs/resultados.md) registra el estudio de codificación
LiDAR, recompensa, estabilidad de PPO, CNN frente a MLP, acciones holonómicas,
memoria recurrente, variación de dinámica, localización y soporte Burger.

Como referencia, estos valores son **resultados históricos documentados**, no
mediciones obtenidas al preparar este README:

| Modelo y condición | Episodios | Éxito |
|---|---:|---:|
| Martha `wide_dyn_s0`, 2D `lab` con obstáculos | 200 | 85,5 % |
| Martha `wide_dyn_s0`, Gazebo `lab`, evaluación `v5` | 100 | 87,0 % |
| Martha `wide_dyn_s0`, Gazebo puntos fijos de `lab`, `v5` | 90 | 91,1 % |
| Burger `burger_s1`, 2D `lab`, inflado 0,40 m | 200 | 88,0 % |
| Burger `burger_s1`, Gazebo `lab`, inflado 0,40 m | 100 | 82,0 % |
| Burger `burger_s1`, Gazebo puntos fijos de `lab`, inflado 0,40 m | 90 | 83,3 % |

Consultar las condiciones y comparaciones en las secciones de reevaluación `v5`
y Burger del 2026-10-07 de la bitácora. Las tablas de 200 episodios 2D y 100 de
Gazebo no constituyen por sí solas una comparación pareada del mismo tamaño.

Las limitaciones registradas incluyen:

- Variación por semilla de entrenamiento; varias configuraciones solo tienen una
  ejecución y las diferencias pequeñas no demuestran superioridad.
- Solapamiento parcial entre semillas de selección del mejor modelo y evaluación
  en los mapas de entrenamiento. `lab` se reservó para generalización.
- Gazebo no determinista: el mismo modelo puede dar resultados distintos al repetir.
- Cambios históricos en reglas de evaluación, perfiles y guarda. Las reevaluaciones
  sustituyen algunas cifras de las primeras bitácoras.
- Checkpoints, CSV y mapas históricos ausentes del clon; regenerar el paquete de
  evidencias requiere disponer de los datos de origen en `runs/`.
- No se aporta aquí una batería cuantitativa equivalente de navegación física de
  Martha. La implementación y las instrucciones de despliegue no sustituyen esa
  validación.

Más detalle en [limitaciones de los datos](docs/limitaciones-datos-tesis.md).
Ese documento recoge el estado de una entrega anterior; para cambios posteriores,
consultar también la bitácora y el código actual.

## Pruebas y solución de problemas

Pruebas completas en el entorno ROS con las dependencias de aprendizaje instaladas:

```bash
python3 -m pytest test -q
```

Un subconjunto de simulación y aprendizaje que no necesita ROS:

```bash
python3 -m pytest test/test_env.py test/test_reward.py test/test_scenarios.py -q
```

En el contenedor original, usar `./tools/ct_ros python3 -m pytest test -q`.

| Problema | Qué revisar |
|---|---|
| `Package 'martha_nav' not found` | Compilar y cargar `install/setup.bash` en esa terminal. |
| `parameter "checkpoint" is required` | Proporcionar una ruta absoluta a un modelo existente. |
| Error de perfil incompatible | Usar la política del robot correcto; cambios de cuerpo o LiDAR requieren reentrenar y reexportar. El inflado es una excepción. |
| Falta `mecanum_drive_controller` | Preparar el controlador compatible; para una demo planar, utilizar `drive:=planar`. |
| El robot permanece detenido | Revisar `/nav_status`, existencia de ruta, TF, mensajes del planificador y recepción reciente de LiDAR y odometría; comprobar la guarda y los eventos de la ESP32. |
| No aparece el LiDAR en RViz real | Remapear el tópico del simulador `/gazebo_ros_lidar/out` a `/scan`, como indica el manual del Burger. |
| No se encuentra un mapa | Crear o copiar sus cuatro archivos; `map:=` recibe el prefijo sin extensión. |
| `tools/ct_ros` no encuentra el contenedor | El script supone `ros2_humble`; usar el entorno documentado o ejecutar ROS directamente en una instalación nativa. |
| No se puede evaluar un modelo antiguo | Algunas configuraciones retiradas, como LiDAR lineal o señal de atasco, se rechazan en el código actual. Consultar el historial y `docs/comandos.md`. |

Después de cambiar `setup.py`, launch, URDF, configuración, RViz o mundos, recompilar
el paquete y volver a cargar el workspace.

## Documentación complementaria

| Documento | Contenido |
|---|---|
| [Comandos detallados](docs/comandos.md) | Operación en el contenedor, evaluación, SLAM y robots reales. |
| [Diseño del planificador](docs/ppo-local-planner-design.md) | Planteamiento y decisiones de arquitectura. |
| [Escenarios y referencias](docs/escenarios-referencias.md) | Escenarios utilizados y contexto bibliográfico. |
| [Resultados](docs/resultados.md) | Bitácora experimental y correcciones posteriores. |
| [Experimentos holonómicos y recurrentes](docs/resultados-noche.md) | Variantes de política. |
| [Análisis del atasco](docs/resultados-atasco.md) | Ablaciones de memoria, objetivo y recompensa. |
| [Diseño del robot real](docs/robot-real-design.md) | Integración de hardware y software. |
| [Paquete de evidencias](docs/entrega/README.md) | Estructura y métricas de la entrega experimental. |
| [Justificación de recompensa](docs/entrega/recompensa.md) | Criterios usados para sus términos y pesos. |

Los documentos de diseño y las bitácoras conservan decisiones de distintas etapas;
para nombres de parámetros y comportamiento vigente, contrastarlos con los launch,
las interfaces de línea de comandos y los perfiles del código.
