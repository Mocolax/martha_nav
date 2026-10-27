<p align="center">
  <img src="docs/portada.svg" alt="Martha Nav — navegación autónoma con aprendizaje por refuerzo, ROS 2 y Gazebo" width="100%">
</p>

# Martha Nav

**Del mapa al movimiento: un robot que sigue una ruta y responde a los obstáculos.**

Sistema de navegación para el robot **Martha** que combina planificación
sobre un mapa con una política **PPO** de aprendizaje por refuerzo. El planificador
global calcula la ruta y la política local usa el LiDAR para decidir las velocidades.

Desarrollado como parte del trabajo de tesis sobre el planificador local de Martha.
Incluye simulador 2D, integración con ROS 2 y Gazebo Classic, políticas entrenadas y
componentes de despliegue en robots reales.

## Explora el proyecto

Elige un apartado para consultar las instrucciones, entender el sistema o revisar las evidencias.

| 🚀 Primeros pasos | 🧭 El sistema | 📊 Pruebas y evidencias |
|:---|:---|:---|
| **[Instalación y primera prueba](#empieza-aquí)**<br>Requisitos, puesta en marcha y demostración. | **[Cómo funciona](#cómo-funciona)**<br>Planificación, LiDAR y políticas de navegación. | **[Evidencias del robot real](#evidencias-del-robot-real)**<br>Galería, registros y pruebas de MARTHA. |
| **[Manual de usuario](docs/manual-usuario.md)**<br>Simulación, entrenamiento y uso en robots reales. | **[Trabajo realizado](#trabajo-realizado)**<br>Componentes desarrollados e integración. | **[Resultados](#resultados)**<br>Métricas y condiciones de evaluación. |
| **[Referencia de comandos](docs/comandos.md)**<br>Comandos para operar y evaluar el sistema. | **[Organización del repositorio](#organización-del-repositorio)**<br>Carpetas y archivos principales. | **[Documentación complementaria](#documentación-complementaria)**<br>Bitácoras y material de consulta. |

---

## Empieza aquí

| Quiero… | Dónde empezar |
|---|---|
| Instalar y probar el proyecto | [Requisitos e instalación](docs/manual-usuario.md#requisitos-e-instalación) |
| Ver el robot navegar | [Manual de simulación](docs/manual-usuario.md#manual-de-simulación) |
| Entrenar una política | [Entrenamiento y evaluación](docs/manual-usuario.md#entrenamiento-y-evaluación) |
| Utilizar Martha real | [Uso en el robot real](docs/manual-usuario.md#uso-en-el-robot-real) |
| Resolver un problema | [Pruebas y solución de problemas](docs/manual-usuario.md#pruebas-y-solución-de-problemas) |

### Primera demostración

Con ROS 2 Humble y el workspace **ya instalados y cargados**, desde la raíz del
repositorio:

```bash
ros2 launch martha_nav sim.launch.py \
  world:=lab drive:=planar rviz:=true \
  checkpoint:="$(pwd)/policies/martha.npz"
```

En RViz, selecciona **2D Goal Pose** sobre una zona libre para enviar una meta.
La modalidad `planar` permite probar navegación sin simular la dinámica de ruedas.
Para instalación desde cero, cancelación de metas y otras opciones, sigue el
[manual paso a paso](docs/manual-usuario.md#manual-de-simulación).

[↑ Volver al menú](#explora-el-proyecto)

---

## Cómo funciona

![Diagrama de navegación de Martha: mapa y meta → Dijkstra; ruta y sensores → PPO → guarda direccional → velocidades para Gazebo o el robot real.](docs/como-funciona.svg)

[Ver el diagrama en formato vectorial](docs/como-funciona.svg).

El LiDAR se reduce a **90 sectores** y sus distancias se codifican como `d/(d+1)`.
La observación reúne esos sectores, distancia y ángulo al objetivo intermedio,
velocidad medida y acción anterior: 96 valores con dos acciones o 98 con tres.
El nodo local calcula comandos a **10 Hz** y detiene el movimiento si faltan datos
recientes de los sensores.

Hay dos perfiles en [`martha_nav/robots.py`](martha_nav/robots.py):

| Perfil | Plataforma | Particularidad |
|---|---|---|
| `martha` | Base con ruedas mecanum | Permite acciones `(v, w)` o `(vx, vy, w)`, según el modelo entrenado. |
| `martha_tower` | Martha con torre de aluminio | Modela los postes que atraviesan el plano del LiDAR. Requiere una política compatible con ese perfil. |

Se incluye una política lista para inferencia:

| Archivo | Entrenamiento de origen | Acciones |
|---|---|---|
| [`policies/martha.npz`](policies/martha.npz) | `wide_dyn_s0` | Avance/retroceso y giro; **no ordena movimiento lateral**. |

Los `.npz` ejecutan la inferencia con NumPy, sin PyTorch. Los checkpoints `.zip`
necesitan las dependencias de aprendizaje y su `config.yaml` asociado. No se incluye
una política para `martha_tower`. El cargador comprueba la compatibilidad del perfil;
no basta con cambiar el nombre del robot para reutilizar una política.

[↑ Volver al menú](#explora-el-proyecto)

---

## Trabajo realizado

| Componente | Desarrollo incluido |
|---|---|
| Simulador 2D | Entorno de aprendizaje con cinemática, LiDAR, colisiones, mapas rasterizados desde archivos `.world`, escenarios paramétricos y obstáculos que no aparecen en el mapa global. |
| Planificación global | Rutas mediante Dijkstra sobre una rejilla de ocupación inflada, seguimiento del avance y selección de un punto objetivo intermedio (*carrot* o zanahoria). |
| Aprendizaje | PPO con extractor CNN para el LiDAR, registro de recompensas, evaluación periódica y selección del mejor modelo; variantes con movimiento lateral, memoria LSTM y recompensa geodésica. |
| Transferencia a Gazebo | Modelos de robot, mundos, sensores, control mecanum o movimiento planar, transformaciones TF y evaluación por episodios. |
| Navegación ROS 2 | Nodos de mapa, planificación global, inferencia de la política, publicación de velocidades, cancelación de metas y guarda direccional ante obstáculos. |
| Localización | Integración con `slam_toolbox` para mapear y localizarse; en simulación también se puede utilizar la pose verdadera de Gazebo. |
| Martha real | Firmware ESP32, control de motores, encoders, giroscopio, puente serial, RPLIDAR y fusión de odometría mediante EKF. |
| Análisis experimental | Comparación de configuraciones, transferencia 2D–Gazebo, trayectorias, éxito, colisiones, estancamientos, SPL y bitácoras de resultados. |
| Verificación | Pruebas del simulador, recompensas, políticas, geometría, nodos, launch, URDF e interfaces de hardware. |

[↑ Volver al menú](#explora-el-proyecto)

---

## Organización del repositorio

```text
martha_nav/
├── martha_nav/
│   ├── sim2d/       # Simulador, observaciones, recompensa y planificación
│   ├── learning/    # Entrenamiento, evaluación y exportación
│   ├── ros/         # Nodos e integración ROS 2
│   └── robots.py    # Perfiles de robot
├── launch/          # sim.launch.py y real.launch.py
├── policies/        # Políticas NumPy incluidas
├── config/          # Controladores, EKF, LiDAR y puntos de evaluación
├── urdf/            # Modelo de Martha
├── worlds/          # Escenarios de Gazebo
├── rviz/            # Configuración de visualización
├── firmware/        # Programa de la ESP32 y control de hardware
├── tools/           # Evaluación, mapas, gráficas y despliegue
├── test/            # Pruebas automatizadas
└── docs/            # Manual, comandos y resultados
```

`runs/`, `maps/` y `entrega_tesis/` son salidas locales excluidas de Git. Un clon
nuevo no contiene los checkpoints históricos, mapas guardados ni todos los CSV
mencionados en las bitácoras. `docs/entrega/` describe el paquete de evidencias,
pero no contiene ese paquete completo.

[↑ Volver al menú](#explora-el-proyecto)

---

## Resultados

La [bitácora de resultados](docs/resultados.md) registra el estudio de codificación
LiDAR, recompensa, estabilidad de PPO, CNN frente a MLP, acciones holonómicas,
memoria recurrente, variación de dinámica y localización.

Resultados principales:

| Modelo y condición | Episodios | Éxito |
|---|---:|---:|
| Martha `wide_dyn_s0`, 2D `lab` con obstáculos | 200 | 85,5 % |
| Martha `wide_dyn_s0`, Gazebo `lab`, evaluación `v5` | 100 | 87,0 % |
| Martha `wide_dyn_s0`, Gazebo puntos fijos de `lab`, `v5` | 90 | 91,1 % |

Consultar las condiciones y comparaciones en la sección de reevaluación `v5`
de la bitácora. Las tablas de 200 episodios 2D y 100 de
Gazebo no constituyen por sí solas una comparación pareada del mismo tamaño.

[↑ Volver al menú](#explora-el-proyecto)

---

## Evidencias del robot real

Pruebas de MARTHA en el laboratorio de robótica y en un salón controlado,
registradas el **21 y 22 de octubre de 2026**.

| MARTHA en el laboratorio | Ruta y meta en RViz |
|:---:|:---:|
| [<img src="docs/evidencias/fotos/14_video5_00m25s_martha_fisico.png" alt="MARTHA durante una prueba en el laboratorio" width="420">](docs/evidencias/README.md#galería-visual) | [<img src="docs/evidencias/fotos/19_R1_00m27s_rviz_goal_ruta_robot.png" alt="Robot, ruta y meta en RViz durante el registro R1" width="420">](docs/evidencias/README.md#galería-visual) |

**[Abrir galería de evidencias](docs/evidencias/README.md#galería-visual)** ·
[🎬 Ver los 9 videos](docs/evidencias/README.md#videos) ·
[Resultados de las pruebas físicas](docs/evidencias/README.md#aciertos-y-fallas)

El registro de los autores reporta **10 llegadas a la meta en 12 pruebas**.
Las dos fallas no tienen video y los registros audiovisuales no corresponden
uno a uno con los ensayos. La galería contiene **21 imágenes y 9 videos disponibles**. Estas pruebas documentales complementan la evaluación
en simulación y no constituyen una batería cuantitativa equivalente.

[↑ Volver al menú](#explora-el-proyecto)

---

## Documentación complementaria

| Documento | Contenido |
|---|---|
| [Evidencias del robot real](docs/evidencias/README.md) | Galería, inventario audiovisual y registro de pruebas físicas. |
| [Manual de usuario](docs/manual-usuario.md) | Instalación, simulación, entrenamiento y operación paso a paso. |
| [Comandos detallados](docs/comandos.md) | Operación en el contenedor, evaluación, SLAM y robots reales. |
| [Escenarios y referencias](docs/escenarios-referencias.md) | Escenarios utilizados y contexto bibliográfico. |
| [Resultados](docs/resultados.md) | Bitácora experimental y correcciones posteriores. |
| [Análisis del atasco](docs/resultados-atasco.md) | Ablaciones de memoria, objetivo y recompensa. |
| [Paquete de evidencias](docs/entrega/README.md) | Estructura y métricas de la entrega experimental. |
| [Justificación de recompensa](docs/entrega/recompensa.md) | Criterios usados para sus términos y pesos. |

Las bitácoras conservan decisiones de distintas etapas;
para nombres de parámetros y comportamiento vigente, contrastarlos con los launch,
las interfaces de línea de comandos y los perfiles del código.

---

Documentación de la rama `feat/sim2d-training`. Las cifras corresponden a los
experimentos registrados y deben interpretarse con sus condiciones de evaluación.
