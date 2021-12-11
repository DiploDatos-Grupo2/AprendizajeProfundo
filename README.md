# Aprendizaje Profundo

Repositorio oficial de la materia optativa "Aprendizaje Profundo" (Deep Learning) de la Diplomatura en Ciencias de Datos de la UNC.

Para comenzar a instalar y configurar el entorno de trabajo por favor seguir las instrucciones detalladas en el [Notebook 0](./0_set_up.ipynb).

## Conectados desde nabucodonosor2

Corremos nuestros experimentos
```sh
sbatch slurm.sh run.sh
```

Una vez completados visualizamos los resultados en MLFlow con el puerto que expusimos en la conexion al server

```sh
mlflow ui --port=XXXX
```

Los resultados:

![Alt text](images/mlflow_1.png?raw=true "MLFlow")
