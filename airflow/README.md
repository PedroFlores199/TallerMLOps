# Ciclo de vida en Airflow

```sh
docker compose -p taller-airflow -f airflow/compose.yaml up --build -d
```

Abrir http://localhost:5116. Usuario local `demo`; SimpleAuth genera su contraseña en `/opt/airflow/simple_auth_manager_passwords.json.generated`, dentro del contenedor. Consultarla localmente, no publicarla. Este perfil es para desarrollo en localhost.

El DAG `taller_model_lifecycle` está pausado inicialmente. Su configuración permite precio base (`1.0`) o subida del 20 % (`1.2`). Genera las órdenes si no existen, entrena un modelo inicial si falta, detecta deriva y elige la rama de reentrenamiento o la de omisión. Al final registra la versión que superó la validación fija.

Desde la interfaz se puede lanzar el DAG con `{"price_factor":1.2}`. Si permanece en cola porque está pausado, activarlo durante la prueba y volver a pausarlo al terminar; al activarlo puede quedar pendiente una ejecución semanal de su calendario. `catchup=false` evita recuperar todo el histórico, pero no anula el intervalo actual.

Los procesos de ML se ejecutan en un entorno Python separado del de Airflow. Los volúmenes `shared`, `metadata` y `airflow_home` conservan modelos, datos, resultados e historial. El primer recorrido completo local tardó aproximadamente un minuto después de construir la imagen. Las repeticiones pueden rechazar un modelo que ya no mejora; no se fuerza su promoción.

Parar conservando resultados: `docker compose -p taller-airflow -f airflow/compose.yaml stop`. Borrar el entorno de esta demo: `docker compose -p taller-airflow -f airflow/compose.yaml down -v`. La segunda orden elimina sus volúmenes y su historial.

Evidencias: `docs/airflow-task-states.json`, `docs/airflow-market-result.json` y la captura `docs/airflow.jpg`.
