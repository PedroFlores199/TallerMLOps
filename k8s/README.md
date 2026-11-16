# Canary y actualización en Kind

Requiere Docker, Kind, kubectl y el entorno Python del proyecto. Utiliza un cluster local llamado `taller-mlops`, un kubeconfig nuevo y separado, y localhost:5110. No cambia el contexto habitual de kubectl.

```sh
docker build --target serving -t taller-mlops:demo .
uv run python scripts/kind_demo.py --kubeconfig artifacts/kubeconfig-demo
```

La prueba crea el cluster, carga la imagen, despliega ambas versiones y espera su disponibilidad. El router asigna aproximadamente un 10 % de los identificadores al challenger, con reparto estable por hash. El modelo inicial y el reentrenado están fijados por versión: no se confunde una etiqueta cambiante con un modelo concreto.

Se realizan 500 solicitudes y se registran las versiones recibidas. Después, mientras continúan las peticiones, se sustituye el modelo champion mediante una actualización gradual con `maxUnavailable=0`. Las instancias reciben cinco segundos de drenaje antes de detenerse; el router puede reintentar inferencias, que no modifican datos.

El informe incluye distribución, respaldos utilizados, solicitudes durante el cambio y reinicios de contenedores. La medición se inicia después de que los servicios estén disponibles. El primer ensayo detectó un corte durante la sustitución; se corrigieron el drenaje y el reintento. La prueba final guardada tuvo 160 respuestas correctas y ningún error. Esto describe la prueba local concreta, sin garantizar cualquier carga de producción.

El script elimina su cluster al terminar, incluso si falla una comprobación. Para inspeccionarlo, añadir `--keep`. Limpieza posterior: `kind delete cluster --name taller-mlops`, utilizando el kubeconfig separado. No reutilizar un kubeconfig existente con este script.
