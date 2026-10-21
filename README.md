<div align="center">

# Clasificación geoespacial de predios

**Análisis de 18.299 polígonos con Python, regresión logística y herramientas SIG**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![GeoPandas](https://img.shields.io/badge/GeoPandas-an%C3%A1lisis%20espacial-139C5A)
![scikit-learn](https://img.shields.io/badge/scikit--learn-clasificaci%C3%B3n-F7931E?logo=scikitlearn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-interpretabilidad-4051B5)
![Estado](https://img.shields.io/badge/estado-prototipo%20exploratorio-orange)

[Explorar el notebook](clasificacion_predios.ipynb) · [Resultados](#resultados-guardados) · [Reproducibilidad](#reproducibilidad)

</div>

## En 30 segundos

Este proyecto explora cómo usar atributos geométricos para clasificar polígonos asociados a predios y apoyar su revisión frente a información catastral. El notebook conecta análisis exploratorio, tratamiento de valores infinitos, modelos interpretables y exportación de geometrías etiquetadas para visualización en SIG.

La ejecución guardada analiza **18.299 polígonos** y evalúa dos experimentos de regresión logística. Las etiquetas se construyen mediante **umbrales heurísticos**, no mediante verificación independiente en campo. Por ello, las métricas muestran concordancia con esas reglas y **no representan precisión operativa de detección de predios nuevos**.

## Qué demuestra el proyecto

- **Análisis geoespacial:** manejo de GeoDataFrames, atributos de similitud geométrica y exportaciones para revisión en QGIS.
- **Preparación de datos:** revisión de faltantes, correlaciones, distribuciones y tratamiento de distancias infinitas.
- **Machine learning interpretable:** estandarización y regresión logística dentro de un pipeline de scikit-learn.
- **Evaluación y análisis estadístico:** matrices de confusión, precision, recall, F1, curvas ROC y análisis de coeficientes con statsmodels.
- **Comunicación técnica:** exploración visual y comparación explícita de dos definiciones de la variable objetivo.

## Problema y alcance

Al contrastar polígonos de distintas fuentes, la superposición, la distancia entre centroides, la diferencia de orientación y la distancia de Hausdorff pueden ayudar a identificar casos que merecen revisión.

Aquí se estudian esas señales mediante dos definiciones de clasificación. El notebook **consume atributos geométricos ya calculados** en los archivos de entrada; no incluye el proceso original de generación de esos atributos ni una validación catastral externa.

### Datos utilizados en la ejecución guardada

| Elemento | Evidencia en el notebook |
|---|---|
| GeoDataFrame de polígonos de origen Google | 18.299 filas y 30 columnas |
| Entradas locales | `gdf_google_actualizado.parquet` y `gdf_catastro.parquet` |
| Atributos principales | `UOI`, `Centroid_Distance`, `angle_difference`, `neighbors_count`, `Hausdorff_Distance` y `classification_score` |
| Distancias de Hausdorff infinitas | 4.895 valores; se sustituyen por el percentil 99 de los valores finitos |
| Partición de evaluación | 14.639 observaciones de entrenamiento y 3.660 de prueba |

**Los archivos de datos no están incluidos en el repositorio.** La procedencia detallada, la cobertura geográfica, las fechas de captura y las condiciones de uso de las fuentes deben documentarse antes de reutilizar el análisis.

## Flujo de trabajo

1. **Cargar y explorar:** lectura de GeoParquet, revisión de estructura, faltantes, distribuciones y correlaciones.
2. **Preparar variables:** selección de atributos y sustitución de distancias infinitas. Se incluye una exploración de `classification_score` con KMeans de tres grupos.
3. **Definir dos objetivos:** convertir `classification_score > 4` o `Hausdorff_Distance > 9` en etiquetas binarias.
4. **Entrenar:** división aleatoria 80/20 con `random_state=0` y pipeline `StandardScaler → LogisticRegression`.
5. **Evaluar e interpretar:** reportes de clasificación, ROC AUC y modelos Logit de statsmodels.
6. **Preparar revisión en SIG:** código para exportar geometrías y etiquetas a GeoPackage, GeoJSON y pickle.

## Resultados guardados

Estos resultados proceden de las **salidas persistidas del notebook**. No se han reproducido en un entorno limpio.

Ambos experimentos usan **3.660 observaciones de prueba**.

| Experimento | Etiqueta positiva: regla heurística | Accuracy de prueba | ROC AUC | F1 de clase positiva |
|---|---|---:|---:|---:|
| 1 | `classification_score > 4` | 93,83% | 0,9835 | 0,90 |
| 2 | `Hausdorff_Distance > 9` | 88,06% | 0,9494 | 0,88 |

**Cómo interpretar la tabla:** se predicen dos reglas diferentes sobre los datos, no etiquetas independientes de “nuevo” y “existente”. Las métricas no prueban desempeño real en campo ni permiten concluir, por sí solas, que un experimento sea mejor detector que el otro.

- **Experimento 1:** utiliza `UOI`, distancia entre centroides, diferencia angular, número de vecinos y distancia de Hausdorff. Para la clase positiva, el reporte guarda precision de **0,93** y recall de **0,87**.
- **Experimento 2:** sustituye la distancia de Hausdorff como predictor por `classification_score`. Para la clase positiva, el reporte guarda precision de **0,90** y recall de **0,86**.

Los gráficos, los reportes por clase y los coeficientes pueden consultarse en [el notebook completo](clasificacion_predios.ipynb).

## Reproducibilidad

### Qué contiene el repositorio

```text
clasificacion_predios/
├── README.md
└── clasificacion_predios.ipynb
```

El notebook conserva código y resultados de una ejecución anterior. Los datos, los modelos serializados y las capas exportadas **no están versionados aquí**.

### Inspeccionar sin ejecutar

Abre [`clasificacion_predios.ipynb`](clasificacion_predios.ipynb) para revisar el análisis y sus salidas guardadas. No necesitas descargar los datos para esa lectura.

<details>
<summary><strong>Preparar una ejecución local y revisar las exportaciones</strong></summary>

**Requisito previo:** disponer de ambos archivos GeoParquet, con autorización para usarlos y con el esquema que espera el notebook. Sin ellos, el análisis no puede ejecutarse de principio a fin.

El notebook registra Python **3.11.11**. Sus dependencias principales son GeoPandas, pandas, NumPy, Matplotlib, seaborn, statsmodels y scikit-learn. Para leer GeoParquet se requiere soporte Parquet, por ejemplo PyArrow.

La siguiente preparación es una guía inicial basada en los imports, **no un entorno probado ni una instalación con versiones fijadas**:

```bash
git clone https://github.com/fabiantovarc/clasificacion_predios.git
cd clasificacion_predios
python -m venv .venv
```

Activa el entorno virtual según tu sistema operativo y, dentro de él:

```bash
python -m pip install jupyterlab geopandas pandas numpy matplotlib seaborn statsmodels scikit-learn pyarrow
python -m jupyter lab
```

Coloca `gdf_catastro.parquet` y `gdf_google_actualizado.parquet` en el directorio de trabajo del notebook. Revisa el esquema, concilia el estado de los dos experimentos y verifica una ejecución desde un kernel limpio. Las celdas de exportación escriben archivos locales y pueden reemplazar archivos del mismo nombre.

### Salidas previstas por el código

- `classification_comnbinada.gpkg`: capa con geometría y variables de comparación.
- `predios_nuevos_1.json` y `predios_nuevos_2.json`: geometrías con la columna `prediction`.
- Archivos pickle con particiones, modelos o GeoDataFrames.

Las exportaciones de predicciones declaran **EPSG:3116**. Aunque sus nombres contienen “nuevos”, el código actual exporta geometrías con etiquetas **sin filtrar exclusivamente `prediction == 1`**. Deben interpretarse como capas etiquetadas para revisión.

</details>

## Limitaciones y próximos pasos

Este es un prototipo exploratorio. Antes de utilizarlo para decisiones catastrales conviene:

1. **Validar las etiquetas:** construir una muestra independiente revisada por especialistas y documentar cómo se generó `classification_score`.
2. **Separar mejor entrenamiento y evaluación:** calcular el reemplazo de infinitos solo con entrenamiento y evaluar con particiones espaciales o temporales, además de un baseline.
3. **Documentar los datos:** especificar fuentes, permisos, cobertura, fechas, esquema y cálculo de cada atributo geométrico.
4. **Consolidar la reproducción:** fijar dependencias, separar el estado de los dos modelos y verificar una ejecución completa desde un kernel limpio.
5. **Revisar la interpretación estadística:** el segundo Logit guarda una advertencia de cuasi-separación; los AIC/BIC de modelos con objetivos diferentes no deben usarse como comparación directa de calidad.

PCA aparece en los imports y en el plan del notebook, pero **no hay un experimento PCA implementado**.

## Autoría y créditos

Repositorio publicado por [@fabiantovarc](https://github.com/fabiantovarc).

El notebook identifica datos de origen Google y Catastro y utiliza herramientas de código abierto del ecosistema Python. La publicación del código no establece por sí sola permisos de redistribución de los datos de entrada.

