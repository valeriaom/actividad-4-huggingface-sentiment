# Evaluación comparativa de modelos preentrenados de Hugging Face

Actividad académica de la materia **Gestión de proyectos de inteligencia artificial**.

## Objetivo

Comparar tres modelos preentrenados para análisis de sentimiento en español y seleccionar la alternativa con mejor equilibrio entre calidad predictiva y eficiencia computacional.

## Caso de uso

Clasificación de reseñas de productos en español en tres categorías:

- `NEG`: 1 o 2 estrellas
- `NEU`: 3 estrellas
- `POS`: 4 o 5 estrellas

El conjunto de evaluación utiliza 5,000 reseñas de prueba en español derivadas del **Multilingual Amazon Reviews Corpus (MARC)**.

## Modelos

| Modelo | Hugging Face ID | Contexto |
|---|---|---|
| NLP Town mBERT | `nlptown/bert-base-multilingual-uncased-sentiment` | Reseñas de productos |
| RoBERTuito | `pysentimiento/robertuito-sentiment-analysis` | Texto social en español |
| XLM-R Twitter | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Twitter multilingüe |

## Resultados principales

| Modelo | Accuracy | F1 macro | Latencia ms/reseña | VRAM pico MB |
|---|---:|---:|---:|---:|
| NLP Town mBERT | 0.7650 | 0.7153 | 5.56 | 780.5 |
| RoBERTuito | 0.6836 | 0.6191 | 5.14 | 556.8 |
| XLM-R Twitter | 0.6754 | 0.5882 | 6.01 | 1202.4 |

**Modelo recomendado:** NLP Town mBERT.

Fue el único modelo que cumplió simultáneamente los umbrales experimentales de `F1_macro >= 0.70` y `Recall_NEG >= 0.70`. RoBERTuito presentó la mejor eficiencia computacional, por lo que representa una alternativa cuando memoria y throughput tienen prioridad.

## Entorno de ejecución

La corrida documentada se realizó en Google Colab con:

- GPU: Tesla T4
- Python: 3.13.15
- PyTorch: 2.11.0+cu130
- Transformers: 5.19.0
- Datasets: 5.1.0
- Scikit-learn: 1.9.1

La latencia depende del hardware y de la carga de la sesión, por lo que debe interpretarse como una comparación relativa dentro de la misma ejecución.

## Estructura

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── dataset_evaluacion.csv
├── notebooks/
│   └── Actividad_4_Valeria_Ochoa_Mondragon.ipynb
├── results/
│   ├── decision_multicriterio.csv
│   ├── environment.json
│   ├── metricas_modelos.csv
│   ├── metricas_por_clase.csv
│   ├── predicciones_modelos.csv
│   └── revision_umbrales.csv
└── src/
    ├── decision.py
    ├── evaluation.py
    └── preprocessing.py
```

## Ejecución

1. Abrir el notebook en Google Colab.
2. Seleccionar una GPU como acelerador de hardware.
3. Ejecutar la instalación de dependencias.
4. Guardar el token de Hugging Face como `HF_TOKEN` en Colab Secrets o utilizar el inicio de sesión interactivo.
5. Ejecutar el notebook de principio a fin.
6. Verificar los archivos generados en `results/`.

## Metodología

Los modelos se evaluaron sobre el mismo conjunto de datos, con batch size y longitud máxima homogéneos. La latencia se midió después de un warm-up y con sincronización de CUDA. La descarga y carga de modelos no se incluyeron en el tiempo principal de inferencia.

Las métricas utilizadas fueron accuracy, precision macro, recall macro y F1 macro, además de desempeño por clase, matrices de confusión, throughput, parámetros y VRAM pico.

## Limitaciones

La transformación de cinco estrellas a tres categorías puede producir discrepancias entre la polaridad lingüística y la etiqueta asignada. La clase neutral fue la más difícil para los tres modelos.

RoBERTuito y XLM-R Twitter presentan además cambio de dominio, ya que fueron desarrollados principalmente con texto de redes sociales. Antes de reutilizar los modelos fuera de un contexto académico debe revisarse la licencia vigente de cada checkpoint.

## Referencias principales

- Keung, P., Lu, Y., Szarvas, G., & Smith, N. A. (2020). *The Multilingual Amazon Reviews Corpus*. EMNLP 2020. https://doi.org/10.18653/v1/2020.emnlp-main.369
- Pérez, J. M., Furman, D. A., Alonso Alemany, L., & Luque, F. M. (2022). *RoBERTuito: A pre-trained language model for social media text in Spanish*. LREC 2022.
- Barbieri, F., Espinosa Anke, L., & Camacho-Collados, J. (2022). *XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond*. LREC 2022.
