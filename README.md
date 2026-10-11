# Evaluación comparativa de modelos preentrenados de Hugging Face

Actividad académica de **Gestión de proyectos de inteligencia artificial**.

## Objetivo
Comparar tres modelos preentrenados para análisis de sentimiento en español y seleccionar la alternativa con mejor equilibrio entre calidad predictiva y eficiencia computacional.

## Caso de uso
Clasificación de reseñas de productos en español en `NEG`, `NEU` y `POS`, usando 5,000 reseñas de prueba del Multilingual Amazon Reviews Corpus (MARC).

## Modelos y resultados
| Modelo | Accuracy | F1 macro | Latencia ms/reseña | VRAM pico MB |
|---|---:|---:|---:|---:|
| NLP Town mBERT | 0.7650 | 0.7153 | 5.56 | 780.5 |
| RoBERTuito | 0.6836 | 0.6191 | 5.14 | 556.8 |
| XLM-R Twitter | 0.6754 | 0.5882 | 6.01 | 1202.4 |

**Modelo recomendado: NLP Town mBERT.** Fue el único que cumplió simultáneamente `F1_macro >= 0.70` y `Recall_NEG >= 0.70`.

## Entorno
La corrida documentada se realizó en Google Colab con GPU Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 5.19.0, Datasets 5.1.0 y Scikit-learn 1.9.1.

## Estructura
```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── Actividad_4_Valeria_Ochoa_Mondragon.ipynb
├── results/
│   ├── decision_multicriterio.csv
│   ├── environment.json
│   ├── metricas_modelos.csv
│   ├── metricas_por_clase.csv
│   └── revision_umbrales.csv
└── src/
    ├── decision.py
    ├── evaluation.py
    ├── generate_dataset.py
    └── preprocessing.py
```

El dataset completo se entrega en el ZIP de la actividad y puede regenerarse con `src/generate_dataset.py`. Las predicciones por registro se generan al ejecutar el notebook; los resultados agregados usados para la decisión se conservan en `results/`.

## Ejecución
1. Abrir el notebook en Google Colab.
2. Activar una GPU.
3. Ejecutar las dependencias.
4. Autenticarse en Hugging Face mediante Colab Secrets o login interactivo.
5. Ejecutar el notebook completo.
6. Revisar los archivos de `results/`.

## Limitaciones
La conversión de cinco estrellas a tres clases puede producir discrepancias entre la polaridad lingüística y la etiqueta. La clase neutral fue la más difícil. RoBERTuito y XLM-R Twitter presentan además cambio de dominio respecto de las reseñas de productos.

## Referencias
- Keung, P., Lu, Y., Szarvas, G., & Smith, N. A. (2020). *The Multilingual Amazon Reviews Corpus*. EMNLP 2020. https://doi.org/10.18653/v1/2020.emnlp-main.369
- Pérez, J. M., Furman, D. A., Alonso Alemany, L., & Luque, F. M. (2022). *RoBERTuito: A pre-trained language model for social media text in Spanish*. LREC 2022.
- Barbieri, F., Espinosa Anke, L., & Camacho-Collados, J. (2022). *XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond*. LREC 2022.
