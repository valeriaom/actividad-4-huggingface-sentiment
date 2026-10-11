LABEL_ORDER = ["NEG", "NEU", "POS"]

def stars_label_to_sentiment(label_0_to_4):
    """Convierte la etiqueta 0-4 del dataset a NEG, NEU o POS."""
    stars = int(label_0_to_4) + 1
    if stars <= 2:
        return "NEG"
    if stars == 3:
        return "NEU"
    return "POS"

def normalize_prediction(model_name, raw_label):
    """Normaliza las salidas de los checkpoints a NEG, NEU y POS."""
    label = str(raw_label).strip().lower()

    if model_name == "NLP Town mBERT":
        if label.startswith(("1", "2")):
            return "NEG"
        if label.startswith("3"):
            return "NEU"
        if label.startswith(("4", "5")):
            return "POS"

    mapping = {
        "negative": "NEG",
        "neg": "NEG",
        "neutral": "NEU",
        "neu": "NEU",
        "positive": "POS",
        "pos": "POS",
    }
    if label in mapping:
        return mapping[label]

    raise ValueError(f"Etiqueta no reconocida para {model_name}: {raw_label}")
