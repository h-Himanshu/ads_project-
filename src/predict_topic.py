"""Assign topics to NEW reviews using the LDA model trained by topic_modelling.py.

Run from the project root (with PYTHONPATH=src):

    python src/predict_topic.py "battery backup is good but charger is slow"
    python src/predict_topic.py "review one" "review two"
    python src/predict_topic.py --csv data/new_reviews.csv --column Review
    python src/predict_topic.py --init-labels      # create topic_labels.json to fill in

"""

import argparse
import json
from pathlib import Path

import pandas as pd
from gensim.models import LdaModel
from gensim.models.phrases import Phraser 

# IMPORTANT: reuse the SAME cleaning and tokenizing as training. If you change
# tokenize() (lemmatization, bigrams, new stopwords) in topic_modelling.py, this
# script picks the change up automatically. Just retrain the model afterwards.
from data_loader import clean_text
from topic_modelling import tokenize , apply_bigrams 

DEFAULT_MODEL_DIR = Path("results/topic_results")
LABELS_FILE = "topic_labels.json"


class TopicPredictor:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, low_confidence=0.40):
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "lda_model"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run "
                "'python src/topic_modelling.py' first."
            )
        self.model = LdaModel.load(str(model_path))
        self.bigram_model=Phraser.load(str(self.model_dir/"bigrams.model"))
        self.num_topics = self.model.num_topics
        self.low_confidence = low_confidence
        self.labels = self._load_labels()

    def _load_labels(self):
        """Optional human-written names, e.g. {"0": "Battery and charging"}."""
        path = self.model_dir / LABELS_FILE
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {int(k): v for k, v in raw.items() if v}
        return {}

    def label_for(self, topic_id):
        return self.labels.get(topic_id, f"Topic {topic_id}")

    def top_words(self, topic_id, n=8):
        return [w for w, _ in self.model.show_topic(topic_id, topn=n)]

    def predict(self, review, top_n=3):
        """Return the dominant topic plus the top_n most likely topics."""
        tokens = tokenize(clean_text(review))
        tokens=apply_bigrams(tokens,self.bigram_model)
        bow = self.model.id2word.doc2bow(tokens)  # unknown words are ignored

        if not bow:
            return {
                "review": review,
                "status": "no_known_words",
                "message": "None of the words were seen during training.",
                "topic_id": None,
                "label": None,
                "probability": None,
                "top_topics": [],
                "known_words": [],
            }

        dist = self.model.get_document_topics(bow, minimum_probability=0)
        dist = sorted(dist, key=lambda x: x[1], reverse=True)
        topic_id, prob = dist[0]
        known = [self.model.id2word[i] for i, _ in bow]

        return {
            "review": review,
            "status": "low_confidence" if prob < self.low_confidence else "ok",
            "message": "",
            "topic_id": int(topic_id),
            "label": self.label_for(topic_id),
            "probability": float(prob),
            "top_words": self.top_words(topic_id),
            "top_topics": [
                {"topic_id": int(t), "label": self.label_for(t), "probability": float(p)}
                for t, p in dist[:top_n]
            ],
            "known_words": known,
        }


def init_labels(predictor):
    """Write a template the user fills in after reading the topics."""
    path = predictor.model_dir / LABELS_FILE
    if path.exists():
        print(f"{path} already exists - not overwriting.")
        return
    template = {str(i): "" for i in range(predictor.num_topics)}
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(f"Created {path}. Fill in a name for each topic. Hints:\n")
    for i in range(predictor.num_topics):
        print(f'  "{i}": top words = {", ".join(predictor.top_words(i))}')


def print_result(r):
    print("-" * 60)
    print(f"Review : {r['review']}")
    if r["status"] == "no_known_words":
        print(f"Result : {r['message']} (too short or unfamiliar text)")
        return
    print(f"Topic  : {r['label']}  (probability {r['probability']:.0%})")
    print(f"Words  : {', '.join(r['top_words'])}")
    print(f"Matched: {', '.join(r['known_words'])}")
    others = ", ".join(f"{t['label']} {t['probability']:.0%}" for t in r["top_topics"][1:])
    if others:
        print(f"Others : {others}")
    if r["status"] == "low_confidence":
        print("Note   : low confidence - the review may mix several topics "
              "or contain few known words.")


def main():
    parser = argparse.ArgumentParser(description="Predict topics for new reviews.")
    parser.add_argument("reviews", nargs="*", help="One or more review texts.")
    parser.add_argument("--csv", help="CSV file with reviews to label.")
    parser.add_argument("--column", default="Review", help="Text column in the CSV.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output", default=None,
                        help="Where to save CSV results (default: <csv>_with_topics.csv).")
    parser.add_argument("--init-labels", action="store_true",
                        help="Create topic_labels.json to fill in with topic names.")
    args = parser.parse_args()

    predictor = TopicPredictor(args.model_dir)

    if args.init_labels:
        init_labels(predictor)
        return

    if args.csv:
        df = pd.read_csv(args.csv)
        results = [predictor.predict(str(t)) for t in df[args.column].fillna("")]
        df["topic_id"] = pd.array([r["topic_id"] for r in results], dtype="Int64")
        df["topic"] = [r["label"] for r in results]
        df["topic_probability"] = [r["probability"] for r in results]
        df["topic_status"] = [r["status"] for r in results]
        out = args.output or str(Path(args.csv).with_name(Path(args.csv).stem + "_with_topics.csv"))
        df.to_csv(out, index=False)
        print(f"Labeled {len(df)} reviews -> {out}")
        return

    if not args.reviews:
        parser.error("Give at least one review text, or use --csv / --init-labels.")
    for review in args.reviews:
        print_result(predictor.predict(review))


if __name__ == "__main__":
    main()