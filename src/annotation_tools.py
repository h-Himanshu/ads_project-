
import argparse
import glob
import hashlib
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from data_loader import (
    GOLD_COLUMN,
    LABEL_ORDER,
    RATING_COLUMN,
    clean_text,
    read_reviews,
    sentiment_from_rating,
)

NEGATIVE_WORDS = {"worst", "wrost", "bad", "poor", "bakwas", "naramro", "waste", "fake", "useless",
                  "faltu", "pheki", "broken", "defective", "defected", "disappointed", "terrible",
                  "horrible", "scam", "duplicate", "refund"}
POSITIVE_WORDS = {"good", "great", "best", "excellent", "amazing", "ramro", "babal", "dami", "love",
                  "nice", "awesome", "perfect", "superb", "satisfied", "fantastic"}


def review_key(product, text):
    return hashlib.md5(f"{product}|{text}".encode("utf-8")).hexdigest()[:12]


def add_keys(data, text_column):
    data = data.copy()
    data["text_clean"] = data[text_column].map(clean_text)
    products = data["Product"].astype(str) if "Product" in data.columns else pd.Series("", index=data.index)
    data["review_key"] = [review_key(p, t) for p, t in zip(products, data["text_clean"])]
    return data


def looks_mismatched(text, star_label):
    words = set(text.lower().split())
    if star_label == "positive":
        return bool(words & NEGATIVE_WORDS) and not words & POSITIVE_WORDS
    if star_label == "negative":
        return bool(words & POSITIVE_WORDS) and not words & NEGATIVE_WORDS
    return False


def make_sheets(args):
    data = add_keys(read_reviews(args.data), args.text_column)
    data = data[data["text_clean"].str.split().str.len() >= args.min_words]
    data = data.drop_duplicates("review_key")
    data["star_label"] = sentiment_from_rating(data[RATING_COLUMN])
    if GOLD_COLUMN in data.columns:                       # skip reviews that are already labelled
        done = data[GOLD_COLUMN].astype("string").str.strip().fillna("") != ""
        data = data[~done]

    # sample more negative / neutral reviews than the natural 85% positive share
    wanted = {"negative": 0.40, "neutral": 0.30, "positive": 0.30}
    parts = []
    for label, share in wanted.items():
        pool = data[data["star_label"] == label]
        flagged = pool[[looks_mismatched(t, label) for t in pool["text_clean"]]]
        rest = pool.drop(flagged.index)
        n = int(args.total * share)
        take = pd.concat([flagged.head(n), rest.sample(min(max(n - len(flagged), 0), len(rest)),
                                                       random_state=args.seed)])
        parts.append(take)
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    overlap = min(args.overlap, len(sample))
    shared, unique = sample.iloc[:overlap], sample.iloc[overlap:]
    folders = [unique.iloc[i::len(args.annotators)] for i in range(len(args.annotators))]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, own in zip(args.annotators, folders):
        sheet = pd.concat([shared, own])[["review_key", "text_clean"]]
        sheet = sheet.sample(frac=1, random_state=sum(map(ord, name)))    # different order for everyone
        sheet = sheet.rename(columns={"text_clean": "Review"}).assign(label="", notes="")
        path = out / f"annotation_{name}.csv"
        sheet.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{path}: {len(sheet)} reviews ({overlap} shared with everyone)")
    print("\nStar-label mix of the sample:", sample["star_label"].value_counts().to_dict())
    print("Fill the 'label' column with: negative / neutral / positive")


def read_sheet(path):
    sheet = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    sheet["label"] = sheet["label"].str.strip().str.lower()
    bad = ~sheet["label"].isin(LABEL_ORDER + [""])
    if bad.any():
        raise ValueError(f"{path}: invalid labels {sheet.loc[bad, 'label'].unique().tolist()}")
    return sheet[sheet["label"] != ""].set_index("review_key")["label"]


def kappa(args):
    sheets = {Path(p).stem: read_sheet(p) for p in args.sheets}
    scores = []
    for (name_a, a), (name_b, b) in itertools.combinations(sheets.items(), 2):
        common = a.index.intersection(b.index)
        if len(common) < 10:
            print(f"{name_a} vs {name_b}: only {len(common)} shared reviews - skipped")
            continue
        score = cohen_kappa_score(a[common], b[common])
        scores.append(score)
        agreement = float((a[common] == b[common]).mean())
        print(f"{name_a} vs {name_b}: kappa = {score:.3f} (raw agreement {agreement:.1%}, {len(common)} reviews)")
    if scores:
        print(f"Average kappa: {np.mean(scores):.3f}   "
              "(about 0.6-0.8 = substantial, above 0.8 = almost perfect)")


def merge(args):
    paths = [p for pattern in args.sheets for p in glob.glob(pattern)]
    if not paths:
        raise FileNotFoundError("No annotation sheets found.")
    votes = {}
    for path in paths:
        for key, label in read_sheet(path).items():
            votes.setdefault(key, []).append(label)

    final, unresolved = {}, []
    for key, labels in votes.items():
        (top, count), = Counter(labels).most_common(1)
        if len(labels) == 1 or count > len(labels) / 2:
            final[key] = top
        else:
            unresolved.append(key)          # no majority: discuss and decide together

    data = add_keys(read_reviews(args.data), args.text_column)
    data[GOLD_COLUMN] = data["review_key"].map(final).fillna("")
    star = sentiment_from_rating(data[RATING_COLUMN])
    labelled = data[GOLD_COLUMN] != ""
    changed = (data.loc[labelled, GOLD_COLUMN] != star[labelled]).sum()
    print(f"Hand-labelled reviews: {int(labelled.sum())} | no majority (needs discussion): {len(unresolved)}")
    print(f"Hand label differs from the star-based label in {changed} reviews "
          f"({changed / max(labelled.sum(), 1):.1%}) - report this number.")
    print("Confusion of star label (rows) vs hand label (columns):")
    print(pd.crosstab(star[labelled], data.loc[labelled, GOLD_COLUMN]).to_string())
    if unresolved:
        pd.DataFrame({"review_key": unresolved}).to_csv(Path(args.output or args.data).with_name("needs_discussion.csv"), index=False)

    output = Path(args.output or args.data)
    if output == Path(args.data):
        backup = output.with_suffix(".backup.csv")
        read_reviews(args.data).to_csv(backup, index=False, encoding="utf-8-sig")
        print(f"Backup of the original dataset: {backup}")
    data.drop(columns=["text_clean", "review_key"]).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {output} with the '{GOLD_COLUMN}' column.")


def main():
    parser = argparse.ArgumentParser(description="Hand-labelling tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-sheets", help="create one annotation sheet per annotator")
    p.add_argument("--data", default="data/review_dataset.csv")
    p.add_argument("--text-column", default="Review")
    p.add_argument("--annotators", nargs="+", default=["Bikash", "Himanshu", "Sumin"])
    p.add_argument("--total", type=int, default=1500, help="reviews in all sheets together")
    p.add_argument("--overlap", type=int, default=200, help="reviews labelled by everybody")
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=make_sheets)

    p = sub.add_parser("kappa", help="agreement between annotators")
    p.add_argument("--sheets", nargs="+", required=True)
    p.set_defaults(func=kappa)

    p = sub.add_parser("merge", help="write gold_sentiment into the dataset")
    p.add_argument("--sheets", nargs="+", required=True, help="sheet files or patterns")
    p.add_argument("--data", default="data/review_dataset.csv")
    p.add_argument("--text-column", default="Review")
    p.add_argument("--output", default=None, help="default: overwrite --data (a backup is kept)")
    p.set_defaults(func=merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()