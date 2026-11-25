"""LDA topic modelling of the reviews, and topic-vs-sentiment analysis.

    python src/topic_modelling.py                         # all reviews, k chosen by coherence (3-10)
    python src/topic_modelling.py --num-topics 6          # force k
    python src/topic_modelling.py --per-category          # also one model per product category
    python src/topic_modelling.py --category Electronics  # one category only

How it differs from the first version
  * apostrophes are handled ("don't" no longer leaves the word "don")
  * romanized-Nepali opinion words and generic words are removed, so topics are
    less about the language ("ramro, thikai, hunxa") or about nothing ("product, got")
  * the model is trained on unique reviews with enough words, but EVERY review
    still gets a topic (reviews with no usable word get topic -1, not dropped)
  * k is searched from 3 to 10 using c_v coherence
  * a model can be trained for one product category only (topics are cleaner)
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # draw into image files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel, TfidfModel
from gensim.models.phrases import Phrases
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from gensim.models.phrases import FrozenPhrases as Phraser
except ImportError:                                   # older gensim
    from gensim.models.phrases import Phraser

from data_loader import (
    RATING_COLUMN,
    clean_text,
    normalize_variants,
    read_reviews,
    sentiment_from_rating,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "review_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "review_dataset_topic_results"

# ----------------------------------------------------------------------------
# Stopwords. Words that carry no TOPIC meaning. (Negations and opinion words are
# kept for sentiment, but removed here because they appear in every topic.)
# EXTEND THESE LISTS after looking at your results, or put one word per line in
# data/extra_stopwords.txt.
# ----------------------------------------------------------------------------
try:
    from nltk.corpus import stopwords as _nltk_stopwords
    ENGLISH_STOPWORDS = set(_nltk_stopwords.words("english"))
    STOPWORD_SOURCE = "NLTK"
except Exception:                                      # package or corpus missing
    ENGLISH_STOPWORDS = set(ENGLISH_STOP_WORDS)
    STOPWORD_SOURCE = "scikit-learn"

# contractions after apostrophes are removed: don't -> dont
CONTRACTION_STOPWORDS = set("""
dont didnt doesnt isnt wasnt arent werent cant couldnt wont wouldnt shouldnt
havent hasnt hadnt youre theyre thats whats its ive hes shes weve ill don didn doesn
""".split())

OPINION_STOPWORDS = set("""
good nice best great awesome amazing excellent perfect perfectly love loved like liked well
wow super superb decent pretty happy highly satisfied recommend recommended fine okay ok
better bad worst poor awful terrible overall thank thanks thankyou
""".split())

# generic words that appear in every kind of review and separate nothing
GENERIC_STOPWORDS = set("""
product products item items got get getting gets really fully try tried guys guy lot bit
thing things use used using buy bought buying ordered order purchase purchased time
one also just make made yes sir dear kinda actually still even much many every
received receive came come going went give gave want wanted need needed hrs
""".split())

# romanized Nepali: grammar words, verbs and opinion words (merged spellings
# from data_loader.VARIANT_MAP are used, e.g. xa -> cha)
NEPALI_ROMANIZED_STOPWORDS = set("""
cha chha xa chaa ho hai ra ko ka ki le lai ma ni nai pani ta tara yo tyo hola mero timro hamro
chai ani malai jasto maile aru ali ekdam dherai sabai kei ke kun kina kasari
chaina hoina hudaina huncha raicha thiyo thiye bhayo bhako lagyo lagcha garxu garchu garyo
gare garnu garna garne garera lida lina dinu dinus paye pako leko liyo kineko hunu hunxa
ramro naramro thik thikai dami babal saman anusar hajur paisa
""".split())


def load_word_file(file_path):
    """One word per line -> set of lowercase words (empty set if no file)."""
    words = set()
    file_path = Path(file_path)
    if file_path.exists():
        for line in file_path.read_text(encoding="utf-8").splitlines():
            word = line.strip().lower()
            if word:
                words.add(word)
    return words


ROMANIZED_FILE_STOPWORDS = load_word_file(PROJECT_ROOT / "data" / "NepaliStopWords_romanized.txt")
EXTRA_STOPWORDS = load_word_file(PROJECT_ROOT / "data" / "extra_stopwords.txt")
STOPWORDS = (ENGLISH_STOPWORDS | CONTRACTION_STOPWORDS | OPINION_STOPWORDS | GENERIC_STOPWORDS
             | NEPALI_ROMANIZED_STOPWORDS | ROMANIZED_FILE_STOPWORDS | EXTRA_STOPWORDS)


def tokenize(text):
    """Lowercase letter-only words of 3+ characters, without stopwords.

    Spelling variants are merged and apostrophes removed first, so the same word
    is counted once. Devanagari text has no a-z letters and gives no tokens.
    """
    words = re.findall(r"[a-z]+", normalize_variants(text))
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def learn_bigrams(all_tokens, min_count=3, threshold=0.4):
    """Learn which neighbouring words are joined (battery backup -> battery_backup)."""
    phrase_model = Phrases(all_tokens, min_count=min_count, threshold=threshold, scoring="npmi")
    found = phrase_model.find_phrases(all_tokens)
    print(f"Learned {len(found)} bigrams. Strongest ones:")
    for phrase, score in sorted(found.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(f"   {phrase}  (score {score:.2f})")
    return Phraser(phrase_model)


def apply_bigrams(review_tokens, bigram_model):
    return list(bigram_model[review_tokens])


def remove_low_tfidf_words(corpus, dictionary, threshold):
    """Optional: drop, from each review, the words whose TF-IDF there is below threshold."""
    tfidf_model = TfidfModel(corpus)
    removed_counts, new_corpus = {}, []
    for bow in corpus:
        scores = dict(tfidf_model[bow])
        new_bow = []
        for word_id, count in bow:
            if scores.get(word_id, 0.0) >= threshold:
                new_bow.append((word_id, count))
            else:
                word = dictionary[word_id]
                removed_counts[word] = removed_counts.get(word, 0) + count
        new_corpus.append(new_bow)
    print("Words removed by the TF-IDF filter (most removed first):")
    for word, count in sorted(removed_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"   {word}: {count}")
    return new_corpus


def build_corpus(texts, min_docs=3, max_ratio=0.5, min_tokens=3, tfidf_threshold=0.0):
    """Tokenize everything, but learn bigrams and the vocabulary from training reviews only.

    Training reviews = first occurrence of each distinct text that still has at
    least `min_tokens` words. All reviews get a bag-of-words for later inference.
    """
    tokens = [tokenize(t) for t in texts]
    first_seen, train_idx = set(), []
    for i, (text, toks) in enumerate(zip(texts, tokens)):
        key = text.lower()
        if len(toks) >= min_tokens and key not in first_seen:
            first_seen.add(key)
            train_idx.append(i)
    if len(train_idx) < 10:
        raise ValueError("Too few reviews with enough words to train a topic model.")

    bigram_model = learn_bigrams([tokens[i] for i in train_idx])
    tokens = [apply_bigrams(t, bigram_model) for t in tokens]

    dictionary = corpora.Dictionary(tokens[i] for i in train_idx)
    dictionary.filter_extremes(no_below=min_docs, no_above=max_ratio)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    if tfidf_threshold > 0:
        corpus = remove_low_tfidf_words(corpus, dictionary, tfidf_threshold)
    train_idx = [i for i in train_idx if len(corpus[i]) > 0]
    return tokens, dictionary, corpus, bigram_model, train_idx


def train_lda(corpus, dictionary, k, seed=42):
    return LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=15,
                    iterations=100, random_state=seed, alpha="auto", eta="auto")


def coherence(model, tokens, dictionary):
    return CoherenceModel(model=model, texts=tokens, dictionary=dictionary,
                          coherence="c_v").get_coherence()


def pick_best_k(corpus, dictionary, tokens, k_values, output_dir, seed=42):
    """Try several topic counts and keep the one with the best coherence."""
    scores = {}
    for k in k_values:
        scores[k] = float(coherence(train_lda(corpus, dictionary, k, seed), tokens, dictionary))
        print(f"  k={k:>2}  coherence={scores[k]:.4f}")
    plt.figure(figsize=(6, 4))
    plt.plot(list(scores), list(scores.values()), marker="o")
    plt.xlabel("Number of topics (k)")
    plt.ylabel("Coherence (c_v)")
    plt.title("Choosing the number of topics")
    plt.tight_layout()
    plt.savefig(output_dir / "coherence_vs_k.png", dpi=160)
    plt.close()
    return max(scores, key=scores.get), scores


def save_ldavis(model, corpus, dictionary, path):
    """Interactive topic map (pyLDAvis). Open the saved HTML file in a browser."""
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis
    except ImportError:
        print("pyLDAvis not installed - skipping (pip install pyLDAvis).")
        return
    try:
        vis = gensimvis.prepare(model, corpus, dictionary, mds="mmds", R=30, sort_topics=False)
        pyLDAvis.save_html(vis, str(path))
    except Exception as error:                         # visualisation must never stop the run
        print(f"pyLDAvis failed ({error}) - skipping.")


def save_wordclouds(model, k, path):
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud not installed - skipping word clouds.")
        return
    cols = min(k, 3)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
    for topic_id, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if topic_id < k:
            frequencies = dict(model.show_topic(topic_id, topn=30))
            ax.imshow(WordCloud(width=500, height=320, background_color="white")
                      .generate_from_frequencies(frequencies))
            ax.set_title(f"Topic {topic_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def topic_sentiment_tables(data, output_dir):
    """Topic x sentiment tables, for rating-based and (if present) predicted sentiment.

    Reviews without a topic (-1) are left out. A chi-square p-value says whether
    the sentiment mix really differs between topics (small p = it does).
    """
    from scipy.stats import chi2_contingency

    order = ["negative", "neutral", "positive"]
    summary = {}
    for column, suffix in (("sentiment", ""), ("predicted_sentiment", "_predicted")):
        if column not in data.columns:
            continue
        subset = data[data["topic"] >= 0]
        values = subset[column].astype("string").str.strip().str.lower()
        crosstab = pd.crosstab(subset["topic"], values).reindex(columns=order, fill_value=0)
        crosstab.to_csv(output_dir / f"topic_sentiment_crosstab{suffix}.csv")
        share = crosstab.div(crosstab.sum(axis=1), axis=0).fillna(0).round(4)
        share.to_csv(output_dir / f"topic_sentiment_share{suffix}.csv")
        try:
            p_value = float(chi2_contingency(crosstab.loc[:, crosstab.sum() > 0])[1])
        except ValueError:
            p_value = None
        summary[column] = {"chi2_p_value": p_value}
        print(f"\nTopic x {column} (share of each topic's reviews):")
        print(share.to_string())
        print(f"chi-square p-value: {p_value}")
    return summary


def run_topic_model(data, out, args):
    """Train, evaluate and save one topic model for the reviews in `data`."""
    out.mkdir(parents=True, exist_ok=True)
    texts = data["text"].tolist()
    tokens, dictionary, corpus, bigram_model, train_idx = build_corpus(
        texts, min_tokens=args.min_tokens, tfidf_threshold=args.tfidf_threshold)
    if len(dictionary) < 5:
        raise ValueError("Vocabulary is too small after filtering. Lower min_docs or add reviews.")
    train_corpus = [corpus[i] for i in train_idx]
    train_tokens = [tokens[i] for i in train_idx]
    print(f"Reviews: {len(data)} | used to train the topic model: {len(train_idx)} | "
          f"vocabulary: {len(dictionary)}")

    if args.num_topics:
        k, scores = args.num_topics, {}
    else:
        print(f"Searching for the best number of topics ({args.min_k}-{args.max_k})...")
        k, scores = pick_best_k(train_corpus, dictionary, train_tokens,
                                range(args.min_k, min(args.max_k, len(train_idx) - 1) + 1),
                                out, args.seed)
    print(f"Using k = {k}")
    model = train_lda(train_corpus, dictionary, k, args.seed)
    final_coherence = float(coherence(model, train_tokens, dictionary))

    topics = {}
    for topic_id in range(k):
        words = [w for w, _ in model.show_topic(topic_id, topn=10)]
        topics[topic_id] = words
        print(f"Topic {topic_id}: {', '.join(words)}")

    # dominant topic for EVERY review (-1 = no usable word)
    topic_ids, topic_probs = [], []
    for bow in corpus:
        if not bow:
            topic_ids.append(-1)
            topic_probs.append(0.0)
        else:
            best = max(model.get_document_topics(bow, minimum_probability=0), key=lambda x: x[1])
            topic_ids.append(int(best[0]))
            topic_probs.append(float(best[1]))
    data = data.copy()
    data["topic"] = topic_ids
    data["topic_prob"] = topic_probs
    data["used_for_training"] = False
    data.loc[data.index[train_idx], "used_for_training"] = True

    labels_path = out / "topic_labels.json"
    if not labels_path.exists():
        labels_path.write_text(json.dumps({str(i): "" for i in range(k)}, indent=2), encoding="utf-8")
    names = {int(i): n for i, n in json.loads(labels_path.read_text(encoding="utf-8")).items() if n}
    data["topic_label"] = [names.get(t, f"Topic {t}") if t >= 0 else "No topic" for t in topic_ids]
    data.to_csv(out / "reviews_with_topics.csv", index=False, encoding="utf-8-sig")

    sentiment_summary = topic_sentiment_tables(data, out)
    save_wordclouds(model, k, out / "topic_wordclouds.png")
    save_ldavis(model, train_corpus, dictionary, out / "ldavis.html")

    summary = {
        "num_topics": k,
        "coherence_c_v": final_coherence,
        "coherence_search": scores,
        "topics_top_words": topics,
        "reviews_per_topic": data["topic"].value_counts().sort_index().to_dict(),
        "reviews_without_topic": int((data["topic"] == -1).sum()),
        "reviews_used_for_training": len(train_idx),
        "mean_dominant_topic_probability": float(data.loc[data["topic"] >= 0, "topic_prob"].mean()),
        "settings": {"min_tokens": args.min_tokens, "tfidf_threshold": args.tfidf_threshold,
                     "seed": args.seed, "stopwords_from": STOPWORD_SOURCE},
        "sentiment_tests": sentiment_summary,
    }
    (out / "topics_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    model.save(str(out / "lda_model"))
    bigram_model.save(str(out / "bigrams.model"))
    print(f"\nCoherence (c_v): {final_coherence:.4f}")
    print(f"Saved everything to: {out}")
    print(f"Next: read the top words and example reviews of each topic, then write a name "
          f"for each topic in {labels_path} and run this script again.")
    return summary


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "category"


def main():
    parser = argparse.ArgumentParser(description="LDA topic modelling of reviews.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--text-column", default="Review")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-topics", type=int, default=None, help="Fix k. Default: best k in min-k..max-k.")
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--min-tokens", type=int, default=3,
                        help="Reviews with fewer topic words than this are not used for training.")
    parser.add_argument("--tfidf-threshold", type=float, default=0.0, help="Remove words with TF-IDF below this. 0 = off.")
    parser.add_argument("--category", default=None, help="Model only reviews of this Category.")
    parser.add_argument("--per-category", action="store_true",
                        help="Also train one model for every category with enough reviews.")
    parser.add_argument("--min-category-reviews", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_topics is not None and args.num_topics < 2:
        parser.error("--num-topics must be at least 2.")

    def absolute(path):
        path = Path(path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    print(f"English stopwords from {STOPWORD_SOURCE}; total stopwords: {len(STOPWORDS)}")
    data = read_reviews(absolute(args.data))
    if args.text_column not in data.columns:
        raise ValueError(f"Text column '{args.text_column}' not found. Columns: {list(data.columns)}")
    data["text"] = data[args.text_column].map(clean_text)
    data = data[data["text"] != ""].reset_index(drop=True)
    if "sentiment" not in data.columns and RATING_COLUMN in data.columns:
        data["sentiment"] = sentiment_from_rating(data[RATING_COLUMN])   # star-based sentiment
    if data.empty:
        raise ValueError("No non-empty reviews were found after text cleaning.")

    out = absolute(args.output_dir)
    if args.category:
        if "Category" not in data.columns:
            raise ValueError("The CSV has no 'Category' column.")
        data = data[data["Category"].astype(str).str.lower() == args.category.lower()].reset_index(drop=True)
        if data.empty:
            raise ValueError(f"No reviews found for category '{args.category}'.")
        run_topic_model(data, out / slug(args.category), args)
        return

    run_topic_model(data, out, args)
    if args.per_category and "Category" in data.columns:
        for category, part in data.groupby("Category"):
            if len(part) < args.min_category_reviews:
                print(f"\nSkipping category '{category}' ({len(part)} reviews).")
                continue
            print(f"\n===== Category: {category} ({len(part)} reviews) =====")
            run_topic_model(part.reset_index(drop=True), out / slug(category), args)


if __name__ == "__main__":
    main()