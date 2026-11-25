"""LDA topic modeling for Daraz reviews, plus topic-vs-sentiment analysis(ailey lai skipped)

Run from the project root:
    python src/topic_modeling.py
    python src/topic_modeling.py --num-topics 6      # skip the search, force k
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # draw into image files only, no window
import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel,TfidfModel

from data_loader import clean_text

from nltk.corpus import stopwords 

from gensim.models.phrases import Phrases, Phraser



ENGLISH_STOPWORDS = set(stopwords.words("english"))

# Words that carry no topic meaning. English + common romanized Nepali.
# EXTEND THIS LIST after looking at your first results.




NEPALI_ROMANIZED_STOPWORDS = set(
    """cha chha xa chaa ho hai ra ko ka ki le lai ma ni nai pani ta tara yo tyo hola mero timro hamro chai ani malai jasto maile aru ali ekdum""".split()
)

OPINION_STOPWORDS = set(
    """good nice best great awesome amazing excellent perfect perfectly love loved like liked well wow super superb decent pretty happy highly satisfied recommend recommended fine okay better bad worst poor awful terrible overall thank thanks thankyou""".split()
)

def load_stopwords_from_file(file_path):
    """Read a text file with one word per line and return the words as a set."""
    words = set()

    with open(file_path, encoding="utf-8") as file:
        for line in file:
            word = line.strip().lower()   # remove spaces/newline, make lowercase
            if word != "":                # skip empty lines
                words.add(word)

    return words


ROMANIZED_FILE_STOPWORDS = load_stopwords_from_file("data/NepaliStopWords_romanized.txt")
print(f"Loaded {len(ROMANIZED_FILE_STOPWORDS)} words from the file")

STOPWORDS = ENGLISH_STOPWORDS | OPINION_STOPWORDS | NEPALI_ROMANIZED_STOPWORDS | ROMANIZED_FILE_STOPWORDS 

def tokenize(text: str) -> list[str]:
    """Lowercase words only (letters), no stopwords, at least 3 characters."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def learn_bigrams(all_tokens, min_count=3, threshold=0.4):
    """Learn which pairs of neighbouring words should be joined.

    all_tokens is a list of reviews, and each review is a list of words.
    This step only LEARNS the pairs. It does not change the text yet.
    """
    phrase_model = Phrases(
        all_tokens,
        min_count=min_count,   # a pair must appear at least this many times
        threshold=threshold,   # how strongly the two words belong together
        scoring="npmi",        # score from -1 to 1 (1 = always together)
    )

    # Show what was learned, so you can check it makes sense
    found = phrase_model.find_phrases(all_tokens)
    print(f"Learned {len(found)} bigrams. Strongest ones:")
    ranked = sorted(found.items(), key=lambda item: item[1], reverse=True)
    for phrase, score in ranked[:30]:
        print(f"   {phrase}  (score {score:.2f})")

    # Phraser is a lighter copy that can only apply the pairs
    bigram_model = Phraser(phrase_model)
    return bigram_model


def apply_bigrams(review_tokens, bigram_model):
    return bigram_model[review_tokens]





def build_corpus(texts, min_docs=3, max_ratio=0.5):
    # Step 1: split every review into words (stopwords are removed here)
    tokens = []
    for text in texts:
        tokens.append(tokenize(text))

    # Step 2: learn the bigrams from ALL reviews
    bigram_model = learn_bigrams(tokens)

    # Step 3: join the pairs in every review
    tokens_with_bigrams = []
    for review_tokens in tokens:
        tokens_with_bigrams.append(apply_bigrams(review_tokens, bigram_model))
    tokens = tokens_with_bigrams

    # Step 4: give every word an ID and drop words that are too rare or too common
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=min_docs, no_above=max_ratio)

    # Step 5: turn each review into (word_id, count) pairs
    corpus = []
    for review_tokens in tokens:
        corpus.append(dictionary.doc2bow(review_tokens))

    return tokens, dictionary, corpus, bigram_model



def train_lda(corpus, dictionary, k, seed=42):
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=15,
        iterations=100,
        random_state=seed,
        alpha="auto",
        eta="auto",
    )


def coherence(model, tokens, dictionary):
    return CoherenceModel(
        model=model, texts=tokens, dictionary=dictionary, coherence="c_v"
    ).get_coherence()


def pick_best_k(corpus, dictionary, tokens, k_values, output_dir):
    """Try several topic counts and keep the one with the best coherence."""
    scores = {}
    for k in k_values:
        model = train_lda(corpus, dictionary, k)
        scores[k] = float(coherence(model, tokens, dictionary))
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

def save_ldavis(model, tokens, dictionary, path):
    """Interactive topic map (pyLDAvis). Open the saved HTML file in a browser."""
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis
    except ImportError:
        print("pyLDAvis not installed - skipping (pip install pyLDAvis).")
        return
    # Raw word counts, even when training used TF-IDF weights.
    counts = [dictionary.doc2bow(t) for t in tokens]
    vis = gensimvis.prepare(model, counts, dictionary,
                            mds="mmds", R=30, sort_topics=False)
    pyLDAvis.save_html(vis, str(path))
 
def remove_low_tfidf_words(corpus,dictionary, threshold): #(optionally applied, may be we will apply if dataset will be very very large)
    """Remove, from each review, the words that are too common to be informative.

    A word is removed from a review when its TF-IDF score in that review is
    below the threshold. Reviews are short, so scores are large: start with
    about 0.1, not the 0.03 used for long documents.
    """
    tfidf_model = TfidfModel(corpus)
    removed_counts={}

    new_corpus = []
    for bow in corpus:
        # tfidf_model[bow] gives (word_id, score) pairs for this one review.
        # Words that appear in every review score 0 and are left out.
        scores = {}
        for word_id, score in tfidf_model[bow]:
            scores[word_id] = score

        new_bow = []
        for word_id, count in bow:
            score = scores.get(word_id, 0.0)   # missing means a score of 0
            if score >= threshold:
                new_bow.append((word_id, count))
            else:
                word = dictionary[word_id]
                removed_counts[word] = removed_counts.get(word,0) + count 
        new_corpus.append(new_bow)
    most_removed = sorted(removed_counts.items(), key=lambda item: item[1], reverse=True)
    print("Words removed by the TF-IDF filter (most removed first):")
    for word, count in most_removed[:15]:
        print(f"   {word}: {count}")

    return new_corpus  


def main():
    parser = argparse.ArgumentParser(description="LDA topic modeling of reviews.")
    parser.add_argument("--data", default="data/dataset2.csv")
    parser.add_argument("--text-column", default="Review")
    parser.add_argument("--output-dir", default="results/topic_results")
    parser.add_argument("--num-topics", type=int, default=None,
                        help="Fix k. If omitted, the best k from 3-10 is chosen.")
    parser.add_argument("--tfidf-threshold", type=float, default=0.0,
                        help="Remove words with TF-IDF below this. 0 = off.")
    
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    data["text"] = data[args.text_column].map(clean_text)
    data = data[data["text"] != ""].drop_duplicates("text").reset_index(drop=True)

    tokens, dictionary, corpus, bigram_model = build_corpus(data["text"])
    if args.tfidf_threshold > 0:
        corpus = remove_low_tfidf_words(corpus,dictionary ,args.tfidf_threshold)

    
    
    # Keep only reviews that still have words after filtering.
    keep = [i for i, bow in enumerate(corpus) if len(bow) > 0]
    print(f"Reviews: {len(data)} | usable after filtering: {len(keep)} | "
          f"vocabulary: {len(dictionary)}")
    data = data.iloc[keep].reset_index(drop=True)
    tokens = [tokens[i] for i in keep]
    corpus = [corpus[i] for i in keep]

    if args.num_topics:
        k = args.num_topics
        scores = {}
    else:
        print("Searching for the best number of topics...")
        k, scores = pick_best_k(corpus, dictionary, tokens, range(3, 11), out)
    print(f"Using k = {k}")

    model = train_lda(corpus, dictionary, k)
    final_coherence = float(coherence(model, tokens, dictionary))

    # 1) Top words per topic -> you manually label these in your report.
    topics = {}
    for topic_id in range(k):
        words = [w for w, _ in model.show_topic(topic_id, topn=10)]
        topics[topic_id] = words
        print(f"Topic {topic_id}: {', '.join(words)}")

    # 2) Dominant topic for each review.
    def dominant(bow):
        dist = model.get_document_topics(bow, minimum_probability=0)
        return max(dist, key=lambda x: x[1])

    best = [dominant(b) for b in corpus]
    data["topic"] = [t for t, _ in best]
    data["topic_prob"] = [p for _, p in best]
    data[["text", "topic", "topic_prob"]].to_csv(
        out / "reviews_with_topics.csv", index=False
    )

    # # 3) Topic x sentiment table + heatmap (the "topic-sentiment cross-tab").
    # order = [s for s in ["negative", "neutral", "positive"] if s in set(data["sentiment"])]
    # crosstab = pd.crosstab(data["topic"], data["sentiment"])[order]
    # crosstab.to_csv(out / "topic_sentiment_crosstab.csv")
    # share = crosstab.div(crosstab.sum(axis=1), axis=0)

    # fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    # axes[0].set_title("Reviews per topic and sentiment")
    # sns.heatmap(share, annot=True, fmt=".0%", cmap="RdYlGn", ax=axes[1])
    # axes[1].set_title("Sentiment share within each topic")
    # plt.tight_layout()
    # plt.savefig(out / "topic_sentiment_heatmap.png", dpi=160)
    # plt.close()

    # 4) Word cloud per topic (optional; needs `pip install wordcloud`).
    try:
        from wordcloud import WordCloud

        cols = min(k, 3)
        rows = (k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
        axes = [axes] if k == 1 else list(pd.Series(axes.ravel() if hasattr(axes, "ravel") else axes))
        for topic_id, ax in enumerate(axes):
            ax.axis("off")
            if topic_id < k:
                freqs = dict(model.show_topic(topic_id, topn=30))
                ax.imshow(WordCloud(width=500, height=320, background_color="white")
                          .generate_from_frequencies(freqs))
                ax.set_title(f"Topic {topic_id}")
        plt.tight_layout()
        plt.savefig(out / "topic_wordclouds.png", dpi=140)
        plt.close()
    except ImportError:
        print("wordcloud not installed - skipping word clouds.")

    save_ldavis(model, tokens, dictionary, out / "ldavis.html")
        
    # 5) Summary file.
    summary = {
        "num_topics": k,
        "coherence_c_v": final_coherence,
        "coherence_search": scores,
        "topics_top_words": topics,
        "reviews_per_topic": data["topic"].value_counts().sort_index().to_dict(),
    }
    (out / "topics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model.save(str(out / "lda_model"))
    bigram_model.save(str(out/"bigram.model"))
    print(f"\nCoherence (c_v): {final_coherence:.4f}")
    print(f"Saved everything to: {out}")


if __name__ == "__main__":
    main()