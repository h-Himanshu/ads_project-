"""Convert a file of Devanagari words into English-letter spellings.

The input file has one Nepali word per line.
The output file has the English-letter spelling of each word, on the same line
number, so line 5 of the output belongs to line 5 of the input.

Run from the project root:
    python src/romanize_file.py
    python src/romanize_file.py --input data/NepaliStopWords.txt --output data/NepaliStopWords_romanized.txt

Needs one package:
    uv pip install indic-transliteration
"""

import argparse
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


def remove_accents(text):
    """Turn letters with marks (like 'ā' or 'ṭ') into plain letters ('a', 't')."""
    split_form = unicodedata.normalize("NFD", text)
    plain_letters = []
    for character in split_form:
        if not unicodedata.combining(character):
            plain_letters.append(character)
    return "".join(plain_letters)


def romanize(devanagari_word):
    """Convert one Devanagari word into English letters."""
    # Step 1: the library converts the word (the result still has accent marks)
    roman = transliterate(devanagari_word, sanscript.DEVANAGARI, sanscript.IAST)
    roman = roman.lower()

    # Step 2: the nasal sounds (ं and ँ) are written with 'n' in everyday typing
    roman = roman.replace("ṃ", "n")
    roman = roman.replace("~", "n")

    # Step 3: remove the accent marks
    roman = remove_accents(roman)
    return roman


def main():
    parser = argparse.ArgumentParser(description="Romanize a file of Nepali words.")
    parser.add_argument("--input", default="data/NepaliStopWords.txt")
    parser.add_argument("--output", default="data/NepaliStopWords_romanized.txt")
    args = parser.parse_args()

    # Read the input file, one word per line
    with open(args.input, encoding="utf-8") as input_file:
        lines = input_file.readlines()

    # Convert each line and write it to the output file
    count = 0
    with open(args.output, "w", encoding="utf-8") as output_file:
        for line in lines:
            word = line.strip()
            if word == "":
                continue
            output_file.write(romanize(word) + "\n")
            count += 1

    print(f"Converted {count} words.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
