"""
Task 2: Preprocessing Social Media Posts
ML Home Assignment - COVID-19 Fake News Detection

Pipeline:
  1. Lowercase text
  2. Convert emojis to text descriptions using unicode handling
  3. Extract and clean hashtags (keep the word, remove #)
  4. Remove URLs
  5. Remove mentions (@user)
  6. Remove punctuation and special characters (carefully)
  7. Tokenize
  8. Remove stopwords
  9. Lemmatize
  10. Rejoin tokens into clean string

Output: Preprocessed train/val/test CSVs saved to data/splits/
"""

import os
import re
import string
import unicodedata
import pandas as pd
import nltk

# Download required NLTK data
nltk.download('stopwords',    quiet=True)
nltk.download('wordnet',      quiet=True)
nltk.download('punkt',        quiet=True)
nltk.download('omw-1.4',      quiet=True)
nltk.download('punkt_tabset', quiet=True)

from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
SPLITS_DIR      = "data/splits"
PROCESSED_DIR   = "data/processed"
TEXT_COLUMN     = "tweet"        # <-- update if your column name differs
LABEL_COLUMN    = "label"

STOP_WORDS      = set(stopwords.words('english'))
LEMMATIZER      = WordNetLemmatizer()

# ─────────────────────────────────────────────────────────────────────
# EMOJI HANDLING  (no external emoji library — pure unicode)
# ─────────────────────────────────────────────────────────────────────
def demojize(text: str) -> str:
    """
    Convert emoji characters to their Unicode name descriptions.
    E.g., 😷 -> 'wearing_face_mask'
    This preserves the semantic meaning of emojis without external libs.
    """
    result = []
    for char in text:
        if unicodedata.category(char) in ('So', 'Sm'):   # Symbol, Other / Math
            name = unicodedata.name(char, '').lower()
            if name:
                name = name.replace(' ', '_')
                result.append(f" {name} ")
            # else skip unrecognised symbols
        else:
            result.append(char)
    return ''.join(result)


# ─────────────────────────────────────────────────────────────────────
# CORE PREPROCESSING STEPS
# ─────────────────────────────────────────────────────────────────────
def lowercase(text: str) -> str:
    """Step 1: Lowercase the text."""
    return text.lower()


def handle_emojis(text: str) -> str:
    """Step 2: Convert emojis to descriptive text."""
    return demojize(text)


def handle_hashtags(text: str) -> str:
    """
    Step 3: Extract hashtag text (keep the word, remove the # symbol).
    E.g., #StayHome -> StayHome
    Important: hashtags often carry stance/sentiment so we keep the word.
    """
    return re.sub(r'#(\w+)', r'\1', text)


def remove_urls(text: str) -> str:
    """Step 4: Remove URLs (http, https, www)."""
    text = re.sub(r'http\S+|https\S+|www\.\S+', '', text)
    return text


def remove_mentions(text: str) -> str:
    """Step 5: Remove @mentions."""
    return re.sub(r'@\w+', '', text)


def remove_special_characters(text: str) -> str:
    """
    Step 6: Remove punctuation and special characters.
    Keep alphanumeric, spaces, and underscore (for emoji descriptions).
    """
    # Keep letters, digits, spaces, underscores
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list:
    """Step 7: Tokenize into words."""
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """
    Step 8: Remove English stopwords.
    NOTE: We keep negation words (not, no, nor, never) because they
    can flip the meaning of a post (fake vs real).
    """
    negation_words = {'not', 'no', 'nor', 'never', 'neither',
                      'nobody', 'nothing', 'nowhere', 'without'}
    return [
        token for token in tokens
        if token not in STOP_WORDS or token in negation_words
    ]


def lemmatize(tokens: list) -> list:
    """Step 9: Lemmatize tokens to their base form."""
    return [LEMMATIZER.lemmatize(token) for token in tokens]


def rejoin(tokens: list) -> str:
    """Step 10: Rejoin tokens into a single string."""
    return ' '.join(tokens)


# ─────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline for a single social media post.

    Steps:
        1.  Lowercase
        2.  Emoji -> text
        3.  Hashtag cleaning
        4.  URL removal
        5.  Mention removal
        6.  Special character removal
        7.  Tokenize
        8.  Stopword removal (preserving negations)
        9.  Lemmatize
        10. Rejoin
    """
    if not isinstance(text, str):
        return ""

    text = lowercase(text)
    text = handle_emojis(text)
    text = handle_hashtags(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_special_characters(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return rejoin(tokens)


# ─────────────────────────────────────────────────────────────────────
# APPLY TO DATAFRAME
# ─────────────────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame, text_col: str = TEXT_COLUMN) -> pd.DataFrame:
    """Apply preprocessing pipeline to all rows in a DataFrame."""
    df = df.copy()
    df[text_col] = df[text_col].apply(preprocess)
    return df


# ─────────────────────────────────────────────────────────────────────
# LOAD, PROCESS, SAVE
# ─────────────────────────────────────────────────────────────────────
def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(SPLITS_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {name}.csv — {len(df)} rows")
    return df


def save_processed(df: pd.DataFrame, name: str):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[INFO] Saved preprocessed {name}.csv -> {path}")


# ─────────────────────────────────────────────────────────────────────
# DEMO / SANITY CHECK
# ─────────────────────────────────────────────────────────────────────
def demo():
    """Quick sanity check on example social media posts."""
    samples = [
        "If you take Crocin thrice a day you are safe. #COVID19 #FakeNews",
        "Wearing mask can protect you from the virus 😷 #StayHome @WHO",
        "Check this out: https://t.co/example NO this is NOT true!!!",
        "Scientists say vaccine is 95% effective 💉💪 #Vaccine #COVID",
    ]
    print("\n" + "=" * 60)
    print("  DEMO: Preprocessing Examples")
    print("=" * 60)
    for s in samples:
        print(f"\n  Original  : {s}")
        print(f"  Processed : {preprocess(s)}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TASK 2: Preprocessing Social Media Posts")
    print("=" * 60)

    # Run demo first
    demo()

    # Process all three splits
    for split_name in ["train", "val", "test"]:
        df = load_split(split_name)
        df = preprocess_dataframe(df, text_col=TEXT_COLUMN)
        save_processed(df, split_name)

    print("\n[DONE] Task 2 Complete! ✓")
    print("=" * 60)