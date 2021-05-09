from collections import Counter
import numpy as np
import string
from stop_words import get_stop_words

# Toggle use of filtering words from stop_words library
filter_stop_words = True

def get_p(freq, total):
    return freq / total

class NaiveBayesClassifier:
    alphabet = set(string.ascii_lowercase)
    generic_words = set()
    if filter_stop_words:
        generic_words = set(get_stop_words('en'))

    def __init__(self, smoothing_factor=1):
        self.smoothing_factor = smoothing_factor

        self.vocabulary = set()
        self.categories = set()

        self.document_count = 0
        self.category_freq = Counter()
        self.category_p = {}

        self.freq_table = {}
        self.p_table = {}
        self.p_missing_word = {}

    def reset(self):
        self.categories = set()

        self.document_count = 0
        self.category_freq = Counter()
        self.category_p = {}

        self.freq_table = {}
        self.p_table = {}
        self.p_missing_word = {}

    def is_a_word(self, text: str):
        if text in NaiveBayesClassifier.generic_words:
            return False

        num_letters = sum(1 for char in text if char in NaiveBayesClassifier.alphabet)
        if num_letters <= 1:
            return False

        return True

    def gen_vocabulary(self, documents: list):
        self.vocabulary = set(word for doc in documents for word in doc if self.is_a_word(word))

    def get_freq_of_vocab_words_in_category(self, x: list, y: list, category):
        indices = [i for i, c in enumerate(y) if c == category]
        vocab_words = [word for i in indices for word in x[i] if word in self.vocabulary]
        return Counter(vocab_words)

    def num_words_in_category(self, category):
        return sum(self.freq_table[category].values())

    def smooth_total(self, total):
        return total + (len(self.vocabulary) * self.smoothing_factor)

    # Used for Task 1
    def train_classifier(self, train_x: list, train_y: list, vocab_from_train=True):
        self.reset()

        # use the training set to create the vocabulary if needed
        if vocab_from_train:
            self.gen_vocabulary(train_x)
        # all the unique expected values is the possible labels for the data
        self.categories = set(train_y)

        # 1.1: store frequency and probabilities of category over all training documents
        self.document_count = len(train_x)
        self.category_freq = Counter(train_y)

        self.category_p = {c: get_p(freq, self.document_count) for c, freq in self.category_freq.items()}

        # 2.1: store table of vocabulary word frequency per category
        for c in self.categories:
            self.freq_table[c] = self.get_freq_of_vocab_words_in_category(train_x, train_y, c)

            total_with_smoothing = self.smooth_total(self.num_words_in_category(c))
            self.p_table[c] = {word: get_p(freq + self.smoothing_factor, total_with_smoothing)
                               for word, freq in self.freq_table[c].items()}
            self.p_missing_word[c] = self.smoothing_factor / total_with_smoothing

    # ----

    def p_category_legacy(self, category):
        return self.category_freq[category] / self.document_count

    def p_word_in_category_legacy(self, word, category):
        freq_with_smoothing = self.freq_table[category][word] + self.smoothing_factor
        total_with_smoothing = self.num_words_in_category(category) + (self.smoothing_factor * len(self.vocabulary))
        return freq_with_smoothing / total_with_smoothing

    def p_category(self, category):
        return self.category_p[category]

    def p_word_in_category(self, word, category):
        # 0 probabilities aren't stored, so just calculate the probability
        return self.p_table[category][word] if word in self.p_table[category] else self.p_missing_word[category]

    # Used for Task 2
    def score_doc_for_category(self, doc: list, category):
        log_p_category = np.log(self.p_category(category))
        vocab_words_in_doc = [word for word in doc if word in self.vocabulary]
        sum_log_p_word_in_category = sum(np.log(self.p_word_in_category(word, category)) for word in vocab_words_in_doc)
        return log_p_category + sum_log_p_word_in_category

    def classify(self, doc: list):
        category_scores = {c: self.score_doc_for_category(doc, c) for c in self.categories}
        return max(category_scores, key=category_scores.get)

    # ----

    # Used for Task 3
    def classify_all(self, docs: list):
        return [self.classify(d) for d in docs]
