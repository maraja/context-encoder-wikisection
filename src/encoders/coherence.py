import torch
from src.bertkeywords.src.similarities import Embedding, Similarities
from src.bertkeywords.src.keywords import Keywords
from src.dataset.utils import dedupe_list


class Coherence:
    def __init__(self, max_words_per_step=2, coherence_threshold=0.4):
        self.max_words_per_step = max_words_per_step
        self.coherence_threshold = coherence_threshold
        similarities_lib = Similarities("bert-base-uncased")

        self.keywords_lib = Keywords(
            similarities_lib.model, similarities_lib.tokenizer)
        self.embedding_lib = Embedding(
            similarities_lib.model, similarities_lib.tokenizer)

    def get_identical_coherent_words(self, sentence1, sentence2, coherence_threshold):
        kw_sentence2 = self.keywords_lib.get_keywords(sentence2)
        kw_sentence1 = self.keywords_lib.get_keywords(sentence1)

        coherent_words = []

        for word2 in kw_sentence2:
            for word1 in kw_sentence1:
                word1_text = word1[0]
                word2_text = word2[0]
                if word1_text == word2_text:
                    # check similarity and add to coherent dictionary
                    emb1 = self.embedding_lib.get_word_embedding(
                        sentence1, word1_text)
                    emb2 = self.embedding_lib.get_word_embedding(
                        sentence2, word2_text)
                    similarity = torch.cosine_similarity(
                        emb1.reshape(1, -1), emb2.reshape(1, -1)
                    )

                    if similarity[0] >= coherence_threshold:
                        # append the tuple with the embedding for each word that's similar
                        coherent_words.append((word1[0], word1[1], emb1))
                        coherent_words.append((word2[0], word2[1], emb2))

        return coherent_words

    def get_similar_coherent_words(self, sentence1, sentence2, coherence_threshold):
        kw_sentence2 = self.keywords_lib.get_keywords(sentence2)
        kw_sentence1 = self.keywords_lib.get_keywords(sentence1)

        coherent_words = []

        for word2 in kw_sentence2:
            for word1 in kw_sentence1:
                word1_text = word1[0]
                word2_text = word2[0]

                # check similarity and add to coherent dictionary
                emb1 = self.embedding_lib.get_word_embedding(
                    sentence1, word1_text)
                emb2 = self.embedding_lib.get_word_embedding(
                    sentence2, word2_text)
                similarity = torch.cosine_similarity(
                    emb1.reshape(1, -1), emb2.reshape(1, -1)
                )

                if similarity[0] >= coherence_threshold:
                    # append the tuple with the embedding for each word that's similar
                    coherent_words.append((word1[0], word1[1], emb1))
                    coherent_words.append((word2[0], word2[1], emb2))

        return coherent_words

    def get_coherence(
        self,
        segment,
        word_sim: float = 1
    ):
        """creates a list of words that are common and strong in a segment.

        Args:
            segments (list[str]): a segment of sentences to get keywords and collect similar ones on
            word_sim (float): If this number is anything less than one, look for similar words higher than the provided value. Otherwise look for only identical words

        Returns:
            list: list of words that are considered high coherence in the segment
        """
        coherence = []
        prev_sentence = None
        for sentence in segment:
            if prev_sentence is None:
                prev_sentence = sentence
                continue
            else:
                if (word_sim == 1):
                    coherent_words = self.get_identical_coherent_words(prev_sentence, sentence, self.coherence_threshold)[
                        :self.max_words_per_step
                    ]
                else:
                    coherent_words = self.get_similar_coherent_words(prev_sentence, sentence, word_sim)[
                        :self.max_words_per_step
                    ]
                coherence.extend(coherent_words)
                prev_sentence = sentence

        return coherence

    def get_coherence_map(
        self,
        segments,
    ):
        coherence_map = []
        for segment in segments:
            coherence_map.append(
                self.get_coherence(segment)
            )

        return coherence_map
