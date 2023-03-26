import torch
from src.bertkeywords.src.similarities import Embedding, Similarities
from src.bertkeywords.src.keywords import Keywords

class Coherence:
    def __init__(self):
        self.keywords_lib = Keywords()

        similarities_lib = Similarities("bert-base-uncased")
        self.embedding_lib = Embedding(similarities_lib.model, similarities_lib.tokenizer)

    def get_coherent_words(self, sentence1, sentence2, coherence_threshold):
        kw_sentence2 = self.keywords_lib.get_keywords(sentence2)
        kw_sentence1 = self.keywords_lib.get_keywords(sentence1)

        coherent_words = []

        for word2 in kw_sentence2:
            for word1 in kw_sentence1:
                word1_text = word1[0]
                word2_text = word2[0]
                if word1_text == word2_text:
                    # check similarity and add to coherent dictionary
                    emb1 = self.embedding_lib.get_word_embedding(sentence1, word1_text)
                    emb2 = self.embedding_lib.get_word_embedding(sentence2, word2_text)
                    similarity = torch.cosine_similarity(
                        emb1.reshape(1, -1), emb2.reshape(1, -1)
                    )

                    if similarity[0] >= coherence_threshold:
                        coherent_words.append(word1_text)

        return coherent_words

    def build_coherence_map(
        self,
        segments: list[list[str]], 
        max_words_per_step=2, 
        coherence_threshold=0.4
    ):
        """creates a list of words that are common and strong in a segment.

        Args:
            segments (list[list[str]]): text segments to get keywords and collect similar ones on
            max_words_per_step (int, optional): every step (every 2 sentences), how many common words to collect. Defaults to 2.
            coherence_threshold (float, optional): threshold for similarity between two words that are the same. Defaults to 0.4.

        Returns:
            list: list of words that are considered high coherence in the segment
        """
        coherence_map = []
        prev_segment = None
        for segment in segments:
            if prev_segment is None:
                prev_segment = segment
                continue
            else:
                coherence_map.extend(
                    self.get_coherent_words(prev_segment, segment, coherence_threshold)[
                        :max_words_per_step
                    ]
                )
                prev_segment = segment

        return coherence_map