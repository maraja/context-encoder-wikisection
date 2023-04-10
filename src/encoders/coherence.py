import torch
from src.bertkeywords.src.similarities import Embedding, Similarities
from src.bertkeywords.src.keywords import Keywords
from src.dataset.utils import dedupe_list, flatten, truncate_string, truncate_by_token


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
        coherence_threshold: float = 1
    ):
        """creates a list of words that are common and strong in a segment.

        Args:
            segments (list[str]): a segment of sentences to get keywords and collect similar ones on
            coherence_threshold (float): If this number is anything less than one, look for similar words higher than the provided value. Otherwise look for only identical words

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
                if (coherence_threshold == 1):
                    coherent_words = self.get_identical_coherent_words(prev_sentence, sentence, self.coherence_threshold)[
                        :self.max_words_per_step
                    ]
                else:
                    coherent_words = self.get_similar_coherent_words(prev_sentence, sentence, coherence_threshold)[
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

    def predict(self, 
                text_data, 
                max_tokens=256, 
                prediction_thresh=0.3, 
                pruning=4, 
                pruning_min=10
            ):
        # pruning = 4  # remove the lowest n important words from coherence map
        # pruning_min = 10  # only prune after n words in the coherence map
        coherence_map = []
        predictions = []
        for i, row in enumerate(text_data):
            # compare the current sentence to the previous one
            if i == 0:
                predictions.append((0,0)) # predict a 0 since it's the start
                pass
            else:
                prev_row = text_data[i - 1]

                row = truncate_by_token(row, max_tokens)
                prev_row = truncate_by_token(prev_row, max_tokens)

                # add the keywords to the coherence map
                coherence_map.extend(self.get_coherence([row, prev_row], coherence_threshold=self.coherence_threshold))
                if pruning > 0 and len(coherence_map) >= pruning_min:
                    sorted_map = sorted(
                        coherence_map, key=lambda tup: tup[1]
                    )  # sort asc by importance based on keybert
                    coherence_map = sorted_map[pruning:][
                        ::-1
                    ]  # get the last n - pruning values and reverse the list

                # get the keywords for the current sentences
                keywords_current = self.keywords_lib.get_keywords_with_embeddings(row)
                keywords_prev = self.keywords_lib.get_keywords_with_embeddings(prev_row)

                # compute the word comparisons between the previous (with the coherence map)
                # and the current (possibly the first sentence in a new segment)
                word_comparisons_with_coherence = self.embedding_lib.compare_keyword_tuples(
                    [*coherence_map, *keywords_prev], keywords_current
                )

                similarities_with_coherence = [
                    comparison[2] for comparison in word_comparisons_with_coherence
                ]
                avg_similarity_with_coherence = sum(similarities_with_coherence) / len(
                    similarities_with_coherence
                )

                # if the two sentences are similar, create a cohesive prediction
                # otherwise, predict a new segment
                if avg_similarity_with_coherence[0] > prediction_thresh:
                    # print(f"Label: {label}, Prediction: {0}")
                    predictions.append((avg_similarity_with_coherence[0], 0))
                else:
                    # start of a new segment, empty the map
                    coherence_map = []
                    # print(f"Label: {label}, Prediction: {1}")
                    predictions.append((avg_similarity_with_coherence[0], 1))
                
                print(".", end = '') 

        return predictions

    # testing functions
    def predict_verbose(self, text_data, max_tokens=256, prediction_thresh=0.3, pruning=4, pruning_min=10):
        coherence_map = []
        predictions = []
        for i, row in enumerate(text_data):
            # compare the current sentence to the previous one
            if i == 0:
                predictions.append((0,0)) # predict a 0 since it's the start
                pass
            else:
                prev_row = text_data[i - 1]

                row = truncate_by_token(row, max_tokens)
                prev_row = truncate_by_token(prev_row, max_tokens)

                # add the keywords to the coherence map
                coherence_map.extend(self.get_coherence([row, prev_row], coherence_threshold=self.coherence_threshold))
                print(f"Coherence Map before pruning: {[x[0] for x in coherence_map]}")
                if pruning > 0 and len(coherence_map) >= pruning_min:
                    sorted_map = sorted(
                        coherence_map, key=lambda tup: tup[1]
                    )  # sort asc by importance based on keybert
                    coherence_map = sorted_map[pruning:][
                        ::-1
                    ]  # get the last n - pruning values and reverse the list

                # get the keywords for the current sentences
                keywords_current = self.keywords_lib.get_keywords_with_embeddings(row)
                keywords_prev = self.keywords_lib.get_keywords_with_embeddings(prev_row)
                print(f"Coherence Map: {[x[0] for x in coherence_map]}, Keywords Current: {[x[0] for x in keywords_current]}")

                # compute the word comparisons between the previous (with the coherence map)
                # and the current (possibly the first sentence in a new segment)
                word_comparisons_with_coherence = self.embedding_lib.compare_keyword_tuples(
                    [*coherence_map, *keywords_prev], keywords_current
                )

                similarities_with_coherence = [
                    comparison[2] for comparison in word_comparisons_with_coherence
                ]
                avg_similarity_with_coherence = sum(similarities_with_coherence) / len(
                    similarities_with_coherence
                )

                # if the two sentences are similar, create a cohesive prediction
                # otherwise, predict a new segment
                if avg_similarity_with_coherence[0] > prediction_thresh:
                    print(f"Similarity: {avg_similarity_with_coherence[0]}, Prediction: {0}")
                    predictions.append((avg_similarity_with_coherence[0], 0))
                else:
                    # start of a new segment, empty the map
                    coherence_map = []
                    print(f"Similarity: {avg_similarity_with_coherence[0]}, Prediction: {1}")
                    predictions.append((avg_similarity_with_coherence[0], 1))

        return predictions
