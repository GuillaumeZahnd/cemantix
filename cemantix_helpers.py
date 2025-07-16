import numpy as np
import matplotlib.pyplot as plt


def calculate_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding_vector(word, embeddings):
    return np.array(embeddings.embed_documents([word])[0])


def calculate_relative_position(corpus_scores, candidate_score):
    return (corpus_scores > candidate_score).sum() +1


def plot_scores(corpus_scores, candidate_score, position, query, corpus, number_of_attempts, number_of_closest_words):

    if position <= len(corpus_scores):

        reply = "[{}] Query: {} | Position: {}/{} | Score: {:.3f}\n".format(
            number_of_attempts, query, position, number_of_closest_words, candidate_score)

        if query in corpus:
            scores = corpus_scores
        else:
            scores = np.concatenate((corpus_scores, np.array((candidate_score,))))
            scores = np.sort(scores)[::-1]
            scores = scores[:number_of_closest_words]

        plt.figure()
        plt.title(reply)
        plt.plot(np.arange(1, number_of_closest_words +1), scores, color="steelblue", zorder=0)
        plt.scatter(position, candidate_score, color="crimson", zorder=1)
        plt.ylabel("Cosine similarity")
        plt.xlabel("Corpus")
        plt.show()

    else:
        reply = "[{}] Query: {} | Position: >{} | Score: {:.3f}\n".format(
            number_of_attempts, query, number_of_closest_words, candidate_score)

    print(reply)
