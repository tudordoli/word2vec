import argparse
import collections
import math
import numpy as np
import random
import sys
from typing import List, Tuple, Dict


def read_corpus(path: str) -> List[str]:
    with open(path, "r", encoding="utf8") as f:
        text = f.read().lower()
    for ch in '".,!?;:\n\r"()[]{}':
        text = text.replace(ch, " ")
    tokens = text.split()
    return tokens


def build_vocab(
    tokens: List[str], min_count: int = 1
) -> Tuple[Dict[str, int], Dict[int, str], np.ndarray]:
    freq = collections.Counter(tokens)
    vocab = [w for w, c in freq.items() if c >= min_count]
    vocab.sort(key=lambda w: -freq[w])
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    counts = np.array([freq[w] for w in vocab], dtype=np.int64)
    return word2idx, idx2word, counts


def subsample(
    tokens: List[str], word2idx: Dict[str, int], counts: np.ndarray, t: float = 1e-5
) -> List[int]:
    N = sum(counts)
    total_counts = counts.sum()
    prob_drop = {}
    for w, i in word2idx.items():
        f = counts[i] / total_counts
        p_keep = (math.sqrt(f / t) + 1) * (t / f)
        prob_drop[i] = min(1.0, p_keep)
    result = []
    for w in tokens:
        if w not in word2idx:
            continue
        i = word2idx[w]
        if random.random() < prob_drop[i]:
            result.append(i)
    return result


def generate_skipgram_pairs(
    indices: List[int], window_size: int
) -> List[Tuple[int, int]]:
    pairs = []
    for center_pos, center_word in enumerate(indices):
        start = max(0, center_pos - window_size)
        end = min(len(indices), center_pos + window_size + 1)
        for ctx_pos in range(start, end):
            if ctx_pos == center_pos:
                continue
            pairs.append((center_word, indices[ctx_pos]))
    return pairs


def make_unigram_table(counts: np.ndarray, power: float = 0.75):
    probs = counts.astype(np.float64) ** power
    probs /= probs.sum()
    return probs


def sigmoid(x):
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(x, dtype=np.float64)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out


def train_sgns(
    pairs: List[Tuple[int, int]],
    counts: np.ndarray,
    vocab_size: int,
    embed_dim: int = 100,
    epochs: int = 1,
    lr: float = 0.025,
    neg_samples: int = 5,
    batch_size: int = 512,
    seed: int = 42,
):

    np.random.seed(seed)
    random.seed(seed)

    W_in = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
    W_out = np.zeros((vocab_size, embed_dim), dtype=np.float64)

    unigram_probs = make_unigram_table(counts)
    pairs_idx = np.arange(len(pairs))

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0
        for start in range(0, len(pairs), batch_size):
            end = min(len(pairs), start + batch_size)
            batch = pairs[start:end]
            for center, context in batch:
                v_c = W_in[center]
                u_o = W_out[context]

                s_pos = np.dot(u_o, v_c)
                sigma_pos = sigmoid(s_pos)
                loss_pos = -np.log(max(sigma_pos, 1e-10))
                total_loss += loss_pos

                g_pos = sigma_pos - 1.0

                grad_v = g_pos * u_o

                W_out[context] -= lr * (g_pos * v_c)

                negs = np.random.choice(
                    vocab_size, size=neg_samples, replace=True, p=unigram_probs
                )

                for neg in negs:
                    u_k = W_out[neg]
                    s_neg = np.dot(u_k, v_c)
                    sigma_neg = sigmoid(-s_neg)
                    loss_neg = -np.log(max(sigma_neg, 1e-10))
                    total_loss += loss_neg

                    g_neg = sigmoid(s_neg)
                    grad_v += g_neg * u_k
                    W_out[neg] -= lr * (g_neg * v_c)

                W_in[center] -= lr * grad_v

        avg_loss = total_loss / len(pairs)
        print(f"[Epoch {epoch+1}/{epochs}] avg loss per pair = {avg_loss:.6f}")

    return W_in, W_out


def nearest_neighbors(
    embeddings: np.ndarray,
    idx2word: Dict[int, str],
    word: str,
    word2idx: Dict[str, int],
    topk: int = 10,
):
    if word not in word2idx:
        print(f"'{word}' not in vocab.")
        return
    i = word2idx[word]
    v = embeddings[i]
    norms = np.linalg.norm(embeddings, axis=1)
    eps = 1e-8
    sims = embeddings.dot(v) / (norms * (np.linalg.norm(v) + eps) + eps)
    nearest = np.argsort(-sims)[1 : topk + 1]
    print(f"Nearest to '{word}':")
    for j in nearest:
        print(f"  {idx2word[j]} (sim={sims[j]:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default=None, help="Plain text file")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--neg", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--sample", action="store_true", help="Use simple built-in tiny sample"
    )
    args = parser.parse_args()

    if args.sample or args.corpus is None:
        print("Using built-in sample corpus")
        sample = "The quick brown fox jumped over the lazy dog. The dog barked. The fox ran away."
        tokens = sample.lower().split()
    else:
        tokens = read_corpus(args.corpus)

    word2idx, idx2word, counts = build_vocab(tokens, min_count=args.min_count)
    vocab_size = len(word2idx)
    print(f"Vocab size after min-count={args.min_count}: {vocab_size}")
    if vocab_size < 2:
        print("Vocabulary too small, increase data or lower min-count")
        sys.exit(1)

    indices = [word2idx[w] for w in tokens if w in word2idx]

    pairs = generate_skipgram_pairs(indices, window_size=args.window)
    print(f"Generated {len(pairs)} (center,context) pairs")

    W_in, W_out = train_sgns(
        pairs,
        counts=np.array([counts[word2idx[w]] for w in word2idx], dtype=np.int64),
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        lr=args.lr,
        neg_samples=args.neg,
        batch_size=args.batch_size,
    )

    print("\nSanity check of nearest neighbors")
    for q in list(word2idx.keys())[:5]:
        nearest_neighbors(W_in, idx2word, q, word2idx, topk=5)
        print("")


if __name__ == "__main__":
    main()
