import numpy as np
import pandas as pd
import tensorflow as tf
from vast_utils import build_model
from model import vast
import torch

class MetricType:
    """Supported metric types"""
    # Cosine similarity
    COS = "COS"
    # L2 distance
    L2 = "L2"

_CSV_RUN_ID_FIELD = "run_id"
_CSV_START_TIME_FIELD = "start_time"
_CSV_END_TIME_FIELD = "end_time"
_CSV_LABEL_FIELD = "label"
_CSV_CATEGORIES_FIELD = "categories"

def _l2_normalize(mat):
    """Helper that L2 normalizes each row of mat"""
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)

def _get_similarity_matrix(mat1: np.array, mat2: np.array, metric: str,
                           standardize: bool) -> np.array:
    """Returns a similarity / distance matrix using metric between mat1 and mat2.
    Entry at (i,j) is the similarity / distance between row i in mat1 and row j in mat2.
    """
    if metric == MetricType.COS:
        # normalize so dot product is cosine similarity
        x = _l2_normalize(mat1)
        y = _l2_normalize(mat2)
        return np.dot(x, y.T)
    if metric == MetricType.L2:
        x = np.reshape(mat1, (mat1.shape[0], 1, mat1.shape[1]))
        y = np.reshape(mat2, (1, mat2.shape[0], mat2.shape[1]))
        ret = np.sqrt(np.sum((x - y)**2, axis=2))
        if standardize:
            ret = (ret - np.mean(ret)) / np.std(ret)
        return ret
    raise ValueError(
        f"Invalid similarity metric {metric}. Please use one of COS, L2")


def _eval_similarity(model: vast.VAST, data: pd.DataFrame, metric: str, standardize: bool):
    """Constructs similarity/distance matrix between the embeddings for ALL scene-label pairs
    in data.
    """
    n = data.shape[0]
    labels = []
    video_paths = []
    for _, row in data.iterrows():
        # get labels
        label = row[_CSV_LABEL_FIELD]
        labels.append(label)

        # get scene video paths
        blob_id = row[_CSV_RUN_ID_FIELD]
        start_ts = row[_CSV_START_TIME_FIELD]
        end_ts = row[_CSV_END_TIME_FIELD]
        name = f"{blob_id}-{start_ts}-{end_ts}.mp4"
        path = f"/home/dross/videos/{name}"
        video_paths.append(path)
    
    # get embeddings
    print(f"calculating {len(video_paths)} video embeddings")
    val_embeddings = model.get_vid_embeddings(video_paths)
    print(f"calculating {len(labels)} text embeddings")
    query_embeddings = model.get_text_embeddings(labels)

    # get similarity matrix
    similarity = _get_similarity_matrix(np.array(val_embeddings),
                                        np.array(query_embeddings), metric,
                                        standardize)
    np.set_printoptions(precision=2, suppress=True)
    print(f"Similarity matrix for metric {metric}:")
    print(str(similarity))

    # calc loss (only relevant for cosine similarity)
    if metric == MetricType.COS:
        labels = np.arange(n)
        loss_b = np.average(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=similarity).numpy())
        loss_t = np.average(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=similarity.T).numpy())
        avg_loss = (loss_b + loss_t) / 2

        print(f"Loss over embeddings: {loss_b}")
        print(f"Loss over captions: {loss_t})")
        print(f"Avg loss: {avg_loss}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    df = pd.read_csv("/home/dross/TSS_eval_small_manual.csv", delimiter="|")
    model = build_model.build_model_fn('/home/dross/retrieval-train_all_pruned')
    _eval_similarity(model, df, MetricType.COS, True)
