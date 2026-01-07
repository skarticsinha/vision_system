import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D numpy arrays
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
