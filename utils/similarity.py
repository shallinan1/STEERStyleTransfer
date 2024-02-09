from typing import List, Union
from sentence_transformers import SentenceTransformer
from torch.nn import CosineSimilarity

# Initialize constants and models
cos: CosineSimilarity = CosineSimilarity(dim=-1)
SENT_TRANS_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
sim_model = SentenceTransformer(SENT_TRANS_MODEL)

def compute_sim(original: Union[str, List[str]], 
                rewrites: Union[str, List[str]]) -> List[float]:
    """
    Compute the cosine similarity between embeddings of the original sentences and their rewrites.

    Args:
    original (Union[str, List[str]]): A string or list of strings representing the original sentences.
    rewrites (Union[str, List[str]]): A string or list of strings representing the rewritten sentences.

    Returns:
    List[float]: A list of cosine similarity scores between the original and rewritten sentences.
    """
    if not isinstance(original, list):
        original = [original]
    if not isinstance(rewrites, list):
        rewrites = [rewrites]
    assert len(original) == len(rewrites), "inputs are different lengths"

    outputs = []
    # Get embeddings of original and rewrites
    embedding_orig = sim_model.encode(original, convert_to_tensor=True, show_progress_bar=False)
    embedding_rew = sim_model.encode(rewrites, convert_to_tensor=True, show_progress_bar=False)   

    # Get similarities between each element in the lists to each other
    outputs = cos(embedding_orig, embedding_rew).tolist()

    return outputs


