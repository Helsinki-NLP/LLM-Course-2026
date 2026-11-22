from stqdm import stqdm
import pandas as pd
import numpy as np
import torch

def embed_chunks(pages_and_chunks: list[dict], embedding_model):
    # Embed each chunk one by one
    for item in stqdm(pages_and_chunks):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

def save_embeddings(pages_and_chunks: list[dict]) -> str:
    # Save embeddings to file
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)
    # TODO: change file name to be unique to avoid clashing with other files
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

    return embeddings_df_save_path

# load embeddings into Tensor
def embeddings_to_tensor(filename: str) -> tuple[torch.Tensor, dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import texts and embedding df
    text_chunks_and_embedding_df = pd.read_csv(filename)

    # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" "))

    ## Convert texts and embedding df to a list of dicts
    #pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

    return embeddings, text_chunks_and_embedding_df.to_dict('records')