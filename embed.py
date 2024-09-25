import sys
import os
import numpy as np
import pickle
from math import ceil
from embedding_utils import *

model_dict = {
    "mol2vec": get_mol2vec,
    "graph2vec": get_graph2vec,
    "chemberta": get_chemberta,
    "cddd": get_cddd,
    "macaw": get_macaw,
    "molformer": get_molformer,
    "gpt2": get_gpt2,
    "bert": get_bert,
    "mte": lambda smiles_list: get_mte(f"MTE/embeddings/{file_name}.npz", smiles_list),
    "mat": get_mat,
    "rmat": get_rmat
}

def get_embedding_list(smiles_list, batch_size=500):
    get_embedding = model_dict[model_name]
    embedding_list = []
    n = len(smiles_list)
    n_batches = ceil(n / batch_size)
    for i in range(0, n, batch_size):
        print(f"\rBatch {i // batch_size + 1} / {n_batches}", end='')
        try:
            real_batch_size = min(batch_size, n - i)
            if model_name == "macaw" and real_batch_size < 100:
                embedding = get_embedding(smiles_list[-min(n, 100):])
                embedding = embedding[-real_batch_size:]
            else:
                embedding = get_embedding(smiles_list[i:i+batch_size])
        except Exception as e:
            print(f"\nError: {e}")
            return np.array(())
        embedding_list.extend(embedding)
    embedding_list = np.array(embedding_list)
    return embedding_list

def main():
    global file_path, file_name, model_name
    file_path = sys.argv[1]
    model_name = sys.argv[2]
    file_name = os.path.basename(file_path).split('.')[0]

    smiles_list = np.genfromtxt(file_path, dtype=str, delimiter='\n', comments=None)[1:]
    print(f"Loaded compounds: {len(smiles_list)}")

    batch_size = 500
    if model_name in ("mat", "rmat"):
        batch_size = 100
    embedding_list = get_embedding_list(smiles_list, batch_size)
    if not embedding_list.size:
        return

    print(f"\nGenerated embedding: {embedding_list.shape[0]} x {embedding_list.shape[1]}")
    save_path = f"embedding/embedding_{file_name}_{model_name}.pkl"
    with open(save_path, "wb") as file:
        pickle.dump(embedding_list, file)
    print(f"Saved embedding to {save_path}")

if __name__ == "__main__":
    main()