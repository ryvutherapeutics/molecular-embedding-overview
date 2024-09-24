import sys
import numpy as np
import pickle
from math import ceil
from embedding_utils import *

model_dict = {
    # "mol2vec": get_mol2vec,
    # "graph2vec": get_graph2vec,
    "chemberta": get_chemberta,
    "cddd": get_cddd,
    "macaw": get_macaw,
    "molformer": get_molformer,
    "gpt2": get_gpt2,
    "bert": get_bert,
    "mte": lambda smiles_list: get_mte(f"MTE/embeddings/{file_name}.npz", smiles_list)
}

def get_embedding_list(smiles_list, model_name, batch_size=20):
    get_embedding = model_dict[model_name]
    embedding_list = []
    n = len(smiles_list)
    n_batches = ceil(n / batch_size)
    for i in range(0, n, batch_size):
        print(f"\rBatch {i // batch_size + 1} / {n_batches}", end='')
        embedding = get_embedding(smiles_list[i:i+batch_size])
        embedding_list.extend(embedding)
    embedding_list = np.array(embedding_list)
    return embedding_list

def main():
    file_path = sys.argv[1]
    model_name = sys.argv[2]
    global file_name
    file_name = file_path.split('.')[0]

    smiles_list = np.genfromtxt(file_path, dtype=str, delimiter='\n', comments=None)
    print(f"Loaded compounds: {len(smiles_list)}")

    embedding_list = get_embedding_list(smiles_list, model_name)
    print(f"\nGenerated embedding: {embedding_list.shape[0]} x {embedding_list.shape[1]}")
    save_path = f"embedding1/embedding_{file_name}_{model_name}.pkl"
    with open(save_path, "wb") as file:
        pickle.dump(embedding_list, file)
    print(f"Saved embedding to {save_path}")

if __name__ == "__main__":
    main()