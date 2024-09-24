# Dense molecular representations for efficient similarity search

## Project objective
This project is about finding a dense, low-dimensional molecular representation with an adjusted distance measure that could be an alternative for Morgan fingerprint with Tanimoto similarity. The key motivation is to be able to create a vector database of compounds, with fast similarity search and clustering procedures giving comparable results to traditional fingerprint approach.

## Usage
Within the project several embedding generation approaches where compared from NLP transformers to data-driven descriptors. To prepare the models run
```bash
chmod +x setup.sh
./setup.sh
```
This will clone the necessary repositories and create a docker container for CDDD REST. **It will increase the repository size by about 900 MB.**
To generate embedding use **embed.sh** script.
```bash
chmod +x embed.sh
./embed.sh smiles.txt cddd
```
The embedding is saved as a pickled array of arrays in **embedding** directory.

## Requirements
- Docker
- Python 3 (version used: 3.12.2) \
Necessary packages are listed in **requirements.txt**.

## Code references
- Mol2vec, [https://github.com/samoturk/mol2vec](https://github.com/samoturk/mol2vec)
- Graph2vec, [https://github.com/soumavaghosh/graph2vec](https://github.com/soumavaghosh/graph2vec)
- ChemBERTa, [https://huggingface.co/DeepChem/ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)
- Continuous and Data-Driven Descriptors, [https://github.com/jrwnter/cddd](https://github.com/jrwnter/cddd)
- Continuous and Data-Driven Descriptors REST, [https://github.com/vaxherra/cddd_rest](https://github.com/vaxherra/cddd_rest)
- Molecular AutoenCoding Auto-Workaround, [https://github.com/LBLQMM/MACAW](https://github.com/LBLQMM/MACAW)
- MolFormer, [https://huggingface.co/ibm/MoLFormer-XL-both-10pct](https://huggingface.co/ibm/MoLFormer-XL-both-10pct)
- MTE, [https://github.com/mpcrlab/MolecularTransformerEmbeddings](https://github.com/mpcrlab/MolecularTransformerEmbeddings)
- GPT2, [https://huggingface.co/entropy/gpt2_zinc_87m](https://huggingface.co/entropy/gpt2_zinc_87m)
- BERT, [https://huggingface.co/unikei/bert-base-smiles](https://huggingface.co/unikei/bert-base-smiles)
- MAT and R-MAT, [https://github.com/gmum/huggingmolecules](https://github.com/gmum/huggingmolecules)