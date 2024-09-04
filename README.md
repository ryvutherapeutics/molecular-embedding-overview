# Dense molecular representations for efficient similarity search

## Project objective
This project is about finding a dense, low-dimensional molecular representation with an adjusted distance measure that could be an alternative for Morgan fingerprint with Tanimoto similarity. The key motivation is to be able to create a vector database of compounds, with fast similarity search and clustering procedures giving comparable results to traditional fingerprint approach.

## Usage
Within the project several embedding generation approaches where compared from NLP transformers to data-driven descriptors. The code to use the models is available in the notebooks. There is also an example of preparing and testing a vector database. \
It is advised to install the packages listed in requirements.txt before use.
