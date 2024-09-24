#!/bin/bash

MODELS=("mol2vec" "graph2vec" "chemberta" "cddd" "mte" "macaw" "molformer" "gpt2" "bert" "mat" "rmat")

display_usage() {
  echo "How to use? $0 <smiles_file_path> <model_name>"
  echo "Models: ${MODELS[*]}"
  exit 1
}

if [ "$#" -ne 2 ]; then
  display_usage
fi

FILE_PATH=$1
MODEL_NAME=$2

if [ ! -e "$FILE_PATH" ]; then
  echo "ERROR File $FILE_PATH doesn't exist"
  exit 1
fi

if [[ ! " ${MODELS[@]} " =~ " ${MODEL_NAME} " ]]; then
  echo "ERROR Invalid model $MODEL_NAME"
  display_usage
fi

EMBEDDING_DIR="embedding1"
if [ ! -d "$EMBEDDING_DIR" ]; then
  mkdir "$EMBEDDING_DIR"
  echo "Embedding directory created"
else
  echo "Embedding directory already exists"
fi

if [ "$MODEL_NAME" == "mte" ]; then
  current_dir=$(pwd)
  cd MTE/
  python3 embed.py --data_path="$current_dir/$FILE_PATH"
  cd ..
fi

python3 embed.py "$FILE_PATH" "$MODEL_NAME"

exit 0
