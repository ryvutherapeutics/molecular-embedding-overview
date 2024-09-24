#!/bin/bash

install_requirements() {
    if [ -f "requirements.txt" ]; then
        echo "Installing requirements..."
        pip install --upgrade pip
        pip install -r requirements.txt

        if [ $? -eq 0 ]; then
            echo "Requirements installed successfully."
        else
            echo "Failed to install requirements."
            exit 1
        fi
    else
        echo "File requirements.txt not found!"
        exit 1
    fi
}

download_mol2vec_model() {
    MODEL_URL="https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl"

    echo "Downloading the model from $MODEL_URL..."
    mkdir models
    wget -O "models/mol2vec_model_300dim.pkl" "$MODEL_URL"

    if [ $? -eq 0 ]; then
        echo "Model downloaded successfully."
    else
        echo "Failed to download the model."
        exit 1
    fi
}

setup_cddd() {
    echo "Preparing the cddd_rest repository..."
    git clone https://github.com/vaxherra/cddd_rest.git

    cd cddd_rest || { echo "Failed to clone cddd_rest"; exit 1; }

    # docker build -t cddd .
    # docker run --rm -p 80:80 cddd &
    cd .. || { echo "Failed to create a CDDD container"; exit 1; }
    mkdir CDDD
    mv cddd_rest/* CDDD
    rm -rf cddd_rest
}

setup_molecular_transformer() {
    echo "Preparing MolecularTransformerEmbeddings repository..."
    git clone https://github.com/mpcrlab/MolecularTransformerEmbeddings.git
    cd MolecularTransformerEmbeddings || { echo "Failed to clone MolecularTransformerEmbeddings"; exit 1; }

    chmod +x download.sh
    ./download.sh
    mkdir MTE
    mv MolecularTransformerEmbeddings/* MTE
    rm -rf MolecularTransformerEmbeddings
    cd .. || { echo "Failed to setup MolecularTransformerEmbeddings"; exit 1; }
}

setup_macaw() {
    echo "Preparing MACAW repository..."
    git clone https://github.com/LBLQMM/MACAW.git
    cd MACAW || { echo "Failed to clone MACAW"; exit 1; }
}

install_requirements
download_mol2vec_model
setup_cddd
setup_molecular_transformer
setup_macaw

echo "Setup completed successfully!"
