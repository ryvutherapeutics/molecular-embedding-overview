import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")


''' Mol2vec '''
def sentences2vec(sentences, mol2vec_model, unseen=None):    
    keys = set(mol2vec_model.wv.key_to_index)
    vec = []
    if unseen:
        unseen_vec = mol2vec_model.wv.get_vector(unseen)
    for sentence in sentences:
        if unseen:
            vec.append(sum([mol2vec_model.wv.get_vector(word) if word in set(sentence) & keys else unseen_vec for word in sentence]))
        else:
            vec.append(sum([mol2vec_model.wv.get_vector(word) for word in sentence if word in set(sentence) & keys]))
    return np.array(vec)

def get_mol2vec(smiles_list):
    from mol2vec.features import mol2alt_sentence, MolSentence
    from gensim.models import word2vec
    global mol2vec_model
    mol2vec_model = word2vec.Word2Vec.load("models/mol2vec_model_300dim.pkl")
    sentence_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        sentence = MolSentence(mol2alt_sentence(mol, radius=2))
        sentence_list.append(sentence)
    vec_list = sentences2vec(sentence_list, mol2vec_model)
    return vec_list


''' Graph2vec '''
def mol2graph(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num = atom.GetAtomicNum(),
            atom_symbol = atom.GetSymbol()
        )  
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type = bond.GetBondType()
        )
    return G

def get_graph2vec(smiles_list):
    global nx, Graph2vec
    import networkx as nx
    from karateclub import Graph2Vec
    graph2vec_model = Graph2Vec()
    graph_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        graph = mol2graph(mol)
        graph_list.append(graph)
    graph2vec_model.fit(graph_list)
    vec_list = graph2vec_model.get_embedding()
    return vec_list


''' ChemBERTa '''
def get_chemberta(smiles_list):
    from tqdm import tqdm
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    chemberta.eval()
    embeddings_cls = torch.zeros(len(smiles_list), 600)
    embeddings_mean = torch.zeros(len(smiles_list), 600)

    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list)):
            encoded_input = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
            model_output = chemberta(**encoded_input)
            embedding = torch.mean(model_output[0], 1)
            embeddings_mean[i] = embedding
            
    return embeddings_mean.numpy()


''' Continuous Data-Driven Descriptors '''
def prepare_json(smiles_list, batch_size=1):
    batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    json_data = {"batches": batches}
    return json_data

def get_descriptors(json_data, url="http://192.168.1.100:80/predict"):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(json_data), headers=headers)
    response_json = response.json()
    response_dict = json.loads(response_json["Prediction"])
    df = pd.DataFrame.from_dict(response_dict)
    descriptor_list = df.iloc[:, 2:].values
    return descriptor_list

def get_cddd(smiles_list, batch_size=1):
    global requests, json, pd
    import requests
    import json
    import pandas as pd
    smiles_list = list(smiles_list)
    json_data = prepare_json(smiles_list)
    descriptor_list = get_descriptors(json_data)
    return descriptor_list


''' Molecular AutoenCoding Auto-Workaround '''
def get_macaw(smiles_list, n_dimensions=20):
    from MACAW import macaw
    mcw = macaw.MACAW(n_components=n_dimensions)
    mcw.fit(smiles_list)
    embedding = mcw.transform(smiles_list)
    return embedding


''' MolFormer '''
def get_molformer(smiles_list):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

    smiles_list = [str(smiles) for smiles in smiles_list]
    inputs = tokenizer(smiles_list, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = np.array(outputs.pooler_output)
    return embedding


'''Molecula Transformer Embeddings - helper '''
def get_mte(file_path, smiles_list):
    embedding_npz = np.load(file_path)
    embedding_list = []
    for smiles in smiles_list:
        embedding = np.mean(embedding_npz[smiles], axis=0)
        embedding_list.append(embedding)
    embedding_list = np.array(embedding_list)
    return embedding_list


''' GPT2 '''
def get_gpt2(smiles_list):
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel, DataCollatorWithPadding
    tokenizer = GPT2TokenizerFast.from_pretrained("entropy/gpt2_zinc_87m", max_len=256)
    model = GPT2LMHeadModel.from_pretrained("entropy/gpt2_zinc_87m")
    collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')

    smiles_list = [str(smiles) for smiles in smiles_list]
    inputs = collator(tokenizer(smiles_list))
    outputs = model(**inputs, output_hidden_states=True)
    full_embeddings = outputs[-1][-1]
    mask = inputs['attention_mask']
    embedding = ((full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1))
    embedding = [vec.detach().numpy() for vec in embedding]
    return embedding


''' BERT for SMILES '''
def get_bert(smiles_list):
    from transformers import BertTokenizerFast, BertModel
    model = BertModel.from_pretrained("unikei/bert-base-smiles")
    tokenizer = BertTokenizerFast.from_pretrained("unikei/bert-base-smiles")
    
    smiles_list = [str(smiles) for smiles in smiles_list]
    inputs = tokenizer(smiles_list, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.detach().numpy()
    embedding = np.mean(embedding, axis=1)
    return embedding


''' Molecule Attention Transformer '''
def load_mat_model():
    from MAT.models_mat import MatModel
    from MAT.configuration_mat import MatConfig
    from MAT.featurization_mat import MatFeaturizer

    # Config
    state_dict = torch.load("models/mat_masking_20M.pt")
    missing_keys = ("generator.proj.weight", "generator.proj.bias")
    missing_sizes = ((1, 1024), (1,))
    for key, size in zip(missing_keys, missing_sizes):
        state_dict[key] = torch.Tensor(np.zeros(size))
    config = MatConfig.from_pretrained("models/mat_masking_20M.json")

    # Featurizer
    featurizer = MatFeaturizer(config)

    # Model
    model = MatModel(config=config)
    model.load_state_dict(state_dict)
    model.eval()

    return config, featurizer, model


def get_mat(smiles_list):
    config, featurizer, model = load_mat_model()
    batch = featurizer(smiles_list)
    batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
    embedded = model.src_embed(batch.node_features)
    encoding = model.encoder(embedded, batch_mask,
                             adj_matrix=batch.adjacency_matrix,
                             distance_matrix=batch.distance_matrix)
    embedding = np.mean(encoding.detach().numpy(), axis=1)
    return embedding


''' Relative Molecule Self-Attention Transformer '''
def load_rmat_model():
    from MAT.models_rmat import RMatModel
    from MAT.configuration_rmat import RMatConfig
    from MAT.featurization_rmat import RMatFeaturizer

    # Config
    state_dict = torch.load("models/rmat_4M.pt")
    missing_keys = ("generator.att_net.0.weight", "generator.att_net.2.weight", "generator.proj.weight", "generator.proj.bias")
    missing_sizes = ((128, 768), (4, 128), (1, 3072), (1,))
    for key, size in zip(missing_keys, missing_sizes):
        state_dict[key] = torch.Tensor(np.zeros(size))
    config = RMatConfig.from_pretrained("models/rmat_4M.json")

    # Featurizer
    featurizer = RMatFeaturizer(config)

    # Model
    model = RMatModel(config=config)
    model.load_state_dict(state_dict)
    model.eval()

    return config, featurizer, model


def get_rmat(smiles_list):
    config, featurizer, model = load_rmat_model()
    batch = featurizer(smiles_list)
    batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
    embedded = model.src_embed(batch.node_features)
    distances_matrix = model.dist_rbf(batch.distance_matrix)
    edges_att = torch.cat((batch.bond_features, batch.relative_matrix, distances_matrix), dim=1)
    encoding = model.encoder(embedded, batch_mask, edges_att=edges_att)
    embedding = np.mean(encoding.detach().numpy(), axis=1)
    return embedding