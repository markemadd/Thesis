import requests
import pandas as pd
from io import StringIO
import time

df = pd.read_csv("/Users/markemad/Documents/Spring 2026/Thesis/Documentation/normalized df-1.csv")
protein_names = df["Protein.names"].tolist()

# Preprocessing: extract primary name before first parenthesis
protein_names_clean = df["Protein.names"].str.extract(r'^([^(]+)')[0].str.strip().tolist()

def query_uniprot_single(original_name, clean_name, organism="9606"):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f'protein_name:"{clean_name}" AND organism_id:{organism}',
        "fields": "accession,gene_names,protein_name",
        "format": "tsv",
        "size": 5
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200 and len(response.text.strip().splitlines()) > 1:
            result_df = pd.read_csv(StringIO(response.text), sep="\t")
            result_df["Query_original"] = original_name  # original full name
            result_df["Query_clean"] = clean_name        # preprocessed name used for query
            return result_df
    except Exception as e:
        print(f"Error querying '{clean_name}': {e}")
    
    return pd.DataFrame()

results = []
for i, (original, clean) in enumerate(zip(protein_names, protein_names_clean)):
    result = query_uniprot_single(original, clean)
    results.append(result)
    
    if i % 100 == 0:
        print(f"Progress: {i}/{len(protein_names)}")
        time.sleep(1)

mapped = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
mapped.to_csv("mapped_proteins.csv", index=False)

# Summary
print(f"Original proteins: {len(protein_names)}")
print(f"Mapped proteins: {mapped['Query_original'].nunique()}")
print(f"Unmapped proteins: {len(protein_names) - mapped['Query_original'].nunique()}")