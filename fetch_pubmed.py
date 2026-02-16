from Bio import Entrez
import os

Entrez.email = "your_email@example.com"  # required but no API key needed

def fetch_papers(query="diabetes treatment", max_results=5):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    id_list = record["IdList"]

    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
    papers = handle.read()

    return papers
