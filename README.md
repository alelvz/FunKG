# FunKG: A knowledge graph framework for metagenomic annotation and protein function prediction

This project explores a knowledge graph-based framework for protein function prediction in metagenomic data. It integrates taxonomic, genomic, and functional annotations into a structured graph and evaluates multiple embedding models (TransE, TransF, TransH, TransD, TransR) for predicting Gene Ontology molecular functions.

## Repository Structure

- `scripts/` – Python scripts for data preprocessing, model training, prediction, and embedding visualization.
- `data/` – Input data files, including contig-protein-taxonomy mappings and GO triples.
- `results/` – Output metrics (`.json`) from trained models for evaluation.
- `AI_Report_AlejandraL.pdf` – Final project report with methodology, results, and future directions.

## Highlights
- Use of PyKEEN for knowledge graph embedding and link prediction.
- Evaluation of models using Hits@10, Median Rank, Harmonic Mean Rank, and MRR.
- Proposal of improvements including InterProScan-based annotation, GO-aware evaluation, and taxon constraint filtering.

