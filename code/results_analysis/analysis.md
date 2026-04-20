# Project: Contrastive Learning Embedding Space Analysis

Do not modify code outside of the folder ContrastLEMBAS/results_analysis. For every file, provide clear and short functionality at the top describing the general workflow. Then, for each, function, provide docstrings that clearly and simply state what the function is doing, arguments, and returns. 

## embeddings.py
Write a simple function that utilizes a saved model from contrast_train.py to create embeddings in a shared embedding space given a dictionary of drug inputs and protein inputs where the names / ids match to model inputs. 

## classifier.py
Use the generated embedding space. Then, write a simple model and training loop of a simple classifier that takes in the protein and drug embedding generated from the model to predict whether the protein-drug pair is an interaction or not an interaction. 

## plot_metrics.py
Given the saved json file saved in contrast_train.py, write simple code to plot the different types of loss (both validation and training) over epochs. Make sure different values are distinguishable by color. 

## top_k_analysis.py
Use the generated embedding space. Then, write functions that perform top-k analysis per drug, finding the k nearest proteins in this shared embedding space. Additionally, write a function where when provided a set of proteins, find the rank of each protein in the ranked list of proteins to the drug. This function should be fast (even in cases with 7k drugs and 3k proteins).

## visualization.py
Use the generated embedding space. Write a function that performs t-SNE, PCA, and UMAP on protein embeddings and drug embeddings separately with coloring and labelling by group. 