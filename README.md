# Cross-Document Coreference Resolution (CDCR) System

## Problem Statement
Develop a machine learning-based system for Cross-Document Coreference Resolution (CDCR) to identify and resolve mentions of the same entity across multiple documents.

## Team Members
- **Gaurav N Kumar** - USN: 1MS22CI027
- **Syed Abdul Haadi** - USN: 1MS22CI070
- **Suramsh S Zingade** - USN: 1MS22CI068
- **Vishal R Shetty** - USN: 1MS22CI077

---

## Project Introduction
The Cross-Document Coreference Resolution (CDCR) project is focused on solving the problem of linking mentions of the same entity across different documents. Coreference resolution plays a crucial role in improving the understanding of text by identifying relationships between words or phrases that refer to the same real-world entity.

The main objective of this project is to design and implement a system that extracts potential entity mentions from documents, computes various similarity metrics (textual, contextual, and semantic), and groups related mentions into clusters representing coreferences. The system utilizes state-of-the-art machine learning techniques, such as sentence transformers and neural networks, to compute these similarities.

Our approach is based on leveraging language models (spaCy) for Named Entity Recognition (NER), a customized similarity computation algorithm, and DBSCAN clustering to group mentions. The system is flexible enough to handle different types of entities and allows for adjustable similarity weights, making it adaptable to diverse use cases. Furthermore, we provide comprehensive outputs in multiple formats, such as JSON, Excel, and text, for easy interpretation and use.

This project contributes to the field of natural language processing (NLP) by providing a robust solution for resolving cross-document coreferences, which is vital for improving tasks such as information retrieval, question answering, and document summarization.

---

## Overview
The CDCR system identifies and resolves coreferences across multiple documents. Coreference resolution refers to the process of linking mentions of the same real-world entity across different texts. This system extracts mentions from documents, computes similarities between them, clusters related mentions, and exports the results in various formats.

### System Configuration
The system is configured using the `CDCRConfig` class, which holds the necessary settings for the operation of the system:

#### `CDCRConfig` Class
- **Encoder model path**: Path to the pre-trained sentence transformer model for encoding mentions.
- **Cross encoder model path**: Path to the cross encoder model used to evaluate contextual relationships between mentions.
- **Device settings**: Configuration to specify whether to run the system on a CPU or GPU.
- **Batch size**: Defines how many documents or mentions to process at once.
- **Context window size**: Defines the number of words around a mention to consider as context.
- **Similarity weights**: Customizable weights for the different similarity components (text, context, and semantic).
- **Clustering parameters**: Parameters for the clustering algorithm, such as DBSCAN settings.
- **Supported entity types**: A list of entity types (e.g., persons, organizations) that the system will recognize and handle.

---

## Main System Components

### 1. Initialization
The `EnhancedCDCRSystem` class handles the setup and initialization of the system, ensuring all necessary models and configurations are loaded and set up.

Steps:
- Load language model: Load the spaCy model for Named Entity Recognition (NER).
- Load encoder models: Load the sentence transformer and cross encoder models.
- Set up custom entity patterns: Define any additional custom patterns for entity recognition.
- Configure logging: Set up logging for debugging and tracking system performance.
- Initialize neural network: Set up a semantic neural network for advanced similarity calculations.

### 2. Mention Extraction Process
The `extract_mentions` function is responsible for extracting potential mentions of entities within a set of documents. This process is crucial for identifying what to resolve.

Steps:
1. Process document: Use spaCy to tokenize and recognize entities in the document.
2. Extract named entities: For each recognized entity type (e.g., "Tim Cook", "Apple"), create a `MentionSpan` object.
3. Extract noun phrases: If not already identified as a named entity, check if the noun phrase matches any custom patterns.
4. Return mentions: Return a list of all extracted mentions for further processing.

### 3. Similarity Computation
The `compute_similarity_matrix` function computes the similarity between all pairs of mentions, considering text similarity, context similarity, and semantic similarity.

Steps:
1. **Text similarity**: Compute cosine similarity between the text embeddings of the mentions.
2. **Context similarity**: Compute cosine similarity between the context embeddings surrounding each mention.
3. **Semantic similarity**: Process the mentions through a neural network to calculate semantic similarity.
4. **Combine similarities**: Use weighted sums to combine the similarities into a single metric, normalizing the result to the range [0, 1].
5. **Apply entity type constraints**: Reduce similarity if the mentions belong to different entity types (e.g., person vs organization).
6. Return final similarity matrix: Return the matrix that captures all pairwise similarities between mentions.

### 4. Mention Clustering
The `cluster_mentions` function clusters mentions based on the computed similarity matrix, using a density-based clustering algorithm (DBSCAN).

Steps:
1. Convert similarity matrix to distance matrix: Since DBSCAN requires distance measures, the similarity matrix is transformed into a distance matrix.
2. Apply DBSCAN: Run the DBSCAN clustering algorithm to group mentions based on similarity.
3. Filter out noise: Identify and discard any mentions that are not clustered (noise points).
4. Return clusters: Return the list of mention clusters.

### 5. Main Pipeline
The `resolve_coreferences` function integrates all the components and resolves coreferences in the given documents.

Steps:
1. Extract mentions: Extract mentions from all input documents.
2. Compute similarity matrix: Calculate the pairwise similarity between all mentions.
3. Cluster mentions: Cluster mentions based on similarity.
4. Export results: Output the results in multiple formats: 
   - JSON: Detailed cluster information.
   - Excel: Structured mention details.
   - Text: Summary of coreference relationships.
5. Return clustered mentions: Return the final clusters representing coreferences.

#### Example Use Case
Given the following documents:
- "Apple CEO Tim Cook announced new AI features."
- "Cook emphasized privacy."
- "Microsoft revealed its new Surface Pro."

The system will:
1. Identify mentions such as "Tim Cook", "Cook", "Apple", and "Microsoft".
2. Calculate the similarities between these mentions.
3. Group "Tim Cook" and "Cook" into the same cluster, representing the same entity.
4. Output the clustered mentions in JSON, Excel, and text formats.

---

## Key Features
1. **Multi-component similarity calculation**: Computes text, context, and semantic similarities to assess coreference potential.
2. **Custom entity pattern recognition**: Allows the detection of custom entity types using predefined patterns.
3. **Flexible clustering parameters**: DBSCAN parameters and similarity weights can be adjusted for different use cases.
4. **Comprehensive output formats**: Results can be exported as JSON, Excel, or plain text for easy consumption.
5. **Debug logging and metrics tracking**: Comprehensive logging for debugging and system performance tracking.
6. **Support for various entity types**: The system can be configured to handle multiple entity types beyond just persons and organizations.
7. **Context-aware mention processing**: Mentions are processed with respect to their surrounding context, improving coreference accuracy.

---

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
