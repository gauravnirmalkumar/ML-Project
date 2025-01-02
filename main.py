from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import CrossEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import spacy
from dataclasses import dataclass
import logging
import wandb
import pandas as pd
import os
import json
from datetime import datetime

@dataclass
class MentionSpan:
    """Structured container for mention information"""
    text: str
    type: str
    start: int
    end: int
    doc_id: int
    context: str
    
class CDCRConfig:
    """Configuration management with adjusted parameters"""
    def __init__(self):
        self.encoder_model = 'sentence-transformers/all-mpnet-base-v2'
        self.cross_encoder = 'cross-encoder/stsb-roberta-large'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.context_window = 50
        self.similarity_weights = {
            'text': 0.7,      # Increased text similarity weight
            'context': 0.2,
            'semantic': 0.1
        }
        self.clustering = {
            'eps': 0.5,       # More lenient clustering threshold
            'min_samples': 1   # Allow smaller clusters
        }
        
        # Entity types remain the same...
        self.entity_types = {
            'PERSON', 'ORG', 'GEOGRAPHICAL', 'LOC', 'FACILITY', 'PRODUCT',
            'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
            'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'NORP',
            'FAC', 'VEHICLE', 'DISEASE', 'CHEMICAL', 'SCIENTIFIC', 'TECHNOLOGY'
        }

class EnhancedCDCRSystem:
    def __init__(self, config: CDCRConfig = None):
        self.config = config if config is not None else CDCRConfig()
        self.output_dir = "cdcr_output"
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_logging()
        self._initialize_models()
        self._initialize_entity_patterns()
        # Add output directory creation

    def _initialize_entity_patterns(self):
        """Initialize custom entity patterns for spaCy"""
        try:
            # Add custom patterns to spaCy's entity ruler
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                # Technology patterns
                {"label": "TECHNOLOGY", "pattern": [{"LOWER": {"IN": ["ai", "artificial intelligence", "machine learning", "blockchain"]}}]},
                # Vehicle patterns
                {"label": "VEHICLE", "pattern": [{"LOWER": {"IN": ["car", "truck", "vehicle", "suv", "van"]}}]},
                # Disease patterns
                {"label": "DISEASE", "pattern": [{"LOWER": {"IN": ["cancer", "diabetes", "covid", "covid-19", "coronavirus"]}}]},
                # Add more patterns as needed
            ]
            ruler.add_patterns(patterns)
            
        except Exception as e:
            self.logger.error(f"Error initializing entity patterns: {str(e)}")
    
    def _setup_logging(self):
        """Setup logging configuration with file output"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'cdcr_output/processing_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        
    def _initialize_models(self):
        """Initialize models with proper error handling"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_model)
            model_config = AutoConfig.from_pretrained(self.config.encoder_model)
            self.encoder = AutoModel.from_pretrained(
                self.config.encoder_model,
                config=model_config
            ).to(self.config.device)
            
            self.cross_encoder = CrossEncoder(self.config.cross_encoder)
            self.nlp = spacy.load('en_core_web_trf')
            
            # Initialize semantic similarity model
            self.semantic_encoder = nn.Sequential(
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(384, 192)
            ).to(self.config.device)
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for input texts"""
        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.encoder(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
                
        return np.array(embeddings)
            
    def extract_mentions(self, documents: List[str]) -> List[MentionSpan]:
        """Extract mentions with debug output"""
        mentions = []
        print("\nDebug: Starting mention extraction...")
        
        for doc_id, text in enumerate(documents):
            print(f"\nProcessing document {doc_id}:")
            print(f"Text: {text}")
            
            doc = self.nlp(text)
            
            # Extract named entities
            print("\nFound entities:")
            for ent in doc.ents:
                print(f"- {ent.text} ({ent.label_})")
                
                entity_type = self._normalize_entity_type(ent.label_)
                if entity_type in self.config.entity_types:
                    context_start = max(0, ent.start_char - self.config.context_window)
                    context_end = min(len(text), ent.end_char + self.config.context_window)
                    
                    mention = MentionSpan(
                        text=ent.text,
                        type=entity_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        doc_id=doc_id,
                        context=text[context_start:context_end]
                    )
                    mentions.append(mention)
                    print(f"  Added mention: {mention.text} ({mention.type})")
            
            # Extract noun phrases that might be missed by NER
            print("\nAnalyzing noun phrases:")
            for np in doc.noun_chunks:
                if not any(ent.start <= np.start and ent.end >= np.end for ent in doc.ents):
                    print(f"- Found noun phrase: {np.text}")
                    # Check if noun phrase matches custom patterns
                    entity_type = self._detect_entity_type(np.text)
                    if entity_type:
                        mention = MentionSpan(
                            text=np.text,
                            type=entity_type,
                            start=np.start_char,
                            end=np.end_char,
                            doc_id=doc_id,
                            context=text[max(0, np.start_char - self.config.context_window):
                                      min(len(text), np.end_char + self.config.context_window)]
                        )
                        mentions.append(mention)
                        print(f"  Added custom mention: {mention.text} ({mention.type})")
        
        print(f"\nTotal mentions found: {len(mentions)}")
        return mentions

    def compute_similarity_matrix(self, mentions: List[MentionSpan]) -> np.ndarray:
        """Compute similarity with multiple components and proper normalization"""
        if not mentions:
            return np.array([])

        # Text similarity
        text_embeds = self._get_embeddings([m.text for m in mentions])
        text_sim = cosine_similarity(text_embeds)
        
        # Context similarity
        context_embeds = self._get_embeddings([m.context for m in mentions])
        context_sim = cosine_similarity(context_embeds)
        
        # Semantic similarity
        semantic_embeds = self._get_semantic_embeddings(text_embeds)
        semantic_sim = cosine_similarity(semantic_embeds)
        
        # Ensure all similarity matrices are in range [0, 1]
        text_sim = np.clip(text_sim, 0, 1)
        context_sim = np.clip(context_sim, 0, 1)
        semantic_sim = np.clip(semantic_sim, 0, 1)
        
        # Weighted combination
        similarity_matrix = (
            self.config.similarity_weights['text'] * text_sim +
            self.config.similarity_weights['context'] * context_sim +
            self.config.similarity_weights['semantic'] * semantic_sim
        )
        
        # Normalize combined similarity to [0, 1] range
        similarity_matrix = similarity_matrix / sum(self.config.similarity_weights.values())
        
        # Apply mention type constraints
        similarity_matrix = self._apply_type_constraints(similarity_matrix, mentions)
        
        # Final normalization
        similarity_matrix = np.clip(similarity_matrix, 0, 1)
        
        return similarity_matrix
    
    def _get_semantic_embeddings(self, text_embeds: np.ndarray) -> np.ndarray:
        """Get semantic embeddings using neural network"""
        with torch.no_grad():
            embeddings = torch.tensor(text_embeds).float().to(self.config.device)
            semantic_embeds = self.semantic_encoder(embeddings)
        return semantic_embeds.cpu().numpy()
    
    def _apply_type_constraints(self, sim_matrix: np.ndarray, 
                              mentions: List[MentionSpan]) -> np.ndarray:
        """Apply entity type constraints to similarity matrix"""
        for i, m1 in enumerate(mentions):
            for j, m2 in enumerate(mentions):
                if m1.type != m2.type:
                    sim_matrix[i, j] *= 0.5
        return sim_matrix
    def _normalize_entity_type(self, label: str) -> str:
        """Normalize entity types to standard categories"""
        # Mapping of similar entity types
        type_mapping = {
            'FAC': 'FACILITY',
            'PRODUCT': 'PRODUCT',
            'ORG': 'ORG',
            'GPE': 'GPE',
            'PERSON': 'PERSON',
            'NORP': 'NORP',
            'LOC': 'LOC',
            'EVENT': 'EVENT',
            'WORK_OF_ART': 'WORK_OF_ART',
            'LAW': 'LAW',
            'LANGUAGE': 'LANGUAGE',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'PERCENT': 'PERCENT',
            'MONEY': 'MONEY',
            'QUANTITY': 'QUANTITY',
            'ORDINAL': 'ORDINAL',
            'CARDINAL': 'CARDINAL'
        }
        return type_mapping.get(label, label)

    def _matches_custom_patterns(self, text: str) -> bool:
        """Check if text matches any custom entity patterns"""
        # Add custom pattern matching logic here
        text_lower = text.lower()
        custom_patterns = {
            'TECHNOLOGY': ['ai', 'artificial intelligence', 'machine learning', 'blockchain'],
            'VEHICLE': ['car', 'truck', 'vehicle', 'suv', 'van'],
            'DISEASE': ['cancer', 'diabetes', 'covid', 'covid-19', 'coronavirus']
        }
        
        return any(text_lower in patterns for patterns in custom_patterns.values())
    
    def _detect_entity_type(self, text: str) -> str:
        """Detect entity type based on custom patterns"""
        text_lower = text.lower()
        custom_patterns = {
            'TECHNOLOGY': ['ai', 'artificial intelligence', 'machine learning', 'blockchain'],
            'VEHICLE': ['car', 'truck', 'vehicle', 'suv', 'van'],
            'DISEASE': ['cancer', 'diabetes', 'covid', 'covid-19', 'coronavirus']
        }
        
        for entity_type, patterns in custom_patterns.items():
            if text_lower in patterns:
                return entity_type
        return None
    
    def cluster_mentions(self, mentions: List[MentionSpan], 
                        sim_matrix: np.ndarray) -> List[List[MentionSpan]]:
        """Improved clustering with debug output"""
        if len(mentions) == 0:
            print("No mentions to cluster")
            return []
            
        print("\nDebug: Starting clustering...")
        print(f"Number of mentions to cluster: {len(mentions)}")
        
        # Ensure similarity matrix is in range [0, 1]
        sim_matrix = np.clip(sim_matrix, 0, 1)
        
        # Print similarity matrix
        print("\nSimilarity matrix:")
        for i in range(len(mentions)):
            for j in range(len(mentions)):
                if sim_matrix[i][j] > 0.3:  # Only print significant similarities
                    print(f"{mentions[i].text} - {mentions[j].text}: {sim_matrix[i][j]:.3f}")
        
        # Convert to distance matrix
        distance_matrix = 1 - sim_matrix
        
        # Clustering
        clustering = DBSCAN(
            metric='precomputed',
            eps=self.config.clustering['eps'],
            min_samples=self.config.clustering['min_samples']
        )
        
        try:
            labels = clustering.fit_predict(distance_matrix)
            print(f"\nClustering labels: {labels}")
            
            # Group mentions by cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(labels):
                if label != -1:  # Skip noise points
                    clusters[label].append(mentions[idx])
            
            result_clusters = list(clusters.values())
            print(f"\nNumber of clusters found: {len(result_clusters)}")
            return result_clusters
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}")
            return [[mention] for mention in mentions]

        
    def resolve_coreferences(self, documents: List[str]) -> List[List[MentionSpan]]:
        """Main coreference resolution pipeline"""
        try:
            mentions = self.extract_mentions(documents)
            if not mentions:
                return []
                
            sim_matrix = self.compute_similarity_matrix(mentions)
            clusters = self.cluster_mentions(mentions, sim_matrix)
            
            # Log metrics if using wandb
            if wandb.run:
                self._log_metrics(clusters, sim_matrix)
                
            return clusters
            
        except Exception as e:
            self.logger.error(f"Coreference resolution failed: {str(e)}")

def demo():
    """Demo function with comprehensive output"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("\nInitializing CDCR system...")
        config = CDCRConfig()
        system = EnhancedCDCRSystem(config)
        
        
        documents = [
            "Apple CEO Tim Cook announced new AI features at their Cupertino headquarters. Cook emphasized privacy.",
            "Microsoft revealed its new Surface Pro laptop at an event in Redmond, Washington. The device integrates AI features similar to those announced by Apple.",
            "Tesla's Cybertruck prototype was showcased at an event in Austin, Texas, alongside comparisons to Ford's electric F-150 Lightning.",
            "Ford unveiled its electric F-150 Lightning earlier this year, with critics drawing comparisons to Tesla's Cybertruck.",
            "A Picasso painting was auctioned for $200 million at Sotheby's in New York, setting a new record.",
            "The Louvre Museum in Paris plans to host an exhibition on Picasso's lesser-known works, complementing his recent auction record.",
            "Elon Musk, CEO of Tesla and SpaceX, shared plans for Mars colonization during a conference in Los Angeles.",
            "NASA's latest mission to Mars is designed to gather data that could support colonization efforts like those proposed by Elon Musk.",
            "Netflix released a new documentary series exploring ancient civilizations, featuring interviews with archaeologists from Harvard University.",
            "Researchers at Harvard University recently published findings on a 5,000-year-old city discovered in Mesopotamia, referenced in Netflix's documentary.",
            "The Tesla Model S electric vehicle was spotted near the Golden Gate Bridge in San Francisco, catching the eye of a group of eco-enthusiasts.",
            "A new group of eco-enthusiasts in Seattle has started promoting electric vehicles like Tesla and Ford’s F-150 Lightning.",
            "Amazon's drone delivery system is now being tested in Seattle. The drones are equipped with advanced GPS systems to compete with Google's delivery drones.",
            "Google's delivery drones, part of Project Wing, were tested in rural Australia, but experts are comparing their performance to Amazon’s systems in Seattle.",
            "The Mona Lisa, painted by Leonardo da Vinci, is displayed at the Louvre Museum in Paris.",
            "A newly discovered sketch, thought to be a preliminary version of the Mona Lisa, was recently authenticated by scientists at Oxford University.",
            "BMW unveiled its newest electric vehicle, the iX, during an auto show in Munich, Germany, emphasizing sustainability features.",
            "The European Space Agency launched a new satellite to monitor climate change, highlighting the importance of sustainability in global industries, like BMW’s electric vehicle initiative.",
            "A team of archaeologists discovered a 5,000-year-old city in Mesopotamia, shedding light on early human civilization.",
            "The James Webb Space Telescope captured high-resolution images of the Mesopotamian region, aiding archaeologists in studying ancient ruins.",
            "A single mother in Mumbai shared her struggles balancing two jobs and raising three children in a bustling city.",
            "A new social program in Mumbai, inspired by the stories of single mothers, aims to provide financial and emotional support to struggling families.",
            "An orphan in a dystopian sci-fi novel embarks on a journey to discover the truth about their parents' mysterious disappearance.",
            "The orphan’s story is mirrored in a new Netflix series about a child navigating a post-apocalyptic world while searching for family.",
            "The Eiffel Tower in Paris is one of the most visited landmarks in the world, attracting millions of tourists annually.",
            "The Louvre Museum, not far from the Eiffel Tower, also draws millions of visitors each year, often including tourists interested in French art and history.",
            "SpaceX launched its Falcon 9 rocket from Cape Canaveral carrying $50 million worth of satellites.",
            "The International Space Station (ISS) recently received supplies delivered by SpaceX's Falcon 9, continuing their collaboration with NASA."
        ]

        
        print("\nProcessing documents...")
        clusters = system.resolve_coreferences(documents)
        
        # Prepare data structures for output
        output_data = {
            'timestamp': timestamp,
            'clusters': [],
            'document_analysis': [],
            'similarity_scores': []
        }
        
        # Process clusters and create structured output
        if clusters:
            for cluster_id, cluster in enumerate(clusters, 1):
                cluster_data = {
                    'cluster_id': cluster_id,
                    'entity_type': cluster[0].type if cluster else "Unlabeled",
                    'mentions': []
                }
                
                for mention in cluster:
                    mention_data = {
                        'text': mention.text,
                        'doc_id': mention.doc_id,
                        'start': mention.start,
                        'end': mention.end,
                        'context': mention.context.strip()
                    }
                    cluster_data['mentions'].append(mention_data)
                    
                output_data['clusters'].append(cluster_data)
            
            # Save to multiple output formats
            base_path = os.path.join(system.output_dir, f'cdcr_results_{timestamp}')
            
            # Save JSON output
            with open(f'{base_path}.json', 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Create Excel output
            df = pd.DataFrame([
                {
                    'Cluster ID': cluster_data['cluster_id'],  # This was correct
                    'Entity Type': cluster_data['entity_type'],
                    'Document ID': mention['doc_id'],
                    'Mention Text': mention['text'],
                    'Start Position': mention['start'],
                    'End Position': mention['end'],
                    'Context': mention['context']
                }
                for cluster_data in output_data['clusters']
                for mention in cluster_data['mentions']
            ])
            
            # Add sorting to make clusters more readable
            df = df.sort_values(['Cluster ID', 'Document ID'])
            
            df.to_excel(f'{base_path}.xlsx', index=False)
            
            # Create a summary file
            with open(f'{base_path}_summary.txt', 'w') as f:
                f.write(f"CDCR Analysis Summary\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Total Documents: {len(documents)}\n")
                f.write(f"Total Clusters: {len(clusters)}\n")
                f.write(f"Total Mentions: {sum(len(c) for c in clusters)}\n")
                f.write("\nCluster Summary:\n")
                for cluster_data in output_data['clusters']:
                    f.write(f"\nCluster {cluster_data['cluster_id']} "
                           f"({cluster_data['entity_type']}):\n")
                    for mention in cluster_data['mentions']:
                        f.write(f"  - {mention['text']} (Doc {mention['doc_id']})\n")
            
            print(f"\nResults exported to {system.output_dir}/")
            print(f"Total Clusters: {len(clusters)}")
            print(f"Total Mentions: {sum(len(c) for c in clusters)}")
            
        else:
            print("\nNo coreference clusters found.")
                
    except Exception as e:
        print(f"\nError in demo: {str(e)}")
        raise

if __name__ == "__main__":
    demo()