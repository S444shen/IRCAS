Unlike previous methods that separate AS prediction and classification, IRCAS achieves higher end-to-end accuracy by addressing splice site localization errors (reducing average error from 2.1 bp to near zero) and improving classification robustness. This makes IRCAS a powerful tool for studying AS across diverse species without relying on reference genomes.

## Model Application

IRCAS can be applied to a wide range of research scenarios and species:

### Research Applications
- **Comparative Genomics**: Study alternative splicing patterns across different species without requiring reference genomes
- **Evolutionary Biology**: Investigate the evolution of splicing mechanisms and isoform diversity
- **Non-Model Organisms**: Analyze alternative splicing in understudied species where genomic resources are limited
- **Disease Research**: Identify aberrant splicing patterns in clinical samples from any organism
- **Agricultural Genomics**: Study crop species and their wild relatives for breeding programs

### Supported Input Data
- Raw RNA-seq transcript assemblies (FASTA format)
- De novo transcriptome assemblies
- Pooled transcript sequences from multiple samples
- Single-cell RNA-seq derived transcripts

### Output Formats
- Detailed AS event annotations with confidence scores
- Splice site coordinates with sub-nucleotide precision
- Classification results for all seven canonical AS types
- Visualization-ready reports for downstream analysis

## Model Training

### Training Architecture
IRCAS employs a multi-stage training approach with three specialized components:

#### Stage 1: Splice Site Detection Training
- **Data**: Validated splice site sequences with flanking regions (±50 bp)
- **Architecture**: Attention-based CNN with positional encoding
- **Loss Function**: Binary cross-entropy with focal loss for imbalanced data
- **Training Strategy**: Progressive learning with curriculum scheduling

#### Stage 2: Graph Construction Training
- **Data**: Transcript pairs with known AS relationships
- **Method**: Self-supervised learning on cDBG topology patterns
- **Objective**: Learn AS-indicative graph structural features
- **Validation**: Cross-validation on diverse species datasets

#### Stage 3: AS Classification Training
- **Data**: Annotated AS events across seven canonical types
- **Architecture**: Hybrid GNN combining GraphSAGE and GAT layers
- **Loss Function**: Multi-class cross-entropy with class balancing
- **Regularization**: Dropout, batch normalization, and early stopping

### Training Parameters
- **Batch Size**: 64 for CNN components, 32 for GNN components
- **Learning Rate**: Adaptive scheduling starting at 1e-4
- **Epochs**: 100 with early stopping (patience=10)
- **Hardware**: GPU-accelerated training (CUDA 11.0+)
- **Training Time**: ~48 hours on NVIDIA V100

### Model Validation
- **Cross-Species Validation**: Tested on 15+ species across kingdoms
- **Benchmarking**: Compared against SUPPA2, rMATS, and other state-of-the-art tools
- **Metrics**: Precision, recall, F1-score, and splice site accuracy

## Data Source

### Training Datasets

#### Primary Training Data
- **GENCODE Human Transcriptome**: 200,000+ validated transcript pairs
- **Ensembl Multi-Species**: Annotations from 50+ vertebrate species
- **TAIR Arabidopsis**: Plant-specific alternative splicing events
- **FlyBase Drosophila**: Invertebrate AS patterns and validation

#### Validation Datasets
- **Species Coverage**: Mammals, birds, fish, plants, fungi, and invertebrates
- **Tissue Diversity**: Brain, liver, heart, leaf, root, and reproductive tissues
- **Developmental Stages**: Embryonic, juvenile, and adult samples
- **Sample Size**: 500,000+ manually curated AS events

### Benchmark Datasets
- **Simulation Data**: Artificially generated AS events with known ground truth
- **Experimental Validation**: RT-PCR validated AS events from literature
- **Public Repositories**: Data from SRA, GEO, and species-specific databases

### Data Preprocessing
- **Quality Control**: Transcript length filtering (>200 bp), redundancy removal
- **Standardization**: Sequence normalization and format conversion
- **Augmentation**: Synthetic negative examples and balanced sampling
- **Split Strategy**: 70% training, 15% validation, 15% testing

### Continuous Updates
- **Monthly Releases**: Updated models with new species data
- **Community Contributions**: User-submitted datasets for model improvement
- **Feedback Integration**: Performance metrics from real-world applications
