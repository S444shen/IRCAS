
<img width="1263" height="1048" alt="workflow" src="https://github.com/user-attachments/assets/10a0845c-f35f-4663-9bcf-54deb37a3dfb" />

Unlike previous methods that separate AS prediction and classification, IRCAS achieves higher end-to-end accuracy by addressing splice site localization errors (reducing average error from 2.1 bp to near zero) and improving classification robustness. This makes IRCAS a powerful tool for studying AS across diverse species without relying on reference genomes.

## Model Application

IRCAS can be applied to a wide range of research scenarios and species:

### Environment 
- conda install environment.yml
- download relevent modelweight from xxx
### Supported Input Data
- Raw RNA-seq transcript assemblies (FASTA format)

### Output Formats
- Accurate transcript pairs
- Splice site coordinates with high precision
- AS event Type with confidence scores

### Usage
#### For plant 
python IRCAS.py transcripts.fa plant --threads n
#### For animal 
python IRCAS.py transcripts.fa plant --threads n

## Model Training

### Training Architecture
IRCAS employs a multi-stage training approach with three specialized components:

#### Stage 1: SUPPA validation dataset preparation
- **Data**: Validated splice site sequences with flanking regions (±50 bp)
- **Architecture**: Attention-based CNN with positional encoding
- **Loss Function**: Binary cross-entropy with focal loss for imbalanced data
- **Training Strategy**: Progressive learning with curriculum scheduling

#### Stage 2: Splice Site Rectification Training
- **Data**: Validated splice site sequences with flanking regions (±50 bp)
- **Method**: Self-supervised learning on cDBG topology patterns
- **Objective**: Learn AS-indicative graph structural features
- **Validation**: Cross-validation on diverse species datasets

#### Stage 3: AS Classification Training
- **Data**: Annotated AS events across seven canonical types
- **Architecture**: Hybrid GNN combining GraphSAGE and GAT layers
- **Loss Function**: Hybrid Loss Function
- **Regularization**: Dropout, batch normalization, and early stopping

## Data Source
all data and model available in xxx
### Training Datasets Source:
- **GENCODE Human Transcriptome**: 200,000+ validated transcript pairs
- **GENCODE Mouse Transcriptome**: Annotations from 50+ vertebrate species
- **TAIR Arabidopsis**: Plant-specific alternative splicing events
- **FlyBase Drosophila**: Invertebrate AS patterns and validation

### Data Preprocessing
- **Augmentation**: Synthetic negative examples and balanced sampling
- **Split Strategy**: 70% training, 15% validation, 15% testing


