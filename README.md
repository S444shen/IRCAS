
<img width="1263" height="1048" alt="workflow" src="https://github.com/user-attachments/assets/10a0845c-f35f-4663-9bcf-54deb37a3dfb" />

Unlike previous methods that separate AS prediction and classification, IRCAS achieves higher end-to-end accuracy by addressing splice site localization errors (reducing average error from 2.1 bp to near zero) and improving classification robustness. This makes IRCAS a powerful tool for studying AS across diverse species without relying on reference genomes.

## Model Application

IRCAS can be applied to a wide range of research scenarios and species:

### Environment 
- conda install environment.yml
- download relevent model weight from [website](http://zhangqblab.cn)
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
python IRCAS.py transcripts.fa animal --threads n

## Model Training

### Training Architecture
IRCAS employs a multi-stage training approach with three specialized components:

#### Stage 1: SUPPA validation dataset preparation
- **Data**: annotation file \
(suppa.py generateEvents \
    -i Araport11_GFF3_genes_transposons.201606.gtf \
    -o araport11_events \
    -f ioi)
- **Positiontranfer**: Transfer position from genome to transcript(suppapositiontransfer.py)
- **Seqeuncetranfer**: Transfer sequence to acceptable format(suppaseqeuncetransfer.py)

#### Stage 2: Splice Site Rectification Training(trainRectifacation.py)
- **Data**: Validated splice site sequences with flanking regions (±50 bp)
- **Architecture**: Attention-based CNN model 
- **Loss Function**: Huber Loss Function
- **Regularization**: Dropout, batch normalization, and early stopping

#### Stage 3: AS Classification Training (trainClassification.py)
- **Data**: Validated splice site sequences with flanking regions (±50 bp)
- **Feature and pygdata**: from sequence to cDBG(node,edge,global features) (dataprecessing_lite.py,feature.py)
- **Architecture**: Hybrid GNN combining transformer and GAT layers 
- **Loss Function**: Hybrid Loss Function
- **Regularization**: Dropout, batch normalization, and early stopping

## Data Source
All data and model available in [website](http://zhangqblab.cn)
### Training Datasets Source:
- **GENCODE Human Transcriptome**: 200,000+ validated transcript pairs
- **GENCODE Mouse Transcriptome**: 100,000+ validated transcript pairs
- **Arabidopsis Transcriptome**: Plant-specific alternative splicing events
- **Rice Transcriptome**: Invertebrate AS patterns and validation

### Data Preprocessing
- **Augmentation**: Synthetic negative examples and balanced sampling
- **Split Strategy**: 5-fold cross-validation with gene-disjoint grouping. The dataset is partitioned into 5 folds using sklearn.model_selection.GroupKFold keyed on gene IDs parsed from the GTF annotation, ensuring that no gene appears in both training and evaluation folds within any single fold split. Within each fold, training is repeated with three distinct random seeds (15 independent runs in total per species). For each fold, the training portion is further split internally into a training subset and a validation subset for early-stopping monitoring; the held-out fold serves as the test set. Performance metrics are reported as mean ± standard deviation across the 15 runs.


