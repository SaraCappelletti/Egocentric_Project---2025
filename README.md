# Egocentric_Project---2025

# NLQ for Egocentric Videos

A Natural Language Query (NLQ) system for egocentric videos implemented using a two-stage pipeline combining VSLNet variants with Video-LLaVA for semantic answer extraction.

## Overview

This project implements a natural language query system for egocentric videos using:

1. **Stage 1: Temporal Localization** - VSLNet (and variants) for temporal moment localization
2. **Stage 2: Semantic Answer Extraction** - Video-LLaVA for extracting semantic answers from temporally localized segments

### VSLNet Variants Implemented
- **VSLNet**: Original implementation
- **VSLBase**: Base variant of VSLNet
- **VSLNet with Non-Shared Encoders**: VSLNet variant using separate encoders

## Project Structure

```
Egocentric_Project---2025/
├── ego4d_nlq_benchmark.ipynb          # Main execution notebook
├── LICENSE
├── NLQ/
│   └── VSLNet/                        # VSLNet implementation
│       ├── main.py                    # VSLNet main script
│       ├── main_VSLBase.py           # VSLBase main script
│       ├── mainVSLNet_dual.py        # VSLNet with non-shared encoders
│       ├── model/                    # Model architectures
│       ├── utils/                    # Utility functions
│       ├── options.py                # Configuration options
│       ├── select_queries.py         # Query selection utilities
│       ├── requirements.txt          # Python dependencies
│       └── runs/                     # Training/evaluation logs
├── Video-LLaVA/                      # Video-LLaVA implementation
│   ├── videollava/                   # Core Video-LLaVA modules
│   │   ├── model/                    # Model implementations
│   │   ├── eval/                     # Evaluation scripts
│   │   ├── train/                    # Training scripts
│   │   └── serve/                    # Serving utilities
│   ├── scripts/                      # Training and evaluation scripts
│   └── assets/                       # Demo files and images
└── Project Details.docx              # Project documentation
```

## Key Components

### Main Execution File
- **`ego4d_nlq_benchmark.ipynb`**: Primary notebook containing:
  - Model initialization and hyperparameter configuration
  - VSLNet variant execution pipeline
  - Video-LLaVA integration for semantic answer extraction
  - Statistical analysis and performance metrics computation
  - Complete pipeline orchestration

### VSLNet Implementation (`NLQ/VSLNet/`)
- **`main.py`**: Standard VSLNet implementation
- **`main_VSLBase.py`**: VSLBase variant execution
- **`mainVSLNet_dual.py`**: VSLNet with non-shared encoders
- **`model/`**: Neural network architectures and components
- **`utils/`**: Data processing and utility functions
- **`options.py`**: Configuration management
- **`select_queries.py`**: Query preprocessing utilities

### Video-LLaVA Integration
- **`videollava/`**: Core Video-LLaVA implementation
- Integration for semantic answer extraction from temporal predictions

## Setup and Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab environment (original implementation)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Egocentric_Project---2025
   ```

2. **Install VSLNet dependencies**
   ```bash
   cd NLQ/VSLNet
   pip install -r requirements.txt
   ```

3. **Install Video-LLaVA dependencies**
   ```bash
   cd ../../Video-LLaVA
   pip install -e .
   ```

4. **Feature Extraction Setup**
   - The project supports Omnivore and EgoVLP features
   - Installation cells for both feature extractors are available in `ego4d_nlq_benchmark.ipynb`

## Usage

### Running the Complete Pipeline

1. **Open the main notebook**
   ```bash
   jupyter notebook ego4d_nlq_benchmark.ipynb
   ```

2. **Configure paths**
   - Update all Google Drive paths to your local/server paths
   - Modify dataset and model weight locations

3. **Execute the pipeline**
   - Run the notebook cells sequentially
   - The notebook handles both VSLNet training/inference and Video-LLaVA integration

### Running Individual Components

#### VSLNet Variants
```bash
cd NLQ/VSLNet

# Standard VSLNet
python main.py --[options]

# VSLBase variant
python main_VSLBase.py --[options]

# VSLNet with non-shared encoders
python mainVSLNet_dual.py --[options]
```

#### Configuration
- Modify `options.py` for hyperparameter tuning
- Use `select_queries.py` for query preprocessing

## Data Requirements

### Datasets
- **Ego4D Dataset**: Required for egocentric video data
- **Feature Files**: Pre-extracted Omnivore/EgoVLP features
- **Annotations**: NLQ annotations for training and evaluation

### Important Notes
- **Dataset weights omitted**: Due to large file sizes, model weights and datasets are not included
- **Feature extraction**: Omnivore and EgoVLP feature extraction code is provided in the main notebook
- **Path configuration**: All Google Drive paths must be updated for local execution

## Two-Stage Pipeline

### Stage 1: Temporal Localization (VSLNet)
- Processes natural language queries and video features
- Outputs temporal boundaries (start/end times) for relevant video segments
- Three variants available for different architectural approaches

### Stage 2: Semantic Answer Extraction (Video-LLaVA)
- Takes temporally localized video segments
- Generates semantic answers to natural language queries
- Leverages large language model capabilities for comprehensive responses

## Model Variants

### VSLNet
- Original video segment localization network
- Cross-modal attention for query-video alignment

### VSLBase
- Simplified baseline version of VSLNet
- Reduced complexity for faster training/inference

### VSLNet with Non-Shared Encoders
- Uses separate encoders for different modalities
- Enhanced representation learning capability

## Configuration and Hyperparameters

Key hyperparameters are defined in:
- `ego4d_nlq_benchmark.ipynb` (main configuration)
- `NLQ/VSLNet/options.py` (VSLNet-specific options)

Typical configurations include:
- Learning rates, batch sizes, epochs
- Model architecture parameters
- Feature dimensions and processing options

## Evaluation and Statistics

The main notebook (`ego4d_nlq_benchmark.ipynb`) includes:
- Performance metric calculations
- Statistical analysis of results
- Comparison between different VSLNet variants
- End-to-end pipeline evaluation

## License

See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Troubleshooting

### Common Issues
1. **Path errors**: Ensure all Google Drive paths are updated to local paths
2. **Missing dependencies**: Install all requirements for both VSLNet and Video-LLaVA
3. **CUDA issues**: Verify GPU compatibility and CUDA installation
4. **Memory errors**: Reduce batch sizes or use gradient checkpointing

### Support
For issues and questions, please create an issue in the repository or refer to the original VSLNet and Video-LLaVA documentation.