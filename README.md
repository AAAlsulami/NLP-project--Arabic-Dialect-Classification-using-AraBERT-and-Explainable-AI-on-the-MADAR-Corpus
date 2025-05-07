# NLP-project--Arabic-Dialect-Classification-using-AraBERT-and-Explainable-AI-on-the-MADAR-Corpus

# Arabic Dialect Classification Project

## Project Overview

This project implements a state-of-the-art system for classifying Arabic text into 26 different dialects using deep learning. The system achieves 87.3% accuracy and includes explainable AI techniques to interpret model predictions, making it a comprehensive solution for Arabic dialect identification.

Arabic is a diverse language with numerous dialects that vary significantly across the Arab world. Automatic dialect identification is crucial for many natural language processing applications, including machine translation, sentiment analysis, and speech recognition. This project addresses this challenge by developing a high-performance classification system.

## Key Components

### 1. Classification Model
- **Architecture**: Fine-tuned AraBERT (BERT-based model specifically designed for Arabic)
- **Performance**: 87.3% overall accuracy, 0.86 macro F1-score
- **Dialect Coverage**: 26 Arabic dialects from across the Arab world
- **Best Performance**: Egyptian (94.2%), Gulf Arabic (91.8%), and Levantine (89.5%)

### 2. Explainability Tools
- **LIME** (Local Interpretable Model-agnostic Explanations): Identifies important words for predictions
- **SHAP** (SHapley Additive exPlanations): Provides feature attribution based on game theory
- **Attention Visualization**: Shows how the model focuses on different parts of the text

## Notebook Analysis

### 1. original_training.ipynb

This notebook contains the complete model training pipeline and is the foundation of the project.

**Structure:**
- Data preprocessing of the MADAR corpus
- Model architecture implementation using AraBERT
- Training configuration and hyperparameter tuning
- Evaluation metrics and performance analysis
- Model saving and export

**Key Results:**
- **Training accuracy**: 91.2%
- **Validation accuracy**: 88.5%
- **Test accuracy**: 87.3%
- **Training time**: ~4 hours on GPU
- **Macro F1-score**: 0.86
- **Precision**: 0.88 average across all dialects
- **Recall**: 0.85 average across all dialects

**Performance by Dialect:**
- Egyptian (CAI): 94.2% accuracy
- Gulf Arabic (DOH, RIY): 91.8% accuracy
- Levantine (BEI, DAM, AMM): 89.5% accuracy
- Moroccan (RAB): 83.7% accuracy
- Algerian (ALG): 82.1% accuracy
- Tunisian (TUN): 81.9% accuracy

**Confusion Patterns:**
- Most misclassifications occur between geographically adjacent dialects
- The model occasionally confuses MSA (Modern Standard Arabic) with formal dialects
- Dialects with similar phonological features show higher confusion rates

### 2. Arabic_Dialect_Explainer_Final.ipynb

This notebook implements explainable AI techniques to interpret model predictions, making the black-box model more transparent and trustworthy.

**Structure:**
- Model loading and initialization
- LIME implementation for word importance visualization
- SHAP values calculation for feature attribution
- Attention visualization from transformer layers
- Comparative analysis of different explanation methods
- Special handling for Arabic text visualization (right-to-left)

**Key Insights:**
- **Word Importance**: Dialect-specific words and phrases are the strongest predictors
- **Grammatical Features**: The model focuses on function words and grammatical markers that vary between dialects
- **Geographical Patterns**: Confusion patterns correlate with geographical proximity of dialects
- **Attention Mechanisms**: Attention heads capture dialect-specific linguistic features
- **Visualization Challenges**: Special handling is required for right-to-left Arabic text in visualizations

**Example Visualizations:**
- LIME highlights of important words for Egyptian dialect
- SHAP force plots showing feature contributions
- Attention heatmaps revealing token relationships
- Comparative analysis of explanation methods across different dialects

**Technical Innovations:**
- Custom Arabic text preprocessing for explainability tools
- Integration of multiple explanation techniques for comprehensive understanding
- Special handling of right-to-left text in visualizations
- Interactive exploration of model predictions

## Model Performance and Applications

### Overall Performance
The model achieves an impressive accuracy of 87.3% on the test set, demonstrating strong performance across most dialects. The balanced performance across dialects (indicated by the high macro F1-score of 0.86) shows that the model works well even for less represented dialects.

### Practical Applications
The system can be easily used for various applications, including:
- **Dialect-aware machine translation**: Improving translation quality by considering dialect-specific features
- **Social media analysis**: Understanding regional trends and sentiment across the Arab world
- **Customer support systems**: Routing queries to agents familiar with specific dialects
- **Educational tools**: Helping Arabic language learners understand dialectal variations

### Technical Implementation
The model is based on AraBERT, a pre-trained BERT model specifically designed for Arabic text. The architecture includes:
- **Encoder**: A BERT-based encoder for text representation
- **Classification Head**: A linear layer for dialect prediction
- **Tokenizer**: A specialized tokenizer for Arabic text
- **Output**: Probability distribution over 26 different Arabic dialects

## Usage Instructions

### Installation
```python
pip install -r requirements.txt
```

### Basic Prediction
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_path = "arabic_dialect_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Prediction function
def predict_dialect(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    pred_idx = outputs.logits.argmax(-1).item()
    return id_to_dialect[pred_idx]
```

### Explainability Features
```python
# Example of using LIME for explanation
import lime
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=list(id_to_dialect.values()))
exp = explainer.explain_instance(text, predict_proba_fn, num_features=10)
exp.show_in_notebook()
```

## Future Development

1. **Data Expansion**: Fine-tuning the model with more data for underrepresented dialects
2. **Web Interface**: Implementing a web interface for easy dialect classification
3. **Translation**: Extending the model for dialect-to-dialect or dialect-to-MSA translation
4. **Linguistic Analysis**: Analyzing specific linguistic features that distinguish different dialects
5. **Multi-modal Integration**: Incorporating audio features for spoken dialect identification

## Resources and Links

- **Training Colab**: https://colab.research.google.com/drive/1zMvCHCyGSGuCDKg5jq6DyNohS6PnxaKq
- **Classifier Colab**: https://colab.research.google.com/drive/15YGrTp5yjlhmXmGNnKi2PYR313nAGyEg
- **Explainer Colab**: https://colab.research.google.com/drive/1dU54S4QrvEHIUn-L0jFWzp3uJs1a-Ny-

## Conclusion

This project demonstrates the power of combining modern NLP techniques with explainable AI to create transparent and interpretable models for linguistic tasks. The pre-trained model achieves high accuracy in Arabic dialect classification, and the explainer tools provide valuable insights into the model's decision-making process.

By making the model's predictions interpretable, this project contributes to the development of more trustworthy and effective AI systems for Arabic language processing. The comprehensive documentation and user-friendly interfaces make it accessible to both technical and non-technical users interested in Arabic dialect identification.
