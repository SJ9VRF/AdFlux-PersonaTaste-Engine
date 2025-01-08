# AdFlux PersonaTaste Engine

![Screenshot_2025-01-06_at_10 52 12_AM_1-removebg-preview](https://github.com/user-attachments/assets/c0a7bde4-9ade-4689-993e-596212a1c083)

---

## Models Overview

### Foundation Models
- **BERT**:  
  Used for classification tasks (e.g., predicting user actions).  

- **GPT**:  
  Generates next-token predictions for user behavior simulation.  

- **T5**:  
  Performs sequence-to-sequence tasks like search-to-purchase mapping.  

## Custom Models
- **Next-Click Predictor**:  
  - **Input**: Current session data.  
  - **Output**: Next action (click, view, search, or purchase).  

- **Ad CTR Predictor**:  
  - **Input**: Ad-related user interaction data.  
  - **Output**: Probability of ad click.  

- **Purchase Likelihood Model**:  
  - **Input**: Session history.  
  - **Output**: Probability of purchase.  

---

### Synthetic Data Details

The synthetic data includes:
- **User Attributes**:  
  - Age  
  - Gender  
  - Location  

- **Session Attributes**:  
  - Session duration  
  - Device type  
  - Session count  

- **Ad Attributes**:  
  - Ad category  
  - Cost-per-click (CPC)  
  - Conversion rate  

- **Event Types**:  
  - Clicks  
  - Views  
  - Searches  
  - Purchases
 
---
## Getting Started

Prerequisites
Python 3.8+
Virtual environment (recommended)
Installation
Clone the repository:
```bash
git clone https://github.com/your-repo/ads_behavior_simulator.git
cd ads_behavior_simulator
```
Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage
1. Download Foundation Models

Run the script to download BERT, GPT, and T5 models:
```bash
python src/foundation_models.py
```
2. Generate Synthetic Data

Generate user interaction data for training:
```bash
python src/generate_synthetic_data.py
```
3. Train Custom Models

Train custom models for tasks like next-click prediction:
```bash
python src/custom_model_training.py
```
4. Run Inference

Run simulations using trained models:
```bash
python src/inference.py
```
5. Search-to-Purchase Mapping

Test search-to-purchase mapping with T5:
```bash
python src/search_to_purchase_mapping.py
```
---

## More Related Models

- **[AdFlux Engine](https://github.com/SJ9VRF/AdFlux-Engine)**: AdFlux Engine is a Foundation Model based Advertisement Simulator designed to predict and simulate user behavior (clicks and views) based on historical interaction data. Leveraging cutting-edge architectures like LAVA, Decision Transformers, Gato, and MuZero, AdFlux Engine helps optimize ad placements, enhance user engagement, and maximize conversion rates by including Reinforcement Learning from Human Feedback. Built with modularity and scalability in mind, AdFlux Engine combines the latest advancements in NLP, reinforcement learning, sequence modeling and RLHF.

- **[AdFlux Agentic Engine](https://github.com/SJ9VRF/AdFlux-Agentic-Engine)**: An agentic engine designed to address the complexities of a sophisticated ad system, simulating agent-based decision-making for more accurate user behavior predictions. By integrating cutting-edge models like Gato and MuZero, the Agentic Engine helps in real-time ad optimization, enhancing campaign performance.
