# RankSense: Intelligent Search Solutions

![image](https://github.com/user-attachments/assets/00f01c2f-9b72-432f-95b5-38f2c34c43e8)


RankSense is a cutting-edge search platform designed to deliver intelligent and precise results by utilizing modern AI technologies. It integrates advanced search methods to provide semantic, multimodal, and reinforcement learning-based search capabilities.

## Features

### 1. Semantic Text Search
- Utilizes **Sentence Transformers** to perform highly accurate text matching.
- Supports multilingual search with models like `all-MiniLM-L6-v2` and `distiluse-base-multilingual-cased-v2`.

### 2. Multimodal Search
- Combines **text embeddings** and **image embeddings** for comprehensive search results.
- Enables users to query using text or images.

### 3. Reinforcement Learning-Based Search
- Implements **PPO (Proximal Policy Optimization)** to enhance search relevance through iterative feedback.
- Learns user preferences to improve result quality over time.

## Installation

### Prerequisites
- Python 3.8+
- Required libraries: Install them using the following command:
  ```bash
  pip install -r requirements.txt
  ```
### Clone the Repository
```bash
git clone https://github.com/your-repo/ranksense.git
cd ranksense
```
### Install Additional Dependencies
```bash
pip install streamlit sentence-transformers stable-baselines3 plotly
```

## Usage
### Start the Application
Run the following command to launch the app:

```bash
streamlit run main.py
```
### Explore Features
Navigate through the sidebar options:

- Introduction: Learn about RankSense.
- Text Search: Perform semantic text searches.
- Multimodal Search: Query using text or images.
- RL Search: Experience reinforcement learning-based search.
- Indexer: Manage and add documents or images to the corpus.
## How It Works
1. Semantic Text Search:

- Text embeddings are computed using Sentence Transformers.
- Queries are ranked using cosine similarity.
2. Multimodal Search:

- Combines text embeddings and image embeddings for search.
- Supports both text and image queries.
3. Reinforcement Learning Search:

- Trains an agent with PPO to rank documents effectively based on user feedback.
- Utilizes a SearchEnv environment for iterative learning.
4. Corpus Management:

- Upload and process text files.
- Generate and store embeddings for efficient search.
## Project Structure
```plaintext
.
├── main.py                  # Main application script
├── embed_img.py             # Image embedding utilities
├── searchEnv.py             # Reinforcement learning environment
├── text_corpus.json         # Example text corpus
├── db/                      # Directory for uploaded files
├── assets/                  # Directory for app assets
└── requirements.txt         # Required Python packages
```
## Technologies Used
**Streamlit**: Interactive UI for web-based search demonstrations.  
**Sentence Transformers**: For semantic embeddings.  
**PPO**: For reinforcement learning.  
**Plotly**: For data visualization.  

## Authors
This project was developed by:  
Mohannad TAZI - [Conatct](https://www.linkedin.com/in/mohannad-tazi/)  
Khawla MOUSTAFI - [Contact](https://www.linkedin.com/in/khawla-moustafi/)  
