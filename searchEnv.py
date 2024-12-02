import gym
import numpy as np
from stable_baselines3 import PPO
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

class SearchEnv(gym.Env):
    def __init__(self, text_documents, query=None, image_query=None):
        super(SearchEnv, self).__init__()

        # Initialize the transformer model for text embeddings
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

        # Store the text documents and image index
        self.text_documents = text_documents
        self.query = query
        
        self.state = None
        self.done = False

        # Define the action space (ranking actions, considering both text and image)
        self.action_space = gym.spaces.Discrete(len(text_documents))  # Rank all text and image docs

        # Define the observation space (query and features of all documents/images)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(text_documents), 2), dtype=np.float32
        )

    def reset(self):
        self.done = False
        # Reset state to initial state (query and embeddings)
        self.state = self.generate_state()
        return self.state

    def generate_state(self):
        # Generate query embedding
        query_embedding = self.model.encode([self.query])[0] if self.query else np.zeros(self.model.get_sentence_embedding_dimension())

        # Generate embeddings for each document in the corpus
        text_doc_embeddings = np.array([self.model.encode([doc['content']])[0] for doc in self.text_documents])

        # Calculate cosine similarities between the query and each document
        text_similarities = cosine_similarity([query_embedding], text_doc_embeddings)[0] if query_embedding is not None else np.zeros(len(self.text_documents))

        # Create combined features: each document will have 2 features: similarity and some constant (or additional info)
        # For now, let's set the second feature to a constant (e.g., 1.0) for simplicity. This can be replaced with other features.
        second_feature = np.ones(len(self.text_documents))  # Example: Constant value (you can replace it with other features)

        # Stack the two features into a combined feature matrix (shape: n_docs, 2)
        combined_features = np.vstack((text_similarities, second_feature)).T  # Shape: (n_docs, 2)

        return combined_features

    def step(self, action):
        if self.done:
            return self.state, 0, self.done, {}

        selected_document = self.text_documents[action]

        # Simulate user feedback (thumbs up / thumbs down)
        reward = self.get_feedback(selected_document)

        # Update done state if certain conditions are met (e.g., user interaction completed)
        self.done = True

        return self.state, reward, self.done, {}

    def get_feedback(self, selected_document):
        # Display the selected document and ask for user feedback
        st.write(f"Selected Document: {selected_document['content']}")
        feedback = st.text_input("Was this document helpful? (yes/no): ").strip().lower()

        # Convert the user's response into a numerical feedback: thumbs up (1) for 'yes', thumbs down (-1) for 'no'
        if feedback == 'yes':
            return 1  # Thumbs up
        elif feedback == 'no':
            return -1  # Thumbs down
        else:
            st.write("Invalid input. Please answer with 'yes' or 'no'.")
            return self.get_feedback(selected_document)  # Recursively ask again if the input is invalid
