from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import json
from PIL import Image
import numpy as np
from embed_img import embed_image
from searchEnv import SearchEnv
from stable_baselines3 import PPO
from streamlit_option_menu import option_menu
import plotly.express as px
import os



model = SentenceTransformer('all-MiniLM-L6-v2')  # A fast, small BERT model
multilingue_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

def search_1(query, documents):
    # Encode the query to obtain its embedding
    query_embedding = model.encode([query])[0]

   
    doc_embeddings = model.encode([doc['content'] for doc in documents])

    # Compute cosine similarities between the query and the document embeddings
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Rank documents by similarity scores
    ranked_docs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    # Return top N results (e.g., 3 results here)
    top_n = 3
    result = [(documents[idx]['title'], score) for idx, score in ranked_docs[:top_n]]
    
    return result


def search_2(text_documents, image_index, query=None, image_query=None, top_n=3):
    # Check if an image query is provided
    if image_query:
        # Convert UploadedFile to a compatible image format (PIL.Image.Image)
        image_query = Image.open(image_query)
        # Embed the image query
        image_query_embedding = embed_image(image_query, path=None)[0]  # Only extract embedding
        query_embedding = image_query_embedding
    elif query:
        # Embed the text query
        query_embedding = multilingue_model.encode([query])[0]
    else:
        raise ValueError("Both query and image_query are None. Provide at least one.")

    text_embeddings = np.vstack([np.array(txt['content']) for txt in text_documents])
    text_similarities = cosine_similarity([query_embedding], text_embeddings)[0]

    # Prepare image embeddings if available
    image_embeddings = np.array([img['embedding'] for img in image_index])  # Ensure correct data format for image embeddings
    image_similarities = cosine_similarity([query_embedding], image_embeddings)[0]

    # Rank text results
    ranked_text = [
        (idx, "text", text_documents[idx]['title'], None, score)
        for idx, score in enumerate(text_similarities)
    ]

    # Rank image results
    ranked_images = [
        (idx, "image", image_index[idx]['path'], image_index[idx]['path'], score)
        for idx, score in enumerate(image_similarities)
    ]

    # Combine and sort results by similarity score
    combined_results = ranked_text + ranked_images
    combined_results = sorted(combined_results, key=lambda x: x[4], reverse=True)

    # Return the top N results
    return combined_results[:top_n]




def display_result_card(title, score, image_path=None):
    st.markdown(f"""
    <div class='result-card'>
        <h4>{title}</h4>
        <p>Relevance Score: {score*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
    if image_path:
        image = Image.open(image_path)
        st.image(image, caption=f"Image Result {idx+1}", use_container_width=True)


def search_with_rl(text_documents, query, model_filename="ppo_model"):
    # Initialize the environment for text search only
    env = SearchEnv(text_documents, query=query)
    
    # Load or train the model
    try:
        model_rl = PPO.load(model_filename)
        print("Loaded pre-trained model.")
    except:
        print("No pre-trained model found. Training new model.")
        model_rl = PPO("MlpPolicy", env, verbose=1)
        model_rl.learn(total_timesteps=10000)
        model_rl.save(model_filename)  # Save after training
    
    # Reset the environment and use the trained model for ranking
    state = env.reset()
    action, _ = model_rl.predict(state)
    
    # Select the document based on the action
    selected_document = text_documents[action]  # Only consider text documents
    
    # Return in the expected format for results
    return [(0, "text", selected_document['title'], None, None)]  # Return just the selected document







# Directory to store files
DB_DIR = "db"
CORPUS_FILE = "text_corpus.json"

# Ensure the folder exists
os.makedirs(DB_DIR, exist_ok=True)

# Function to load the existing corpus
def load_corpus():
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, "r") as f:
            return json.load(f)
    return []

# Function to save the corpus
def save_corpus(corpus):
    with open(CORPUS_FILE, "w") as f:
        json.dump(corpus, f, indent=4)

# Function to save the uploaded file and generate metadata
def process_uploaded_file(uploaded_file):
    file_path = os.path.join(DB_DIR, uploaded_file.name)

    # Save the uploaded file in the DB folder
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Try reading the file with different encodings
    encodings = ["utf-8", "utf-16", "latin-1"]
    file_content = None
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                file_content = f.read()
            break  # Exit the loop if successful
        except UnicodeDecodeError:
            continue  # Try the next encoding

    if file_content is None:
        raise ValueError(f"Could not decode file: {uploaded_file.name}. Unsupported encoding.")

    # Generate embeddings
    file_embeddings = multilingue_model.encode(file_content).tolist()

    # Return metadata for the corpus
    return {
            "title": uploaded_file.name,
            "path": file_path,
            "content": file_embeddings
        }

#########################################################################################################
#########################################################################################################

st.set_page_config(
    page_title="RankSense", 
    layout="wide", 
    page_icon=":mag:",
)



st.markdown("""
<style>
    .big-font {
        font-size: 30px;
        font-weight: bold;
        color: #4CAF50;
    }
    .button {
        background-color: #4CAF50;
        border-radius: 5px;
        color: white;
        font-size: 16px;
    }
    .result-card {
        background-color: #f2f2f2;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
            .css-18e3th9 {{
        background-color: #f2f2f2; 
    }}
               .current-page {{
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #E5C100;
            color: #000;
            font-size: 16px;
        }}
</style>
""", unsafe_allow_html=True)


# Header (smaller than the title)
st.header("RankSense: Intelligent Search Solutions")


with st.sidebar:
    page = option_menu(
        menu_title="Navigation",  # Required
        options=["Introduction", "Text Search", "Multimodal Search", "RL Search", "indexer"],  # Required
        icons=["house", "search", "camera", "robot"],  # Optional
        menu_icon="cast",  # Optional
        default_index=0,  # Optional
    )




# Stylish current page indicator
st.markdown(
    f"""
        <i style="  margin: 0;">You are currently on: <b>{page}</b></i>
   
    """,
    unsafe_allow_html=True,
)


if page == "Introduction":
    st.subheader("Welcome to RankSense!")
    st.write("""
    RankSense is a cutting-edge search platform designed to deliver intelligent and precise search results by utilizing modern AI technologies. 
    Our application integrates multiple search methods:
    
    - **Semantic Text Search**: Use advanced sentence embeddings for highly accurate text matching.
    - **Multimodal Search**: Combine text and image embeddings to provide comprehensive search results across different media types.
    - **Reinforcement Learning-Based Search**: Enhance search relevancy through iterative user feedback and machine learning.

    ### What Can RankSense Do for You?
    1. **Powerful Information Retrieval**: Whether you're searching through text, images, or a combination of both, RankSense ensures you find what you're looking for efficiently.
    2. **Interactive Learning System**: The RL Search mode collects your feedback to improve search quality over time.
    3. **Scalable Applications**: Our platform is suitable for educational, research, and enterprise-level tasks.

    ### Why Choose RankSense?
    - **Accurate Results**: Built with state-of-the-art Sentence Transformers and image models.
    - **Flexible Input Modes**: Support for multimodal searches to suit diverse needs.
    - **User-Centered Design**: Intuitive UI for seamless interaction.

    ### Explore Our Features
    Navigate through the app using the sidebar to try out each feature. We recommend starting with the **Text Search** section to experience our semantic search engine.
    """)

    st.image("assets/DALL·E 2024-12-01 19.51.52 - A visually striking image depicting the collaboration between humans and machines using AI to manage and process information. The artwork features a M.webp", caption="RankSense - Empowering Intelligent Search", use_container_width=True)


elif page == 'Text Search':
    st.subheader('Semantic Search Engine')
    st.write('This is a demo of a semantic search engine using Sentence Transformers.')
    st.write('The search engine is capable of searching through a corpus of text documents.')
    st.write('The search engine uses text embeddings to retrieve relevant results.')


    query = st.text_input('Enter your query:', key='query1')

    with open('corpus.json', 'r') as file:
        documents = json.load(file)  # Load the list of documents


    if st.button('Search', key='search1'):
        result = search_1(query, documents)
        for idx, (doc, score) in enumerate(result):
            display_result_card(doc, score)
            


elif page == 'Multimodal Search':
    st.subheader('Advanced Semantic Search System with Multilingue and Multimodal Search')
    st.write('This is a demo of a semantic search engine using Sentence Transformers and CLIP models.')
    st.write('The search engine is capable of searching through a corpus of text documents and a collection of images.')
    st.write('The search engine uses a combination of text embeddings and image embeddings to retrieve relevant results.')

    text_query = st.text_input('Enter your query:', key='query2')
    query_image = st.file_uploader('Upload an image:', type=['jpg', 'png'], key='image_query')

    # Load documents and image index
    with open('text_corpus.json', 'r') as text_file:
        text_documents = json.load(text_file)  # Load text documents
        for doc in text_documents:
            doc['content'] = np.array(doc['content']) 


    with open('image_index.json', 'r') as image_file:
        image_index = json.load(image_file)  # Load image index
        for img in image_index:
            img['embedding'] = np.array(img['embedding'])

    if st.button('Search', key='search2'):
        if text_query or query_image:
            results = search_2(text_documents, image_index, query=text_query, image_query=query_image)
        
            for idx, (result_idx, result_type, title, image_path, score) in enumerate(results):
                display_result_card(title, score, image_path=image_path if result_type == "image" else None)
        else:
            st.write("Please enter a query or upload an image.")



elif page == 'RL Search':
   
    st.subheader('Advanced Semantic Search with RL-based Feedback')
    st.write('This is a demo of a semantic search engine using Sentence Transformers and RL-based feedback.')
    st.write('The search engine is capable of searching through a corpus of text documents.')
    st.write('The search engine uses text embeddings to retrieve relevant results and collects user feedback to improve search results.')

    text_query = st.text_input('Enter your query:', key='query3')

    # Load documents
    with open('corpus.json', 'r') as text_file:
        text_documents = json.load(text_file)  # Load text documents

    if st.button('Search', key='search3'):
        # Perform RL-based search
        if text_query:   
            results = search_with_rl(text_documents, text_query)
            
            for idx, (result_idx, result_type, title, _, score) in enumerate(results):
                # Display result
                with st.expander(f"Result {idx+1}: {title}"):
                    st.markdown(f"**Title**: {title}")
                    st.markdown(f"**Score**: {score*100 if score else 'N/A'}%")
                    st.button('Learn More', key=f'btn{idx}')
                
                # User feedback buttons (thumbs up/down)
                feedback = st.radio("Is this result relevant?", options=["👍 Yes", "👎 No"], key=f"feedback{idx}")
                if feedback == "👍 Yes":
                    st.write("Thumbs up!")
                elif feedback == "👎 No":
                    st.write("Thumbs down!")




elif page == 'indexer':
    
    

# Streamlit page
    st.title("Upload and Add Documents to Corpus")

# Upload functionality
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

    if uploaded_file is not None:
    # Process the uploaded file
        new_document = process_uploaded_file(uploaded_file)
    
    # Add the document to the corpus
        if st.button("Add to Corpus"):
            corpus = load_corpus()
            corpus.append(new_document)
            save_corpus(corpus)
            st.success(f"File '{uploaded_file.name}' has been saved and added to the corpus!")

# Display the current corpus
    st.subheader("Current Corpus")
    corpus = load_corpus()
    if corpus:
        st.json(corpus)  # Show as JSON for better readability
    else:
        st.write("No documents in the corpus yet.")





# Footer
st.markdown(
    """
    <div class="footer" style="margin-top=:0px;">
        © 2024 RankSense. All rights reserved. | <a href="https://mohannadtazi.vercel.app/">Website</a> | Contact: mohanandtazi.dev@gmail.com
    </div>
    """,
    unsafe_allow_html=True,
)