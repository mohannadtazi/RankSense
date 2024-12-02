import json
import random

# Sample subjects
subjects = ["Mathematics", "Physics", "Computer Science", "History", "Literature", "Biology", "Chemistry", "Art", "Economics", "Philosophy"]

# Function to generate content for a random subject
def generate_content(subject):
    content_samples = {
        "Mathematics": "Mathematics is the study of numbers, shapes, and patterns. It is used in a wide range of fields including engineering, economics, and physics.",
        "Physics": "Physics is the branch of science concerned with the nature and properties of matter and energy. It includes subjects like mechanics, thermodynamics, and electromagnetism.",
        "Computer Science": "Computer Science deals with the study of algorithms, data structures, programming languages, and software development.",
        "History": "History is the study of past events, particularly in human affairs. It helps us understand the progression of civilization over time.",
        "Literature": "Literature refers to written works, especially those considered of superior or lasting artistic merit. It includes novels, poems, plays, and short stories.",
        "Biology": "Biology is the study of living organisms, divided into many specialized fields such as genetics, ecology, and physiology.",
        "Chemistry": "Chemistry is the branch of science that studies the composition, structure, properties, and changes of matter.",
        "Art": "Art includes visual arts, such as painting and sculpture, as well as performing arts, such as theater and music. It expresses human creativity.",
        "Economics": "Economics is the social science that studies the production, distribution, and consumption of goods and services.",
        "Philosophy": "Philosophy is the study of fundamental questions regarding existence, reason, knowledge, and ethics. It explores various schools of thought."
    }
    return content_samples.get(subject, "Content not available for this subject.")

# Function to create a list of documents in JSON format
def create_documents():
    documents = []
    for subject in subjects:
        document = {
            "title": f"{subject} Overview",
            "content": generate_content(subject)
        }
        documents.append(document)
    
    # Write the documents to a JSON file
    with open('corpus.json', 'w') as json_file:
        json.dump(documents, json_file, indent=4)

# Create documents
create_documents()

print("Documents have been generated and saved to 'documents.json'.")
