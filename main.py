# Install required libraries
#!pip install -q numpy scikit-learn transformers sentence-transformers torch

# Import libraries
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google.colab import files

class BookDiscussionChatbot:
    def __init__(self):
        # Initialize models and load data
        print("Initializing book discussion chatbot...")
        self.file_upload()
        self.load_data()
        self.initialize_models()
        print("Chatbot ready! Let's discuss the book.")
    
    def file_upload(self):
        """Handle file upload in Colab"""
        print("\nPlease upload your book text file:")
        uploaded = files.upload()
        for filename in uploaded.keys():
            self.book_filename = filename
            print(f"Uploaded {filename}")
    
    def load_data(self):
        """Load and preprocess the book dataset"""
        with open(self.book_filename, 'r', encoding='utf-8') as file:
            self.original_text = file.read()
        
        # Extract metadata (title, author) from the beginning of the text
        self.extract_metadata()
        
        # Split into meaningful chunks (paragraphs)
        self.paragraphs = [p.strip() for p in self.original_text.split('\n\n') if p.strip()]
        
        # Create cleaned versions for processing
        self.cleaned_paragraphs = [self.preprocess_text(p) for p in self.paragraphs]
    
    def extract_metadata(self):
        """Extract book title and author from the text"""
        # Common patterns in Project Gutenberg texts
        title_match = re.search(r'Title:\s*(.+)\s*Author:', self.original_text)
        author_match = re.search(r'Author:\s*(.+)\s*Release Date:', self.original_text)
        
        self.book_title = title_match.group(1).strip() if title_match else "this book"
        self.book_author = author_match.group(1).strip() if author_match else "unknown author"
        
        print(f"\nLoaded: {self.book_title} by {self.book_author}")
    
    def initialize_models(self):
        """Initialize NLP models"""
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_paragraphs)
        
        # Sentence Transformer for semantic search
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.paragraph_embeddings = self.sentence_model.encode(self.cleaned_paragraphs)
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
    def find_relevant_passages(self, query, top_n=3):
        """Find the most relevant passages from the book"""
        # Preprocess query
        cleaned_query = self.preprocess_text(query)
        
        # TF-IDF similarity
        query_vec = self.vectorizer.transform([cleaned_query])
        tfidf_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Semantic similarity
        query_embedding = self.sentence_model.encode([cleaned_query])
        semantic_similarities = cosine_similarity(query_embedding, self.paragraph_embeddings).flatten()
        
        # Combined score
        combined_scores = 0.5 * semantic_similarities + 0.5 * tfidf_similarities
        
        # Get top N passages
        top_indices = np.argsort(combined_scores)[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0.2:  # Minimum similarity threshold
                results.append({
                    'text': self.paragraphs[idx],
                    'similarity': combined_scores[idx]
                })
        
        return results
    
    def generate_response(self, query):
        """Generate a response based strictly on the book content"""
        # Handle metadata questions first
        if 'author' in query.lower():
            return f"The author of {self.book_title} is {self.book_author}."
        
        if 'title' in query.lower():
            return f"The title of this book is {self.book_title}."
        
        # Find relevant passages
        passages = self.find_relevant_passages(query)
        
        if not passages:
            return "I couldn't find relevant information about that in the book. Could you ask about something else?"
        
        # Build response with the most relevant passages
        response = ["Here's what I found in the book:"]
        for i, passage in enumerate(passages[:2]):  # Show top 2 most relevant
            text = passage['text']
            if len(text) > 300:  # Trim long passages
                text = text[:300] + "..."
            response.append(f"\n{text}")
        
        return "\n".join(response)
    
    def discuss(self):
        """Start interactive discussion about the book"""
        print(f"\nLet's discuss {self.book_title} by {self.book_author}.")
        print("Ask me about anything in the book, or type 'exit' to quit.\n")
        
        while True:
            try:
                user_input = input("You: ")
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nThanks for the discussion! Goodbye.")
                    break
                
                # Get and display response
                response = self.generate_response(user_input)
                print(f"\nBot: {response}\n")
            
            except KeyboardInterrupt:
                print("\nDiscussion ended. Goodbye!")
                break
            except Exception as e:
                print(f"\nSorry, I encountered an error: {str(e)}")
                continue

# Run the chatbot
if __name__ == "__main__":
    chatbot = BookDiscussionChatbot()
    chatbot.discuss()
