# Advanced Book Based Chatbot
A smart, self-improving book chatbot that answers questions by combining retrieval-based search, QA models, and GPT-2 generation.

This project implements an Advanced Book Chatbot designed to intelligently interact with users about the contents of a given book. It uses a combination of TF-IDF-based retrieval, semantic search with Sentence Transformers, and question-answering models (RoBERTa-SQuAD2) to extract accurate answers from the book. If the chatbot cannot find a strong match, it generates new, contextually relevant responses using GPT-2. The system continuously improves by building a knowledge base and enriching its dataset with newly generated information, allowing it to become smarter over time. It supports file uploading in Google Colab, real-time chat interaction, and incremental learning from conversations.

