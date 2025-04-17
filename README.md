# AI-Powered Beverage Store Assistant

A sophisticated AI-powered beverage store assistant system that combines face authentication, RAG (Retrieval-Augmented Generation), and image recognition capabilities to provide personalized customer service.

## Features

### 1. Face Authentication
- Real-time face detection and recognition
- Secure customer identification
- Session management for authenticated users
- Anonymous chat option available

### 2. Intelligent Chat System
- Context-aware responses using RAG (Retrieval-Augmented Generation)
- Personalized recommendations based on purchase history
- Support for both SQL-based queries and semantic search
- Multi-turn conversation handling

### 3. Image Recognition
- Visual drink recognition and analysis
- Ingredient and composition detection
- Smart drink recommendations based on visual similarity
- OCR integration for text extraction from images

### 4. Database Integration
- SQLite database for storing product and customer information
- Vector stores for semantic search capabilities
- Purchase history tracking
- Customer preference management

## Technical Architecture

### Core Components
- **Face Authentication**: Using InsightFace for face detection and recognition
- **RAG System**: Combines PhoBERT embeddings with Google's Generative AI
- **Vector Stores**: FAISS for efficient similarity search
- **Web Interface**: Flask + SocketIO for real-time communication

### Key Technologies
- Flask for web server
- Socket.IO for real-time communication
- Google Generative AI for natural language processing
- FAISS for vector similarity search
- PhoBERT for Vietnamese language understanding
- SQLite for data storage

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in `.env`:
   ```
   GOOGLE_API_KEY=your_google_api_key
   HUGGINGFACE_HUB_TOKEN=your_huggingface_token
   ```

5. Initialize the database and vector stores:
   ```bash
   python init_db.py  # Creates and populates the SQLite database
   ```

6. Start the application:
   ```bash
   python app.py
   ```

## Project Structure

```
project/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── utils.py             # Utility functions
├── models/
│   ├── face_auth.py     # Face authentication system
│   ├── rag_system.py    # RAG implementation
│   ├── img_export_info.py # Image analysis
│   └── prompts.py       # System prompts
├── vector_store/        # FAISS vector stores
├── Database.db         # SQLite database
└── requirements.txt    # Project dependencies
```

## API Endpoints

- `/`: Main application interface
- `/authenticate`: Face authentication endpoint
- `/chat`: Chat message handling
- `/process_image`: Image analysis endpoint
- `/confirm_auth`: Authentication confirmation

## Security Considerations

- API keys are stored securely in environment variables
- Face embeddings are stored securely in the database
- SQL injection prevention through query validation
- Secure session management
- Rate limiting on authentication attempts

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Specify your license here]

## Acknowledgments

- InsightFace for face recognition
- Google for Generative AI capabilities
- VinAI for PhoBERT model
- Meta AI for FAISS 