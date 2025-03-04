# Mental-Health-Chatbot

## Elle 💙 - Mental Health Companion & Knowledge Base

### Project Overview
Elle is a multi-modal mental health AI assistant that adapts its therapeutic approach dynamically based on user needs, while also providing a knowledge base for document-based question answering. Built for the **Sarvam AI Litmus Test (ML Focus)**, Elle demonstrates advanced LLM integration techniques with a focus on **context management, multi-modal interactions, and specialized domain adaptation**.

![Elle Mental Health Companion](https://img.icons8.com/color/96/000000/mental-health.png)

---

## Key ML/AI Features

### 1. Dynamic Therapeutic Approach Selection
Elle implements an advanced context-aware system that intelligently selects different therapeutic approaches based on user input:

- **Intent Recognition:** Analyzes user messages for emotional patterns and psychological needs.
- **Specialized Therapy Selection:** Routes to one of six therapeutic approaches:
  - General Emotional Support (empathetic listening)
  - Suicide Prevention (crisis intervention)
  - Anger Management (emotion regulation)
  - Motivational Support (for depression)
  - Dialectical Behavior Therapy (DBT)
  - Cognitive Behavioral Therapy (CBT)

### 2. Multi-Modal Interaction
Elle enhances therapeutic interactions using multiple modalities:

- **Text:** Primary communication through text-based chat.
- **Image Generation:** Contextually relevant imagery generated based on therapeutic context.
- **Speech Recognition:** Voice input support for accessibility.
- **Text-to-Speech:** Audio output of responses when enabled.

### 3. Context Management Architecture
Elle implements a sophisticated **context management system**:

- **Conversation Memory:** Full conversation history tracking.
- **Automatic Summarization:** Periodic summarization of conversations using the Gemini model.
- **Memory Integration:** Summary-enhanced prompting that provides concise context.
- **Persistence:** Conversation summaries stored in CSV format for continuity across sessions.

### 4. Retrieval-Augmented Generation (RAG)
The Knowledge Base component demonstrates advanced **RAG techniques**:

- **Document Processing:** PDF text extraction with recursive chunking.
- **Embedding Generation:** Semantic representation using Gemini Embedding model.
- **Similarity Search:** Vector-based retrieval using cosine similarity.
- **Context-Enhanced Generation:** Combines retrieved chunks with user queries for accurate answers.

### 5. Prompt Engineering & Specialized Domain Adaptation

- **Role-Based Prompting:** Detailed system prompts defining therapeutic roles and approaches.
- **Domain-Specific Conditioning:** Mental health specialized prompting.
- **External Knowledge Integration:** YAML-based prompt library for extensibility.
- **Multi-Turn Optimization:** Progressively builds context for coherent multi-turn dialogues.

---

## Technical Architecture

### System Components
- **Frontend:** Streamlit UI with custom CSS for user interaction.
- **Context Management System:**
  - Session state management
  - User identification through UUID
  - Message history tracking
- **LLM Integration Layer:**
  - Gemini Pro for conversation
  - Gemini Flash for knowledge base RAG
  - Embedding models for semantic search
- **Vector Search System:**
  - In-memory vector storage for document embeddings
  - Cosine similarity-based retrieval
- **Multi-Modal Processing:**
  - Speech recognition (Google Speech API)
  - Text-to-speech (pyttsx3)
  - Image suggestion (Gemini + Unsplash)

### Data Flow
1. User inputs query through text or speech.
2. System analyzes query for appropriate therapeutic approach.
3. Context is enhanced with conversation summary.
4. Query + context sent to appropriate Gemini model.
5. Response generation with possible image enhancement.
6. Response presented to user.
7. Conversation updated and periodically summarized.

### ML Models Used
- **Gemini 1.5 Pro:** Main conversation model for therapeutic responses.
- **Gemini 1.5 Flash:** Knowledge base query processing for faster document Q&A.
- **Gemini Embedding Model:** Document and query embedding for semantic search.
- **Google Speech Recognition:** Converting speech to text.

---

## Evaluation and Performance Optimization

- **Therapy Type Tracking:** Monitoring which therapeutic approaches are used most frequently.
- **Session Summary Analysis:** Automatic generation of session insights.
- **Error Handling:** Robust exception handling throughout the pipeline.
- **Caching Strategy:** Embeddings cached in session state for repeated queries.

### Deployment Requirements
- **Python 3.8+ environment**
- Install dependencies:
- **Gemini API key** (set as environment variable or in `.env` file)
- Run the application:

### Scalability Considerations
Elle’s architecture supports scaling through:

- **Stateless Design:** Core processing is stateless and can be horizontally scaled.
- **Separate Concerns:** UI, embedding generation, and LLM inference can scale independently.
- **Vector Database Integration:** Ready for integration with production vector databases.
- **Session Management:** UUID-based user tracking supports distributed sessions.

---

## Future Enhancements

- **Model Distillation:** Creating specialized smaller models for each therapy type.
- **Response Quality Metrics:** Implementing feedback loops for response quality.
- **Streaming Responses:** Adding incremental response rendering.
- **Advanced RAG:** Implementing hybrid search with metadata filtering.
- **Custom Fine-tuning:** Developing specialized models for mental health support.

---

## ML Research Innovations

- **Contextual Therapy Routing:** Automatic selection of therapeutic approach based on semantic content.
- **Multi-Modal Therapeutic Experience:** Combining text, speech, and images for holistic support.
- **Memory Summarization:** Using LLMs to distill conversation into meaningful context.
- **Domain-Specific RAG:** Knowledge base tailored for mental health resources.

---

### Developed by Joel Bansal for the **Sarvam AI Litmus Test - ML Focus**

