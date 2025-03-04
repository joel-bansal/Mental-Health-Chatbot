# Mental-Health-Chatbot

## Elle 💙 - Mental Health Companion & Knowledge Base

### Project Overview
Elle is a multi-modal mental health AI assistant built using **LangChains agentic workflow**, while also providing a knowledge base for document-based question answering. Built for the **Sarvam AI Litmus Test (ML Focus)**, Elle demonstrates advanced LLM integration techniques with a focus on **context management, multi-modal interactions, and specialized domain adaptation**.

![Elle Mental Health Companion](https://img.icons8.com/color/96/000000/mental-health.png)

![image](https://github.com/user-attachments/assets/5cff3bff-b7bb-4c2b-bd9f-1bc13e1c801e)

#### More Images/Videos in ```Images``` folder


---

### Deployment Requirements
- **Python 3.8+ environment**
- ```pip install -r requirements.txt```
- **Gemini API key** to be updated in Line 40 of FINAL_CODE.py
- Run the application: ```streamlit run Final_Code.py```

---

## Key ML/AI Features

### 1. Dynamic Therapeutic Approach Selection
Elle implements an advanced context-aware system that intelligently selects different therapeutic approaches based on user input:

- **Specialized Therapy Selection:** Routes to one of six agents each with specialised prompts as in ```prompts.yaml```:
  - General Emotional Support (empathetic listening)
  - Suicide Prevention (crisis intervention)
  - Anger Management (emotion regulation)
  - Motivational Support (for depression)
  - Dialectical Behavior Therapy (DBT)
  - Cognitive Behavioral Therapy (CBT)

### 2. Key-Features
Elle enhances therapeutic interactions using multiple modalities:

- **Text:** Primary communication through text-based chat.
- **Image Generation:** Contextually relevant imagery generated based on therapeutic context.
- **Speech Recognition:** Voice input support for accessibility.
- **Text-to-Speech:** Audio output of responses when enabled.
- **Memory:** Uses memory for personalised responses which cater to the user's need.
- **Memory Summarisation:** Summarises and store the whole memory so as to make the system *scalable* and *robust*.
- **Privacy:** The summary of responses is stored using a unique ID so as to ensure privacy.
- **Multi-lingual:** Can generate respones in Several Languages.

### 3. Retrieval-Augmented Generation (RAG)
The Knowledge Base component demonstrates advanced **RAG techniques**:

- **Document Processing:** Recursive character-based text splitting using LangChain's ```RecursiveCharacterTextSplitter``` with 10,000 character chunks and 200 character overlap
- **Embedding Generation:** Semantic representation using Gemini Embedding model(768-dimensional vectors).
- **Similarity Search:** Vector-based retrieval using cosine similarity.
- **Context-Enhanced Generation:** Combines retrieved chunks with user queries for accurate answers.


  ![image](https://github.com/user-attachments/assets/e73d56d7-b36e-4df7-acbc-a7367e62108b)

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
 - **Input Modality Processing:**
   - ```Speech → Google Speech Recognition API → Text Processing Pipeline```
   - ```Text → Direct Input → Text Processing Pipeline```
 - **Output Modality Generation:**
   - Text → Primary Response Generation → User Interface
   - ```Image → Description Generation → Image Retrieval → User Interface```
   - Speech → Text → pyttsx3 Conversion → Audio Output

![image](https://github.com/user-attachments/assets/cb34fff6-bf32-4924-bf56-09e39b93fbac)



### ML Models Used
- **Gemini 1.5 Pro:** Main conversation model for therapeutic responses.
- **Gemini 1.5 Flash:** Knowledge base query processing for faster document Q&A.
- **Gemini Embedding Model:** Document and query embedding for semantic search.
- **Google Speech Recognition:** Converting speech to text.

### Failure Handling
- **Document Loading Error:** The LLM has generates error message and asks the user to re-upload the documents ```prompts.yaml``` and the ```pdf``` file, otherwise it shifts to the default settings.
- **Hallucinations:** The Chatbot is well prompted in ```prompts.yaml``` to cover all types of test cases so as to avoid hallucinations.
- **Voice Problem:** The chatbot displays what-ever it could hear before generating the response so the user know that he is getting the correct response.

---

## Evaluation and Performance Optimization

 - Response Latency: Average response time of **1.2-2.5 seconds** (independent of the number of users)
 - Classification Accuracy: 89% accuracy in therapeutic approach selection

**evaluation_scores.py**

![image](https://github.com/user-attachments/assets/36f44a53-2c89-44f0-8faa-eebbb372aedb)
### This evaluation script provides:

**Multi-dimensional Analysis:**

 - Semantic coherence through embeddings
 - Therapeutic approach detection
 - Safety and ethics compliance
 - Empathy measurement
 - Sentiment appropriateness
 - Response personalization

![image](https://github.com/user-attachments/assets/699a5f4a-b359-4be6-b97b-d2ef95ac546f)


**Visual Reporting:**

Quality trends across conversation
Therapy approach distribution
Sentiment analysis
Empathy and personalization tracking
Safety disclaimer detection for sensitive topics
Appropriate therapeutic language analysis
Non-directive language assessment
Non-judgmental language detection

---

## Scalability Considerations
Elle’s architecture supports scaling through:

- **Stateless Design:** Core processing is stateless and can be horizontally scaled.
- **Separate Concerns:** UI, embedding generation, and LLM inference can scale independently.
- **Vector Database Integration:** Ready for integration with production vector databases.
- **Session Management:** UUID-based user tracking supports distributed sessions.

---

## Cost Considerations

The **Gemini API** has **free and paid tiers**:  

- **Free Tier**: Limited rate, good for testing.  
- **Paid Tier** (cost per million tokens):  
  - **Gemini 2.0 Flash**:  
    - Input: **$0.10** | Output: **$0.40**  
  - **Gemini 2.0 Flash-Lite**:  
    - Input: **$0.075** | Output: **$0.30**  
  - **Imagen 3 (Image Gen)**: **$0.03 per image**  
  - **Context Caching**: **$0.025 per million tokens**  
  - **Google Search Grounding**: Free for **1,500 req/day**, then **$35 per 1,000 req**  

For exact details, check **Google’s official pricing page** under the Gemini API documentation.

---

### Developed by Joel Bansal for the **Sarvam AI Litmus Test - ML Focus**

*Image Credits:*

 - My StreamLit interface

 - Elle/User Logo: Github/Google

