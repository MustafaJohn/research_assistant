# Research Agent

An AI-powered research assistant that discovers relevant research areas and topics by analyzing web sources using DuckDuckGo search, storing them in FAISS vector memory, and generating insights using Google Gemini.

## Features

- 🔍 **Web Research**: Automatically searches and analyzes relevant web sources
- 🧠 **Vector Memory**: Uses FAISS for efficient semantic search and storage
- 🤖 **AI Analysis**: Leverages Google Gemini for intelligent research topic generation
- 💬 **Conversation History**: Track research sessions and follow-up questions
- 📄 **Export**: Download results in Markdown format with citations
- 🌐 **Web Interface**: Beautiful, modern UI built with Next.js and Tailwind CSS
- 🚀 **Production Ready**: Proper error handling, logging, and configuration management

## Project Structure

```
research-agent/
├── agent/                 # Backend Python code
│   ├── agents/           # Agent implementations
│   ├── api.py            # FastAPI web server
│   ├── memory/           # Vector and graph memory
│   ├── orchestration/    # LangGraph orchestration
│   ├── tools/            # Web scraping and LLM tools
├── frontend/             # frontend
│   └── index.html          
└── README.md
```

## Setup

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   cd agent
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

3. **Run the API server:**
   ```bash
   # Development
   python -m uvicorn api.main:app --reload

   # Or use the CLI
   python main.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

## Deployment

### Backend Deployment (Railway/Render)

1. **Railway:**
   - Push your code to GitHub
   - Connect your repository to Railway
   - Set environment variables in Railway dashboard
   - Deploy!

2. **Render:**
   - Connect your GitHub repository
   - Create a new Web Service
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - Add environment variables

### Frontend Deployment (Vercel)

1. Push your code to GitHub
2. Import your repository in Vercel
3. Set environment variables:
   - `NEXT_PUBLIC_API_URL`: Your backend API URL
4. Deploy!

## Environment Variables

### Backend (.env)
- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `N_RESULTS`: Number of search results (default: 20)
- `RATE_LIMIT`: Rate limit for web requests (default: 1.5)
- `LLM_MODEL`: Gemini model to use (default: gemini-2.5-pro)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `RAG_TOP_K`: Max retrieved chunks included in RAG context (default: 8)
- `MAX_RAG_CONTEXT_CHARS`: Hard cap for RAG context size sent to summarizer (default: 12000)
- `MAX_CHUNKS_PER_PAPER`: Max chunks embedded from each paper abstract (default: 4)
- `MAX_ABSTRACT_CHARS`: Max abstract chars retained from source APIs (default: 1800)

### Frontend (.env.local)
- `NEXT_PUBLIC_API_URL`: Backend API URL

## Usage

### CLI Mode
```bash
cd agent
python main.py
```

### Web Interface
1. Start the backend API
2. Start the frontend
3. Open `http://localhost:3000`
4. Enter a research topic and click "Research"

## API Endpoints

- `POST /api/research` - Create a new research job
- `GET /api/jobs/{job_id}` - Get job status
- `GET /api/jobs` - List all jobs
- `DELETE /api/jobs/{job_id}` - Delete a job
- `GET /api/export/{job_id}` - Export job results
- `POST /api/conversation` - Create conversation research
- `GET /api/conversations/{conversation_id}` - Get conversation history

## Development

### Adding New Features

1. **New Agent**: Add to `agents/` directory
2. **New Tool**: Add to `tools/` directory
3. **Update Graph**: Modify `orchestration/graph.py`

### Testing

```bash
# Backend tests (when implemented)
cd agent
pytest

# Frontend tests
cd frontend
npm test
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
