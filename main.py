from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import voyageai
from supabase import create_client
from chunking import chunk_text
from anthropic import Anthropic

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
    "https://recall-blue.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Client Initializations
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

@app.get("/health")
def health():
    return {"status": "ok"}

class IngestRequest(BaseModel):
    document_id: str

@app.post("/ingest")
def ingest(req: IngestRequest):
    doc = supabase.table("documents").select("title, content").eq("id", req.document_id).single().execute()
    title = doc.data["title"]
    content = doc.data["content"]

    full_text = f"Title: {title}\n\n{content}"
    chunks = chunk_text(full_text)
    embeddings = voyage.embed(chunks, model="voyage-3-large", input_type="document").embeddings

    rows = [
        {"document_id": req.document_id, "chunk_index": i, "content": c, "embedding": e}
        for i, (c, e) in enumerate(zip(chunks, embeddings))
    ]
    supabase.table("chunks").insert(rows).execute()

    return {"chunks_created": len(rows)}

class QueryRequest(BaseModel):
    workspace_id: str
    question: str

@app.post("/query")
def query(req: QueryRequest):
    q_embedding = voyage.embed([req.question], model="voyage-3-large", input_type="query").embeddings[0]

    results = supabase.rpc("match_chunks", {
        "query_embedding": q_embedding,
        "match_workspace_id": req.workspace_id,
        "match_count": 5
    }).execute().data

    context = "\n\n".join(f"[Source: {r['title']}]\n{r['content']}" for r in results)

    message = anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"Answer the question using only the context below. Cite sources by title.\n\nContext:\n{context}\n\nQuestion: {req.question}"
        }]
    )

    answer = message.content[0].text
    supabase.table("queries").insert({
        "workspace_id": req.workspace_id, "question": req.question, "answer": answer
    }).execute()

    return {"answer": answer, "sources": [r["title"] for r in results]}