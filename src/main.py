from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.graph import get_rag_app
from src.config import HOST, PORT

api = FastAPI(title="Enterprise RAG Agent API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@api.get("/health")
def health_check():
    return {"status": "healthy"}

@api.post("/query", response_model=QueryResponse)
async def process_rag_query(request: QueryRequest):
    try:
        rag_app = get_rag_app()
        inputs = {"messages": [HumanMessage(content=request.query)]}
        result = await rag_app.ainvoke(inputs)
        final_answer = result["messages"][-1].content
        return QueryResponse(answer=final_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:api", host=HOST, port=PORT, reload=False)
