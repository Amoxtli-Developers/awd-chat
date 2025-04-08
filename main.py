import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

AWS_REGION = os.environ.get("AWS_REGION")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID")
INFERENCE_PROFILE_ARN = os.environ.get("INFERENCE_PROFILE_ARN")

bedrock_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    try:
        print(f"Making request with text: {request.text}")
        print(f"Using Knowledge Base ID: {KNOWLEDGE_BASE_ID}")
        print(f"Using Inference Profile ARN: {INFERENCE_PROFILE_ARN}")
        
        response = bedrock_client.retrieve_and_generate(
            input={"text": request.text},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": INFERENCE_PROFILE_ARN 
                }
            }
        )
        
        answer = response["output"]["text"]
        return ChatResponse(answer=answer)
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)