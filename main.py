#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OwlAI main entry point
"""

import logging
import logging.config
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from multiprocessing import freeze_support
import uvicorn
from contextlib import asynccontextmanager
import json
from fastapi.responses import StreamingResponse
from typing import Optional

from owlai.nest import AgentManager
from owlai.config.agents import OWL_AGENTS_CONFIG
from owlai.services.system import is_dev
from owlai.services.telemetry import RequestLatencyTracker

logger = logging.getLogger(__name__)


# Pydantic models for API
class AgentInfo(BaseModel):
    name: str
    description: str

    class Config:
        from_attributes = True


class ColorTheme(BaseModel):
    primary: str
    secondary: str

    class Config:
        from_attributes = True


class AgentDetails(BaseModel):
    id: str
    name: str
    description: str
    welcome_title: str
    owl_image_url: str
    color_theme: ColorTheme
    default_queries: List[str]

    class Config:
        from_attributes = True


class AgentResponse(BaseModel):
    id: str
    name: str
    description: str

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    agent_id: str
    question: str
    query_id: str
    session_id: str

    class Config:
        from_attributes = True


class QueryResponse(BaseModel):
    agent_id: str
    question: str
    answer: str


# Global agent manager instance
agent_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI application"""
    # Startup
    global agent_manager
    if agent_manager is None:
        agent_manager = AgentManager(
            agents_config=OWL_AGENTS_CONFIG, enable_cleanup=True
        )
        logger.info("AgentManager initialized successfully")
    yield
    # Shutdown
    agent_manager = None


# Initialize FastAPI app
app = FastAPI(
    title="OwlAI API",
    description="API for interacting with OwlAI agents",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

FRONTEND_AGENT_DATA = {
    "rag-droit-general-pinecone": {
        "name": "Marianne",
        "description": "Marianne est une petite chouette qui répond à vos questions et demandes sur le droit français. Attention de ne pas prendre trop au sérieux les petites chouettes d'OwlAI, leurs réponses sont fournies à titre expérimental. Marianne est open source et 100% gratuite mais pas encore tout à fait au point... Notre but est l'amélioration continue alors laissez-nous vos commentaires!",
        "default_queries": OWL_AGENTS_CONFIG["rag-droit-general-pinecone"][
            "default_queries"
        ],
        "image_url": "Marianne.jpg",
        "color_theme": {
            "primary": "#0055A4",  # French blue
            "secondary": "#FFFFFF",  # White
        },
        "welcome_title": "Voici Marianne, une intelligence artificielle sur le droit français",
    },
    "fr-law-qna-complete": {
        "name": "Marianne",
        "description": "Marianne est une petite chouette qui répond à vos questions et demandes sur le droit français. Attention de ne pas prendre trop au sérieux les petites chouettes d'OwlAI, leurs réponses sont fournies à titre expérimental. Marianne est open source et 100% gratuite mais pas encore tout à fait au point... Notre but est l'amélioration continue alors laissez-nous vos commentaires!",
        "default_queries": OWL_AGENTS_CONFIG["fr-law-qna-complete"]["default_queries"],
        "image_url": "Marianne.jpg",
        "color_theme": {
            "primary": "#0055A4",  # French blue
            "secondary": "#FFFFFF",  # White
        },
        "welcome_title": "Voici Marianne, une intelligence artificielle sur le droit français",
    },
}

OPTIONAL_AGENTS = {
    "rag-naruto": {
        "name": "Kiyomi Uchiha",
        "description": "Ask me about Naruto (spoiler alert!)",
        "default_queries": OWL_AGENTS_CONFIG["rag-naruto"]["default_queries"],
        "image_url": "Kiyomi.jpg",
        "color_theme": {
            "primary": "#000000",
            "secondary": "#FF0000",
        },
        "welcome_title": "Fan of the anime series Naruto.",
    },
    "rag-droit-fiscal": {
        "name": "Marine",
        "description": "Posez vos questions sur le droit fiscal",
        "default_queries": OWL_AGENTS_CONFIG["rag-droit-fiscal"]["default_queries"],
        "image_url": "Nathalie.jpg",
        "color_theme": {
            "primary": "#000000",
            "secondary": "#FF0000",
        },
        "welcome_title": "Expert en droit fiscal",
    },
    "rag-droit-admin": {
        "name": "Nathalie",
        "description": "Posez vos questions sur le droit administratif",
        "default_queries": OWL_AGENTS_CONFIG["rag-droit-admin"]["default_queries"],
        "image_url": "Nathalie.jpg",
        "color_theme": {
            "primary": "#000000",
            "secondary": "#FF0000",
        },
        "welcome_title": "Expert en droit administratif",
    },
}


@app.get("/agents/info", response_model=List[AgentInfo])
async def get_agents_info():
    """Get detailed information about all agents"""
    agent_keys = agent_manager.get_agents_keys()
    return [
        AgentInfo(
            name=FRONTEND_AGENT_DATA[agent_key]["name"],
            description=FRONTEND_AGENT_DATA[agent_key]["description"],
        )
        for agent_key in agent_keys
        if agent_key in FRONTEND_AGENT_DATA
    ]


@app.post("/query", response_model=QueryResponse)
async def query_agent(payload: QueryRequest):
    """Query an agent with a question"""
    logger.info(
        f"Received query request from agent {payload.agent_id}: {payload.question}"
    )

    if payload.agent_id not in FRONTEND_AGENT_DATA:
        logger.error(f"Agent {payload.agent_id} not found")
        raise HTTPException(
            status_code=404, detail=f"Agent {payload.agent_id} not found"
        )

    logger.info(f"Invoking agent: {payload.agent_id}")
    response = agent_manager.invoke_agent(payload.agent_id, payload.question)

    if response is None:
        logger.error(f"Failed to get response from agent {payload.agent_id}")
        raise HTTPException(status_code=500, detail="Failed to get response from agent")

    logger.info(f"Successfully got response from agent {payload.agent_id}")
    return QueryResponse(
        agent_id=payload.agent_id, question=payload.question, answer=response
    )


@app.get("/agents/{agent_id}/details", response_model=AgentDetails)
async def get_agent_details(agent_id: str):
    """Get detailed information about a specific agent"""
    if agent_id not in FRONTEND_AGENT_DATA:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_data = FRONTEND_AGENT_DATA[agent_id]
    return AgentDetails(
        id=agent_id,
        name=agent_data["name"],
        description=agent_data["description"],
        welcome_title=agent_data["welcome_title"],
        owl_image_url=f"/public/{agent_data['image_url']}",
        color_theme=ColorTheme(**agent_data["color_theme"]),
        default_queries=agent_data["default_queries"],
    )


@app.get("/agents/{agent_id}/default-queries", response_model=List[str])
async def get_default_queries(agent_id: str) -> List[str]:
    """Get default queries for a specific agent"""
    if agent_id not in FRONTEND_AGENT_DATA:
        raise HTTPException(status_code=404, detail="Agent not found")

    return FRONTEND_AGENT_DATA[agent_id]["default_queries"]


queryid_to_messageid_map = {}


@app.post("/stream-query")
async def stream_query(payload: QueryRequest):
    """Stream a response from an agent"""
    # Create latency tracker
    latency = RequestLatencyTracker(payload.query_id)

    logger.info(
        f"Received streaming query request from agent {payload.agent_id}: {payload.question}"
    )

    # Verify agent exists
    if payload.agent_id not in FRONTEND_AGENT_DATA:
        logger.error(f"Agent {payload.agent_id} not found")
        raise HTTPException(
            status_code=404, detail=f"Agent {payload.agent_id} not found"
        )

    async def generate():
        first_token_logged = False
        try:
            latency.mark("generate_start")
            logger.info(f"Streaming query for agent {payload.agent_id}")
            # Get the agent instance
            agent = agent_manager.get_agent("rag-droit-general-pinecone")
            latency.mark("agent_initialized")

            logger.info(f"Agent {payload.agent_id} found")

            # Stream the response
            async for chunk in agent.stream_message(payload.question):
                if not first_token_logged:
                    latency.mark("first_token_generated")
                    first_token_logged = True
                yield f"event: message\ndata: {json.dumps({'content': chunk})}\n\n"

            # After streaming is complete
            latency.mark("streaming_complete")

            if agent._conversation_id and agent._memory:
                message_id = agent._memory.get_last_message_id(agent._conversation_id)
                logger.debug(f"Message ID: {message_id}")
                if message_id:
                    global queryid_to_messageid_map
                    logger.debug(
                        f"Mapping query_id {payload.query_id} to message_id {message_id}"
                    )
                    queryid_to_messageid_map[payload.query_id] = message_id
                    yield f"event: complete\ndata: {json.dumps({'message_id': str(message_id)})}\n\n"
            else:
                raise Exception("No conversation ID or memory found for agent")

            # Log final latency breakdown at debug level
            latencies = latency.get_latency_breakdown()
            logger.debug(
                f"Request {payload.query_id} complete - Latency breakdown: {latencies}"
            )

        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


##################################################################


# Additional models for new features
class FeedbackRequest(BaseModel):
    query_id: str
    agent_id: str
    rating: int
    comment: Optional[str] = None

    class Config:
        from_attributes = True


class ContactFormRequest(BaseModel):
    name: str
    email: str
    message: str

    class Config:
        from_attributes = True


class DocumentChunk(BaseModel):
    id: str
    content: str
    source: str
    relevance_score: float

    class Config:
        from_attributes = True


def map_agent_data(agent_name):
    """Helper function to map agent data to AgentDetails model."""
    agent_data = FRONTEND_AGENT_DATA[agent_name]
    return AgentDetails(
        id=agent_name,
        name=agent_data["name"],
        description=agent_data["description"],
        welcome_title=agent_data["welcome_title"],
        owl_image_url=f"/public/{agent_data['image_url']}",
        color_theme=ColorTheme(**agent_data["color_theme"]),
        default_queries=agent_data["default_queries"],
    )


@app.get("/agents", response_model=List[AgentDetails])
async def list_agents():
    """Get list of available agents with their details"""
    # including only rag agents for backward compatibility
    agent_names = [
        name for name in agent_manager.get_agents_keys() if name.startswith("rag-")
    ]
    agents = [
        map_agent_data(agent_name)
        for agent_name in agent_names
        if agent_name in FRONTEND_AGENT_DATA
    ]
    return agents


@app.get("/default-agent", response_model=AgentDetails)
def get_default_agent():
    """Get the default agent for the single-agent page."""
    agent_name = agent_manager.get_focus_owl().name
    if not agent_name:
        raise HTTPException(status_code=404, detail="No agents available")
    return map_agent_data(agent_name)


# Feedback system endpoints
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for a query response.
    Rating should be between 1 and 5.
    """
    logger.info(f"Received feedback request: {feedback.dict()}")
    logger.debug(f"Current query_id mapping state: {queryid_to_messageid_map}")

    if not 1 <= feedback.rating <= 5:
        logger.warning(f"Invalid rating value: {feedback.rating}")
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

    # Get the message_id from the mapping
    message_id = queryid_to_messageid_map.get(feedback.query_id)
    if not message_id:
        logger.error(
            f"Query ID {feedback.query_id} not found in mapping. Available query IDs: {list(queryid_to_messageid_map.keys())}"
        )
        raise HTTPException(
            status_code=404,
            detail="Query ID not found. The message might have expired or was not properly saved.",
        )

    try:
        # Store the feedback in the database
        agent_manager.memory.log_feedback(
            message_id=message_id, score=feedback.rating, comments=feedback.comment
        )
        logger.info(f"Feedback stored successfully for message {message_id}")
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to store feedback")


@app.post("/contact")
async def submit_contact_form(contact: ContactFormRequest):
    """Submit contact form feedback."""
    logger.info(f"Received contact form submission from {contact.email}")
    return {"status": "success", "message": "Contact form submitted"}


# Document chunks visualization endpoint
@app.get("/query/{query_id}/chunks")
async def get_query_chunks(query_id: str):
    """Get document chunks used to answer a specific query."""
    logger.info(f"Getting chunks for query {query_id}")

    # Get message_id from mapping
    message_id = queryid_to_messageid_map.get(query_id)
    if not message_id:
        logger.error(f"Query ID {query_id} not found in mapping")
        raise HTTPException(
            status_code=404,
            detail="Query ID not found. The message might have expired or was not properly saved.",
        )

    # Get the preceding tool message
    tool_message = agent_manager.memory.get_preceding_tool_message(message_id)
    if not tool_message:
        logger.debug(f"No tool message found for message {message_id}")
        return []
    # Parse the content into chunks
    chunks = []
    content = tool_message["content"]

    # logger.debug(f"Content: {content}")

    # Split content into numbered sections
    import re

    # Split by numbered sections
    sections = re.split(r"(?=\d+\.\s*\[Source:)", content.strip())

    for section in sections:
        if not section.strip():
            continue

        # Extract source, score and content using regex
        match = re.match(
            r"\d+\.\s*\[Source: (.*?)(?:\s*-\s*Score:\s*(-?\d*\.?\d+))?\](.*)",
            section,
            re.DOTALL,
        )
        if match:
            source = match.group(1).strip()
            score = (
                float(match.group(2)) if match.group(2) else -10
            )  # Default to -10 if no score
            content = match.group(3).strip()
            # logger.debug(f"Source: {source}, Score: '{score}'")
            chunks.append(
                DocumentChunk(
                    id=source,  # The name from [Source : XYZ]
                    content=content,  # The actual content after brackets
                    source=source,  # The file name (same as source)
                    relevance_score=score,  # Use the parsed score
                )
            )

    if not chunks:
        logger.error(f"No valid chunks found in tool message content: {content}")
        raise HTTPException(
            status_code=404,
            detail="Failed to parse source documents for this query.",
        )

    return chunks


# Enhanced logging endpoint
@app.get("/query/{query_id}/logs")
async def get_query_logs(query_id: str):
    """Get detailed logs for a specific query (for development purposes)."""
    # Mock response with sample logs
    logs = {
        "query_id": query_id,
        "timestamp": "2024-04-02T10:00:00Z",
        "processing_time": 1.5,
        "llm_interactions": [
            {
                "timestamp": "2024-04-02T10:00:00Z",
                "type": "prompt",
                "content": "Sample prompt content",
            },
            {
                "timestamp": "2024-04-02T10:00:01Z",
                "type": "response",
                "content": "Sample response content",
            },
        ],
        "tool_invocations": [
            {
                "timestamp": "2024-04-02T10:00:00.5Z",
                "tool": "document_search",
                "parameters": {"query": "sample search"},
                "result": "sample result",
            }
        ],
    }
    return logs


# Version endpoint
@app.get("/version")
async def get_version():
    """Get the current version of OwlAI."""
    return {"version": "0.2.0"}


@app.get("/feedback/all")
async def get_all_feedback():
    """Get all feedback entries with associated query information."""
    # Mock feedback data for demonstration
    mock_feedback = [
        {
            "query_id": "agent1-1",
            "agent_id": "agent1",
            "rating": 5,
            "comment": "Very helpful and accurate response!",
            "timestamp": "2024-04-02T10:00:00Z",
            "query": "Can you explain what constitutes a valid civil contract in French law?",
            "response": "A valid civil contract in French law requires four essential elements: consent (consentement), capacity (capacité), a defined object (objet), and a lawful cause (cause licite). The parties must give their free and informed consent, be legally capable of entering into contracts, agree on a specific and legal purpose, and have a legitimate reason for the contract. Additionally, certain contracts may require specific formalities, such as being in writing or notarized.",
        },
        {
            "query_id": "agent2-1",
            "agent_id": "agent2",
            "rating": 2,
            "comment": "The response was a bit confusing and could be more detailed.",
            "timestamp": "2024-04-02T09:45:00Z",
            "query": "What are the main elements of criminal liability in French law?",
            "response": "Criminal liability in French law involves several elements. There needs to be a legal text defining the offense, and the person must have committed the act intentionally or through negligence.",
        },
        {
            "query_id": "agent1-2",
            "agent_id": "agent1",
            "rating": 4,
            "timestamp": "2024-04-02T09:30:00Z",
            "query": "What are the basic property rights under French civil law?",
            "response": "Under French civil law, property rights (droits réels) include: the right to use (usus), the right to enjoy benefits (fructus), and the right to dispose of property (abusus). These rights are protected by the Civil Code and can only be limited by law or contract. Property owners have extensive rights but must exercise them within legal boundaries and respect neighbors' rights.",
        },
        {
            "query_id": "agent3-1",
            "agent_id": "agent3",
            "rating": 5,
            "comment": "Perfect explanation, exactly what I needed!",
            "timestamp": "2024-04-02T09:15:00Z",
            "query": "What is the process for filing an administrative appeal?",
            "response": "To file an administrative appeal in France, you must first identify whether you need a recours gracieux (appeal to the same authority) or recours hiérarchique (appeal to a higher authority). The appeal must be filed within 2 months of the decision. Include all relevant documents, clearly state your arguments, and keep proof of submission. The administration then has 2 months to respond, and silence is considered a rejection.",
        },
        {
            "query_id": "agent2-2",
            "agent_id": "agent2",
            "rating": 1,
            "comment": "The answer was not relevant to my question.",
            "timestamp": "2024-04-02T09:00:00Z",
            "query": "What constitutes legitimate self-defense under French penal law?",
            "response": "Self-defense in French law requires proportionality between the threat and the response. The danger must be immediate and the response necessary.",
        },
    ]
    return mock_feedback


@app.get("/query/{query_id}")
async def get_shared_query(query_id: str):
    """
    Retrieve a shared query and its response.
    This is a mock implementation - in production, you would fetch this from your database.
    """
    # Mock data - replace with actual database query in production
    mock_shared_queries = {
        "example-query-1": {
            "question": "What is artificial intelligence?",
            "response": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines...",
            "timestamp": "2024-03-20T10:30:00Z",
        }
    }

    if query_id not in mock_shared_queries:
        raise HTTPException(status_code=404, detail="Shared content not found")

    return mock_shared_queries[query_id]


def main():
    """Main entry point for the application."""
    server = None
    try:
        # Run the FastAPI server
        config = uvicorn.Config(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=is_dev,  # Enable auto-reload only in development
            workers=1,  # Single worker to prevent module reloading
        )
        server = uvicorn.Server(config)
        server.run()

    except KeyboardInterrupt:
        logger.info("Application terminated by the user")
        if server:
            server.should_exit = True
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception(e)
    finally:
        if server:
            server.should_exit = True


if __name__ == "__main__":
    # Required for Windows multiprocessing

    freeze_support()
    main()
