#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OwlAI main entry point
"""

import logging
import logging.config
import yaml
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from multiprocessing import freeze_support
import uvicorn
from contextlib import asynccontextmanager
import os
import json
from fastapi.responses import StreamingResponse

from owlai.agent_manager import AgentManager
from owlai.db import RAG_AGENTS_CONFIG, OWL_AGENTS_CONFIG
from owlai.owlsys import load_logger_config


# Pydantic models for API
class AgentInfo(BaseModel):
    name: str
    description: str


class ColorTheme(BaseModel):
    primary: str
    secondary: str


class AgentDetails(BaseModel):
    id: str
    name: str
    description: str
    welcome_title: str
    owl_image_url: str
    color_theme: ColorTheme
    default_queries: List[str]


class AgentResponse(BaseModel):
    id: str
    name: str
    description: str


class QueryRequest(BaseModel):
    agent_id: str
    question: str


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
    load_logger_config()
    global agent_manager
    if agent_manager is None:
        agent_manager = AgentManager()
        logging.info("AgentManager initialized successfully")
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


def get_rag_agent_default_queries(agent_name: str) -> List[str]:
    """Get default queries for a specific agent from RAG_AGENTS_CONFIG"""
    return next(
        (
            config["default_queries"]
            for config in RAG_AGENTS_CONFIG
            if config["name"] == agent_name
        ),
        [],
    )


FRONTEND_AGENT_DATA = {
    "rag-naruto-v1": {
        "name": "Kiyomi Uchiha",
        "description": "Ask me about Naruto (spoiler alert!)",
        "default_queries": get_rag_agent_default_queries("rag-naruto-v1"),
        "image_url": "Kiyomi.jpg",
        "color_theme": {
            "primary": "#000000",
            "secondary": "#FF0000",
        },
        "welcome_title": "Fan of the anime series Naruto.",
    },
    "rag-fr-general-law-v1": {
        "name": "Marianne",
        "description": "Une question générale sur le droit français ? (attention je ne retiens pas encore le contexte de la conversation)",
        "default_queries": get_rag_agent_default_queries("rag-fr-general-law-v1"),
        "image_url": "Marianne.jpg",
        "color_theme": {
            "primary": "#0055A4",  # French blue
            "secondary": "#FFFFFF",  # White
        },
        "welcome_title": "Experte en droit français",
    },
    "rag-fr-tax-law-v1": {
        "name": "Marine",
        "description": "Une question sur le droit fiscal français ? (attention je ne retiens pas encore le contexte de la conversation)",
        "default_queries": get_rag_agent_default_queries("rag-fr-tax-law-v1"),
        "image_url": "Marine.jpg",
        "color_theme": {
            "primary": "#FFFFFF",  # White
            "secondary": "#FF0000",  # Red
        },
        "welcome_title": "Experte en droit fiscal français",
    },
    "rag-fr-admin-law-v1": {
        "name": "Nathalie",
        "description": "Une question sur le droit administratif français ? (attention je ne retiens pas encore le contexte de la conversation)",
        "default_queries": get_rag_agent_default_queries("rag-fr-admin-law-v1"),
        "image_url": "Nathalie.jpg",
        "color_theme": {
            "primary": "#4A90E2",
            "secondary": "#F5A623",
        },
        "welcome_title": "Experte en droit administratif français",
    },
}


@app.get("/agents", response_model=List[AgentDetails])
async def list_agents():
    """Get list of available agents with their details"""
    agent_names = agent_manager.get_agents_names()
    agents = []

    for agent_name in agent_names:
        if agent_name in FRONTEND_AGENT_DATA:
            agent_data = FRONTEND_AGENT_DATA[agent_name]
            agents.append(
                AgentDetails(
                    id=agent_name,
                    name=agent_data["name"],
                    description=agent_data["description"],
                    welcome_title=agent_data["welcome_title"],
                    owl_image_url=f"/public/{agent_data['image_url']}",
                    color_theme=ColorTheme(**agent_data["color_theme"]),
                    default_queries=agent_data["default_queries"],
                )
            )

    return agents


@app.get("/agents/info", response_model=List[AgentInfo])
async def get_agents_info():
    """Get detailed information about all agents"""
    agent_names = agent_manager.get_agents_names()
    return [
        AgentInfo(
            name=FRONTEND_AGENT_DATA[agent_name]["name"],
            description=FRONTEND_AGENT_DATA[agent_name]["description"],
        )
        for agent_name in agent_names
        if agent_name in FRONTEND_AGENT_DATA
    ]


@app.post("/query", response_model=QueryResponse)
async def query_agent(payload: QueryRequest):
    """Query an agent with a question"""
    logging.info(
        f"Received query request from agent {payload.agent_id}: {payload.question}"
    )

    if payload.agent_id not in FRONTEND_AGENT_DATA:
        logging.error(f"Agent {payload.agent_id} not found")
        raise HTTPException(
            status_code=404, detail=f"Agent {payload.agent_id} not found"
        )

    logging.info(f"Invoking agent: {payload.agent_id}")
    response = agent_manager.invoke_agent(payload.agent_id, payload.question)

    if response is None:
        logging.error(f"Failed to get response from agent {payload.agent_id}")
        raise HTTPException(status_code=500, detail="Failed to get response from agent")

    logging.info(f"Successfully got response from agent {payload.agent_id}")
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


@app.post("/stream-query")
async def stream_query(payload: QueryRequest):
    """Stream a response from an agent"""
    logging.info(
        f"Received streaming query request from agent {payload.agent_id}: {payload.question}"
    )

    # Verify agent exists
    if payload.agent_id not in FRONTEND_AGENT_DATA:
        logging.error(f"Agent {payload.agent_id} not found")
        raise HTTPException(
            status_code=404, detail=f"Agent {payload.agent_id} not found"
        )

    async def generate():
        try:
            # Get the agent instance
            agent = agent_manager.owls[payload.agent_id]

            # Stream the response
            async for chunk in agent.stream_message(payload.question):
                yield f"data: {json.dumps({'content': chunk})}\n\n"

        except Exception as e:
            logging.error(f"Error streaming response: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def main():
    """Main entry point for OwlAI"""
    # Load environment variables
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing required OPENAI_API_KEY environment variable")

    # Determine if we're in development mode
    is_development = os.getenv("OWL_ENV", "production").lower() == "development"

    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=is_development,  # Enable auto-reload only in development
        workers=1,  # Single worker to prevent module reloading
    )


if __name__ == "__main__":
    # Required for Windows multiprocessing
    freeze_support()
    main()
