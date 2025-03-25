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

from owlai.edwige import AgentManager


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


def load_logger_config():
    """Load logging configuration from logging.yaml"""
    with open("logging.yaml", "r") as logger_config:
        config = yaml.safe_load(logger_config)
        logging.config.dictConfig(config)


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


@app.get("/agents", response_model=List[AgentDetails])
async def list_agents():
    """Get list of available agents with their details"""
    agents_info = agent_manager.get_agents_info()
    return [
        AgentDetails(
            id=f"agent{i+1}",
            name=info.split(": ")[0],
            description=info.split(": ")[1],
            welcome_title=f"Welcome to {info.split(': ')[0]}",
            owl_image_url=f"/public/{info.split(': ')[0]}.png",
            color_theme=ColorTheme(primary="#4A90E2", secondary="#F5A623"),
            default_queries=agent_manager.owls[info.split(": ")[0]].default_queries
            or [],
        )
        for i, info in enumerate(agents_info)
    ]


@app.get("/agents/info", response_model=List[AgentInfo])
async def get_agents_info():
    """Get detailed information about all agents"""
    agents_info = agent_manager.get_agents_info()
    return [
        AgentInfo(name=info.split(": ")[0], description=info.split(": ")[1])
        for info in agents_info
    ]


@app.post("/query", response_model=QueryResponse)
async def query_agent(payload: QueryRequest):
    """Query an agent with a question"""
    logging.info(
        f"Received query request from agent {payload.agent_id}: {payload.question}"
    )

    # Extract agent name from agent_id (e.g., "agent1" -> "French Law Agent Alpha")
    agent_index = int(payload.agent_id.replace("agent", "")) - 1
    agents_info = agent_manager.get_agents_info()

    if agent_index < 0 or agent_index >= len(agents_info):
        logging.error(f"Agent {payload.agent_id} not found")
        raise HTTPException(
            status_code=404, detail=f"Agent {payload.agent_id} not found"
        )

    agent_name = agents_info[agent_index].split(": ")[0]
    logging.info(f"Invoking agent: {agent_name}")

    response = agent_manager.invoke_agent(agent_name, payload.question)

    if response is None:
        logging.error(f"Failed to get response from agent {agent_name}")
        raise HTTPException(status_code=500, detail="Failed to get response from agent")

    logging.info(f"Successfully got response from agent {agent_name}")
    return QueryResponse(
        agent_id=payload.agent_id, question=payload.question, answer=response
    )


@app.get("/agents/{agent_id}/details", response_model=AgentDetails)
async def get_agent_details(agent_id: str):
    """Get detailed information about a specific agent"""
    agent_index = int(agent_id.replace("agent", "")) - 1
    agents_info = agent_manager.get_agents_info()

    if agent_index < 0 or agent_index >= len(agents_info):
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_name = agents_info[agent_index].split(": ")[0]
    agent = agent_manager.owls[agent_name]

    return AgentDetails(
        id=agent_id,
        name=agent.name,
        description=agent.description,
        welcome_title=f"Welcome to {agent.name}",
        owl_image_url=f"/public/{agent.name}.png",  # You'll need to add this static file
        color_theme=ColorTheme(primary="#4A90E2", secondary="#F5A623"),
        default_queries=agent.default_queries or [],
    )


@app.get("/agents/{agent_id}/default-queries", response_model=List[str])
async def get_default_queries(agent_id: str) -> List[str]:
    """Get default queries for a specific agent"""
    agent_index = int(agent_id.replace("agent", "")) - 1
    agents_info = agent_manager.get_agents_info()

    if agent_index < 0 or agent_index >= len(agents_info):
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_name = agents_info[agent_index].split(": ")[0]
    agent = agent_manager.owls[agent_name]

    return agent.default_queries or []


def main():
    """Main entry point for OwlAI"""
    # Load environment variables
    load_dotenv()

    # Determine if we're in development mode
    is_development = os.getenv("OWLAI_ENV", "production").lower() == "development"

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
