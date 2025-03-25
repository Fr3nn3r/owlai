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
from pydantic import BaseModel
from dotenv import load_dotenv
from multiprocessing import freeze_support
import uvicorn
from contextlib import asynccontextmanager

from owlai.edwige import AgentManager


# Pydantic models for API
class AgentInfo(BaseModel):
    name: str
    description: str


class AgentInvokeRequest(BaseModel):
    agent_name: str
    message: str


class AgentInvokeResponse(BaseModel):
    response: str


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
    agent_manager = AgentManager()
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


@app.get("/agents", response_model=List[str])
async def list_agents():
    """Get list of available agent names"""
    return agent_manager.get_agents_names()


@app.get("/agents/info", response_model=List[AgentInfo])
async def get_agents_info():
    """Get detailed information about all agents"""
    agents_info = agent_manager.get_agents_info()
    return [
        AgentInfo(name=info.split(": ")[0], description=info.split(": ")[1])
        for info in agents_info
    ]


@app.post("/agents/invoke", response_model=AgentInvokeResponse)
async def invoke_agent(request: AgentInvokeRequest):
    """Invoke an agent with a message"""
    if request.agent_name not in agent_manager.get_agents_names():
        raise HTTPException(
            status_code=404, detail=f"Agent {request.agent_name} not found"
        )

    response = agent_manager.invoke_agent(request.agent_name, request.message)
    if response is None:
        raise HTTPException(status_code=500, detail="Failed to get response from agent")

    return AgentInvokeResponse(response=response)


def main():
    """Main entry point for OwlAI"""
    # Load environment variables
    load_dotenv()

    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
    )


if __name__ == "__main__":
    # Required for Windows multiprocessing
    freeze_support()
    main()
