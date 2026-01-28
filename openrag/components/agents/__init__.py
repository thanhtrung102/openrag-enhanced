"""
Agentic RAG Components for OpenRAG

This module provides agentic capabilities for enhanced RAG:
- Multi-step query orchestration
- Query planning and decomposition
- Tool-augmented retrieval

PR #5: Agentic RAG Integration
"""

from .orchestrator import (
    AgentState,
    StepType,
    ExecutionStep,
    AgentResult,
    AgentConfig,
    AgenticRAGOrchestrator,
)

from .planner import (
    QueryComplexity,
    QueryAnalysis,
    PlanStep,
    QueryPlanner,
)

from .tools import (
    Tool,
    ToolResult,
    CalculatorTool,
    WebSearchTool,
    CodeExecutorTool,
    DateTimeTool,
    ToolRegistry,
    create_default_tools,
)

__all__ = [
    # Orchestrator
    "AgentState",
    "StepType",
    "ExecutionStep",
    "AgentResult",
    "AgentConfig",
    "AgenticRAGOrchestrator",
    # Planner
    "QueryComplexity",
    "QueryAnalysis",
    "PlanStep",
    "QueryPlanner",
    # Tools
    "Tool",
    "ToolResult",
    "CalculatorTool",
    "WebSearchTool",
    "CodeExecutorTool",
    "DateTimeTool",
    "ToolRegistry",
    "create_default_tools",
]
