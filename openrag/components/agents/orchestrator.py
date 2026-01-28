"""
Agentic RAG Orchestrator

Coordinates multi-step RAG workflows with tool calling,
planning, and iterative refinement capabilities.

Supports:
- Multi-step query decomposition
- Tool-augmented retrieval
- Self-reflection and correction
- Iterative answer refinement
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


class StepType(Enum):
    """Types of execution steps."""
    RETRIEVE = "retrieve"
    REASON = "reason"
    TOOL_CALL = "tool_call"
    SYNTHESIZE = "synthesize"
    REFLECT = "reflect"


@dataclass
class ExecutionStep:
    """A single step in the agent's execution."""
    step_id: str
    step_type: StepType
    description: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    status: str = "pending"
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentResult:
    """Result from agent execution."""
    query: str
    answer: str
    steps: List[ExecutionStep]
    sources: List[Dict[str, Any]]
    confidence: float
    total_duration_ms: float
    state: AgentState
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "sources": self.sources,
            "confidence": self.confidence,
            "total_duration_ms": self.total_duration_ms,
            "state": self.state.value,
            "metadata": self.metadata,
        }


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_steps: int = 10
    max_retries: int = 2
    enable_reflection: bool = True
    enable_planning: bool = True
    confidence_threshold: float = 0.7
    timeout_seconds: float = 60.0


class AgenticRAGOrchestrator:
    """
    Orchestrates multi-step RAG workflows.

    The orchestrator coordinates between planning, retrieval,
    tool execution, and synthesis to answer complex queries
    that may require multiple steps.

    Example:
        orchestrator = AgenticRAGOrchestrator(
            llm_client=llm,
            retriever=retriever,
            tools=tools,
        )

        result = await orchestrator.run(
            query="Compare Python and Rust for systems programming"
        )

        print(f"Answer: {result.answer}")
        print(f"Steps taken: {len(result.steps)}")

    Integration with OpenRAG:
        # In the RAG pipeline
        orchestrator = AgenticRAGOrchestrator(
            llm_client=openai_client,
            retriever=rag_retriever,  # OpenRAG's retriever
            tools=[
                WebSearchTool(),
                CalculatorTool(),
                CodeExecutorTool(),
            ],
        )

        # Handle complex queries with agents
        async def agentic_rag(query: str):
            if requires_multi_step(query):
                return await orchestrator.run(query)
            else:
                return await simple_rag(query)
    """

    def __init__(
        self,
        llm_client,
        retriever,
        planner=None,
        tools: Optional[List[Any]] = None,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            llm_client: LLM client for generation
            retriever: Retriever for document search
            planner: Optional planner for query decomposition
            tools: Optional list of tools for augmented retrieval
            config: Agent configuration
        """
        self.llm_client = llm_client
        self.retriever = retriever
        self.planner = planner
        self.tools = {t.name: t for t in (tools or [])}
        self.config = config or AgentConfig()

        self._state = AgentState.IDLE
        self._steps: List[ExecutionStep] = []
        self._step_counter = 0

    @property
    def state(self) -> AgentState:
        return self._state

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Execute the agentic RAG workflow.

        Args:
            query: User query to answer
            context: Optional additional context

        Returns:
            AgentResult with answer and execution trace
        """
        import time

        start_time = time.time()
        self._steps = []
        self._step_counter = 0
        self._state = AgentState.PLANNING

        sources = []
        answer = ""
        confidence = 0.0

        try:
            # Step 1: Plan the execution
            if self.config.enable_planning and self.planner:
                plan = await self._plan(query, context)
            else:
                plan = await self._default_plan(query)

            self._state = AgentState.EXECUTING

            # Step 2: Execute the plan
            execution_context = {
                "query": query,
                "context": context or {},
                "retrieved_docs": [],
                "intermediate_results": [],
            }

            for step_plan in plan:
                if len(self._steps) >= self.config.max_steps:
                    logger.warning("Max steps reached, stopping execution")
                    break

                step_result = await self._execute_step(step_plan, execution_context)

                if step_result:
                    execution_context["intermediate_results"].append(step_result)

                    # Collect sources
                    if "sources" in step_result:
                        sources.extend(step_result["sources"])

            # Step 3: Synthesize final answer
            self._state = AgentState.REFLECTING if self.config.enable_reflection else AgentState.COMPLETED

            answer, confidence = await self._synthesize(query, execution_context)

            # Step 4: Optional reflection and refinement
            if self.config.enable_reflection and confidence < self.config.confidence_threshold:
                answer, confidence = await self._reflect_and_refine(
                    query, answer, confidence, execution_context
                )

            self._state = AgentState.COMPLETED

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            self._state = AgentState.FAILED
            answer = f"I encountered an error while processing your query: {str(e)}"
            confidence = 0.0

        total_duration = (time.time() - start_time) * 1000

        return AgentResult(
            query=query,
            answer=answer,
            steps=self._steps,
            sources=sources,
            confidence=confidence,
            total_duration_ms=total_duration,
            state=self._state,
            metadata={
                "tools_used": list(set(
                    s.input_data.get("tool")
                    for s in self._steps
                    if s.step_type == StepType.TOOL_CALL
                )),
                "retrieval_count": sum(
                    1 for s in self._steps if s.step_type == StepType.RETRIEVE
                ),
            },
        )

    async def _plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate execution plan using planner."""
        step = self._create_step(
            StepType.REASON,
            "Planning query decomposition",
            {"query": query},
        )

        try:
            plan = await self.planner.plan(query, context, list(self.tools.keys()))
            step.output_data = {"plan": plan}
            step.status = "completed"
            return plan

        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            # Fall back to default plan
            return await self._default_plan(query)

    async def _default_plan(self, query: str) -> List[Dict[str, Any]]:
        """Generate a simple default plan."""
        return [
            {
                "step_type": "retrieve",
                "description": "Search for relevant documents",
                "params": {"query": query},
            },
            {
                "step_type": "synthesize",
                "description": "Generate answer from retrieved context",
                "params": {},
            },
        ]

    async def _execute_step(
        self,
        step_plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Execute a single step in the plan."""
        import time

        step_type_str = step_plan.get("step_type", "reason")
        step_type = StepType(step_type_str) if step_type_str in [e.value for e in StepType] else StepType.REASON

        step = self._create_step(
            step_type,
            step_plan.get("description", f"Execute {step_type_str}"),
            step_plan.get("params", {}),
        )

        start_time = time.time()

        try:
            if step_type == StepType.RETRIEVE:
                result = await self._execute_retrieve(step_plan, context)

            elif step_type == StepType.TOOL_CALL:
                result = await self._execute_tool(step_plan, context)

            elif step_type == StepType.REASON:
                result = await self._execute_reason(step_plan, context)

            elif step_type == StepType.SYNTHESIZE:
                result = await self._execute_synthesize(step_plan, context)

            elif step_type == StepType.REFLECT:
                result = await self._execute_reflect(step_plan, context)

            else:
                result = {"message": f"Unknown step type: {step_type}"}

            step.output_data = result
            step.status = "completed"
            step.duration_ms = (time.time() - start_time) * 1000

            return result

        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            step.duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Step execution failed: {e}")
            return None

    async def _execute_retrieve(
        self,
        step_plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a retrieval step."""
        params = step_plan.get("params", {})
        query = params.get("query", context["query"])
        top_k = params.get("top_k", 5)

        # Use the configured retriever
        results = await self.retriever.retrieve(query, top_k=top_k)

        # Handle different retriever result formats
        if hasattr(results, 'results'):
            docs = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results.results]
            context_str = results.get_combined_context() if hasattr(results, 'get_combined_context') else ""
        else:
            docs = results if isinstance(results, list) else [results]
            context_str = "\n\n".join(d.get("content", str(d)) for d in docs)

        context["retrieved_docs"].extend(docs)

        return {
            "documents": docs,
            "context": context_str,
            "count": len(docs),
            "sources": [{"content": d.get("content", "")[:200], "metadata": d.get("metadata", {})} for d in docs],
        }

    async def _execute_tool(
        self,
        step_plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool call."""
        params = step_plan.get("params", {})
        tool_name = params.get("tool")
        tool_input = params.get("input", {})

        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}

        tool = self.tools[tool_name]
        result = await tool.execute(tool_input, context)

        return {
            "tool": tool_name,
            "input": tool_input,
            "output": result,
        }

    async def _execute_reason(
        self,
        step_plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a reasoning step."""
        params = step_plan.get("params", {})
        prompt = params.get("prompt", "")

        # Build context from previous steps
        prev_results = context.get("intermediate_results", [])
        context_str = "\n".join(
            str(r.get("output", r)) for r in prev_results[-3:]
        )

        full_prompt = f"""Based on the following context, {step_plan.get('description', 'reason about the query')}.

Context:
{context_str}

Query: {context['query']}

{prompt}

Provide your reasoning:"""

        response = await self.llm_client.generate(full_prompt, temperature=0.3)

        return {
            "reasoning": response,
            "prompt": full_prompt[:500],
        }

    async def _execute_synthesize(
        self,
        step_plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesize information from retrieved documents."""
        docs = context.get("retrieved_docs", [])
        query = context["query"]

        # Build context from documents
        doc_context = "\n\n".join(
            f"Document {i+1}:\n{d.get('content', str(d))}"
            for i, d in enumerate(docs[:5])
        )

        prompt = f"""Based on the following documents, answer the query.

Documents:
{doc_context}

Query: {query}

Provide a comprehensive answer based only on the information in the documents. If the documents don't contain enough information, say so."""

        response = await self.llm_client.generate(prompt, temperature=0.3)

        return {
            "synthesis": response,
            "doc_count": len(docs),
        }

    async def _execute_reflect(
        self,
        step_plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a reflection step."""
        params = step_plan.get("params", {})
        current_answer = params.get("answer", "")

        prompt = f"""Reflect on the following answer to the query. Identify any issues, gaps, or areas for improvement.

Query: {context['query']}

Current Answer: {current_answer}

Reflection (identify issues and suggest improvements):"""

        reflection = await self.llm_client.generate(prompt, temperature=0.3)

        return {
            "reflection": reflection,
            "original_answer": current_answer[:500],
        }

    async def _synthesize(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> tuple[str, float]:
        """Synthesize the final answer."""
        step = self._create_step(
            StepType.SYNTHESIZE,
            "Synthesizing final answer",
            {"query": query},
        )

        try:
            # Gather all intermediate results
            results = context.get("intermediate_results", [])
            docs = context.get("retrieved_docs", [])

            # Build comprehensive context
            context_parts = []

            for result in results:
                if "synthesis" in result:
                    context_parts.append(result["synthesis"])
                elif "reasoning" in result:
                    context_parts.append(result["reasoning"])
                elif "output" in result:
                    context_parts.append(str(result["output"]))

            combined_context = "\n\n".join(context_parts) if context_parts else ""

            # If no synthesis yet, do it now
            if not combined_context and docs:
                doc_context = "\n\n".join(
                    f"Document {i+1}:\n{d.get('content', str(d))[:1000]}"
                    for i, d in enumerate(docs[:5])
                )
                combined_context = doc_context

            prompt = f"""Based on the following information, provide a final comprehensive answer to the query.

Information:
{combined_context[:4000]}

Query: {query}

Provide a clear, well-structured answer. Also rate your confidence (0-1) in the answer at the end in the format: [Confidence: X.X]"""

            response = await self.llm_client.generate(prompt, temperature=0.3)

            # Extract confidence from response
            confidence = 0.8
            if "[Confidence:" in response:
                try:
                    conf_str = response.split("[Confidence:")[1].split("]")[0].strip()
                    confidence = float(conf_str)
                    response = response.split("[Confidence:")[0].strip()
                except (ValueError, IndexError):
                    pass

            step.output_data = {"answer": response[:500], "confidence": confidence}
            step.status = "completed"

            return response, confidence

        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            return f"I was unable to synthesize an answer: {str(e)}", 0.0

    async def _reflect_and_refine(
        self,
        query: str,
        answer: str,
        confidence: float,
        context: Dict[str, Any],
    ) -> tuple[str, float]:
        """Reflect on and refine the answer."""
        step = self._create_step(
            StepType.REFLECT,
            "Reflecting and refining answer",
            {"answer": answer[:500], "confidence": confidence},
        )

        try:
            # Reflection prompt
            prompt = f"""Review the following answer and improve it if needed.

Query: {query}

Current Answer: {answer}

Current Confidence: {confidence}

If the answer is incomplete, unclear, or could be improved, provide an enhanced version. If it's already good, return it as-is.

Improved Answer:"""

            improved_answer = await self.llm_client.generate(prompt, temperature=0.3)

            # Slight confidence boost for refined answers
            new_confidence = min(confidence + 0.1, 0.95)

            step.output_data = {
                "original_answer": answer[:300],
                "improved_answer": improved_answer[:300],
                "new_confidence": new_confidence,
            }
            step.status = "completed"

            return improved_answer, new_confidence

        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            return answer, confidence

    def _create_step(
        self,
        step_type: StepType,
        description: str,
        input_data: Dict[str, Any],
    ) -> ExecutionStep:
        """Create and register a new execution step."""
        self._step_counter += 1
        step = ExecutionStep(
            step_id=f"step_{self._step_counter}",
            step_type=step_type,
            description=description,
            input_data=input_data,
        )
        self._steps.append(step)
        return step

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get the full execution trace."""
        return [s.to_dict() for s in self._steps]
