"""
Tests for Agentic RAG Components (PR #5)

Tests orchestrator, planner, and tools.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import components
from openrag.components.agents import (
    AgentState,
    StepType,
    ExecutionStep,
    AgentResult,
    AgentConfig,
    AgenticRAGOrchestrator,
    QueryComplexity,
    QueryAnalysis,
    QueryPlanner,
    Tool,
    ToolResult,
    CalculatorTool,
    WebSearchTool,
    CodeExecutorTool,
    DateTimeTool,
    ToolRegistry,
    create_default_tools,
)


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()

        assert config.max_steps == 10
        assert config.max_retries == 2
        assert config.enable_reflection is True
        assert config.enable_planning is True
        assert config.confidence_threshold == 0.7

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            max_steps=5,
            enable_reflection=False,
            confidence_threshold=0.9,
        )

        assert config.max_steps == 5
        assert config.enable_reflection is False
        assert config.confidence_threshold == 0.9


class TestExecutionStep:
    """Tests for ExecutionStep."""

    def test_step_to_dict(self):
        """Test step serialization."""
        step = ExecutionStep(
            step_id="step_1",
            step_type=StepType.RETRIEVE,
            description="Search for documents",
            input_data={"query": "test"},
            output_data={"documents": []},
            status="completed",
        )

        d = step.to_dict()

        assert d["step_id"] == "step_1"
        assert d["step_type"] == "retrieve"
        assert d["description"] == "Search for documents"
        assert d["status"] == "completed"


class TestAgentResult:
    """Tests for AgentResult."""

    def test_result_to_dict(self):
        """Test result serialization."""
        result = AgentResult(
            query="What is Python?",
            answer="Python is a programming language.",
            steps=[],
            sources=[],
            confidence=0.9,
            total_duration_ms=100.0,
            state=AgentState.COMPLETED,
        )

        d = result.to_dict()

        assert d["query"] == "What is Python?"
        assert d["answer"] == "Python is a programming language."
        assert d["confidence"] == 0.9
        assert d["state"] == "completed"


class TestQueryPlanner:
    """Tests for QueryPlanner."""

    @pytest.fixture
    def planner(self):
        """Create planner without LLM."""
        return QueryPlanner(llm_client=None, use_llm_planning=False)

    def test_detect_simple_complexity(self, planner):
        """Test simple query detection."""
        complexity = planner._detect_complexity("what is python?")
        assert complexity == QueryComplexity.SIMPLE

    def test_detect_comparison_complexity(self, planner):
        """Test comparison query detection."""
        complexity = planner._detect_complexity("compare python vs java")
        assert complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]

    def test_detect_multihop_complexity(self, planner):
        """Test multi-hop query detection."""
        complexity = planner._detect_complexity("why does python then affect performance?")
        assert complexity == QueryComplexity.MULTI_HOP

    def test_detect_intent_comparison(self, planner):
        """Test comparison intent detection."""
        intent = planner._detect_intent("compare python and java")
        assert intent == "comparison"

    def test_detect_intent_instructional(self, planner):
        """Test instructional intent detection."""
        intent = planner._detect_intent("how to install python")
        assert intent == "instructional"

    def test_detect_intent_definitional(self, planner):
        """Test definitional intent detection."""
        intent = planner._detect_intent("what is machine learning")
        assert intent == "definitional"

    def test_extract_entities(self, planner):
        """Test entity extraction from query."""
        entities = planner._extract_entities("Python and Java are programming languages")
        assert "Python" in entities
        assert "Java" in entities

    def test_detect_required_tools(self, planner):
        """Test tool requirement detection."""
        available_tools = ["calculator", "web_search"]

        tools = planner._detect_required_tools(
            "calculate 2 + 2",
            available_tools,
        )
        assert "calculator" in tools

        tools = planner._detect_required_tools(
            "latest python news",
            available_tools,
        )
        assert "web_search" in tools

    @pytest.mark.asyncio
    async def test_simple_plan(self, planner):
        """Test simple plan generation."""
        plan = await planner.plan("what is python?")

        assert len(plan) >= 2
        assert plan[0]["step_type"] == "retrieve"
        assert plan[-1]["step_type"] == "synthesize"

    @pytest.mark.asyncio
    async def test_analyze_query(self, planner):
        """Test query analysis."""
        analysis = await planner.analyze_query(
            "compare python and javascript for web development",
            available_tools=["web_search"],
        )

        assert analysis.complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
        assert analysis.intent == "comparison"
        assert "Python" in analysis.entities or "JavaScript" in analysis.entities


class TestCalculatorTool:
    """Tests for CalculatorTool."""

    @pytest.fixture
    def calculator(self):
        return CalculatorTool()

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self, calculator):
        """Test basic arithmetic operations."""
        result = await calculator.execute({"expression": "2 + 3"})
        assert result.success
        assert result.output == 5

        result = await calculator.execute({"expression": "10 * 5"})
        assert result.success
        assert result.output == 50

    @pytest.mark.asyncio
    async def test_complex_expression(self, calculator):
        """Test complex expressions."""
        result = await calculator.execute({"expression": "(10 + 5) * 2"})
        assert result.success
        assert result.output == 30

    @pytest.mark.asyncio
    async def test_power_operation(self, calculator):
        """Test power operation."""
        result = await calculator.execute({"expression": "2 ** 8"})
        assert result.success
        assert result.output == 256

        # Also test ^ notation
        result = await calculator.execute({"expression": "2^8"})
        assert result.success
        assert result.output == 256

    @pytest.mark.asyncio
    async def test_builtin_functions(self, calculator):
        """Test allowed builtin functions."""
        result = await calculator.execute({"expression": "abs(-5)"})
        assert result.success
        assert result.output == 5

        result = await calculator.execute({"expression": "max(1, 5, 3)"})
        assert result.success
        assert result.output == 5

    @pytest.mark.asyncio
    async def test_invalid_expression(self, calculator):
        """Test handling of invalid expressions."""
        result = await calculator.execute({"expression": "import os"})
        assert not result.success
        assert result.error is not None


class TestCodeExecutorTool:
    """Tests for CodeExecutorTool."""

    @pytest.fixture
    def executor(self):
        return CodeExecutorTool()

    @pytest.mark.asyncio
    async def test_simple_code(self, executor):
        """Test simple code execution."""
        result = await executor.execute({"code": "print('Hello')"})
        assert result.success
        assert "Hello" in result.output

    @pytest.mark.asyncio
    async def test_calculation(self, executor):
        """Test calculation code."""
        result = await executor.execute({"code": "print(sum([1, 2, 3, 4, 5]))"})
        assert result.success
        assert "15" in result.output

    @pytest.mark.asyncio
    async def test_dangerous_code_blocked(self, executor):
        """Test that dangerous code is blocked."""
        result = await executor.execute({"code": "import os"})
        assert not result.success
        assert "error" in result.error.lower()

        result = await executor.execute({"code": "open('file.txt')"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_unsupported_language(self, executor):
        """Test unsupported language handling."""
        result = await executor.execute({"code": "console.log('hi')", "language": "javascript"})
        assert not result.success
        assert "Unsupported" in result.error


class TestDateTimeTool:
    """Tests for DateTimeTool."""

    @pytest.fixture
    def dt_tool(self):
        return DateTimeTool()

    @pytest.mark.asyncio
    async def test_now_operation(self, dt_tool):
        """Test 'now' operation."""
        result = await dt_tool.execute({"operation": "now"})
        assert result.success
        assert "T" in result.output  # ISO format has T separator

    @pytest.mark.asyncio
    async def test_today_operation(self, dt_tool):
        """Test 'today' operation."""
        result = await dt_tool.execute({"operation": "today"})
        assert result.success
        assert "-" in result.output  # Date format YYYY-MM-DD

    @pytest.mark.asyncio
    async def test_format_operation(self, dt_tool):
        """Test 'format' operation."""
        result = await dt_tool.execute({
            "operation": "format",
            "format": "%Y"
        })
        assert result.success
        assert len(result.output) == 4  # Year only


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    @pytest.fixture
    def search_tool(self):
        return WebSearchTool()

    @pytest.mark.asyncio
    async def test_mock_search(self, search_tool):
        """Test mock search functionality."""
        result = await search_tool.execute({"query": "test query"})
        assert result.success
        assert isinstance(result.output, list)
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_num_results(self, search_tool):
        """Test limiting number of results."""
        result = await search_tool.execute({
            "query": "test",
            "num_results": 2
        })
        assert result.success
        assert len(result.output) == 2


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()
        tool = CalculatorTool()

        registry.register(tool)

        assert "calculator" in registry.get_tool_names()
        assert registry.get_tool("calculator") is tool

    def test_unregister_tool(self):
        """Test tool unregistration."""
        registry = ToolRegistry()
        tool = CalculatorTool()

        registry.register(tool)
        registry.unregister("calculator")

        assert "calculator" not in registry.get_tool_names()

    def test_get_tools_schema(self):
        """Test getting tools schema."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(DateTimeTool())

        schema = registry.get_tools_schema()

        assert len(schema) == 2
        assert all("name" in s for s in schema)
        assert all("description" in s for s in schema)


class TestCreateDefaultTools:
    """Tests for create_default_tools."""

    def test_create_default_tools(self):
        """Test default tools creation."""
        tools = create_default_tools()

        assert len(tools) == 4
        tool_names = [t.name for t in tools]

        assert "calculator" in tool_names
        assert "web_search" in tool_names
        assert "code_executor" in tool_names
        assert "datetime" in tool_names


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(
            success=True,
            output=42,
            metadata={"key": "value"},
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["output"] == 42
        assert d["error"] is None

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(
            success=False,
            output=None,
            error="Something went wrong",
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["error"] == "Something went wrong"


# Integration tests with mocked LLM

class TestAgenticRAGOrchestratorIntegration:
    """Integration tests for AgenticRAGOrchestrator."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = AsyncMock()
        mock.generate.return_value = "This is a test answer. [Confidence: 0.85]"
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever."""
        mock = AsyncMock()
        mock.retrieve.return_value = MagicMock(
            results=[
                MagicMock(
                    to_dict=lambda: {"content": "Test document", "metadata": {}},
                    content="Test document",
                )
            ],
            get_combined_context=lambda: "Test context",
        )
        return mock

    @pytest.mark.asyncio
    async def test_simple_run(self, mock_llm, mock_retriever):
        """Test simple orchestrator run."""
        orchestrator = AgenticRAGOrchestrator(
            llm_client=mock_llm,
            retriever=mock_retriever,
            config=AgentConfig(enable_planning=False, enable_reflection=False),
        )

        result = await orchestrator.run("What is Python?")

        assert result.state == AgentState.COMPLETED
        assert len(result.answer) > 0
        assert len(result.steps) > 0

    @pytest.mark.asyncio
    async def test_with_tools(self, mock_llm, mock_retriever):
        """Test orchestrator with tools."""
        tools = [CalculatorTool(), DateTimeTool()]

        orchestrator = AgenticRAGOrchestrator(
            llm_client=mock_llm,
            retriever=mock_retriever,
            tools=tools,
            config=AgentConfig(enable_planning=False, enable_reflection=False),
        )

        assert "calculator" in orchestrator.tools
        assert "datetime" in orchestrator.tools

    def test_get_execution_trace(self, mock_llm, mock_retriever):
        """Test execution trace retrieval."""
        orchestrator = AgenticRAGOrchestrator(
            llm_client=mock_llm,
            retriever=mock_retriever,
        )

        # Before any execution
        trace = orchestrator.get_execution_trace()
        assert isinstance(trace, list)


# Run with: pytest tests/test_agentic_rag.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
