"""
Tools for Agentic RAG

Provides tool interfaces for the agentic RAG orchestrator,
enabling augmented retrieval with external capabilities.

Supports:
- Web search integration
- Calculator operations
- Code execution
- Custom tool registration
"""

import logging
import re
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


class Tool(ABC):
    """Base class for agentic RAG tools."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a tool.

        Args:
            name: Unique tool name
            description: Human-readable description
            parameters: JSON schema for tool parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    @abstractmethod
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Execute the tool.

        Args:
            input_data: Tool input parameters
            context: Optional execution context

        Returns:
            ToolResult with output or error
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for LLM tool calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class CalculatorTool(Tool):
    """
    Calculator tool for mathematical operations.

    Safely evaluates mathematical expressions.

    Example:
        calc = CalculatorTool()
        result = await calc.execute({"expression": "2 + 3 * 4"})
        print(result.output)  # 14
    """

    # Safe operations whitelist
    ALLOWED_NAMES = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'len': len, 'pow': pow,
    }

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluates mathematical expressions. Supports basic arithmetic (+, -, *, /, **) and functions (abs, round, min, max, sum, pow).",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    },
                },
                "required": ["expression"],
            },
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        expression = input_data.get("expression", "")

        try:
            # Sanitize expression
            sanitized = self._sanitize_expression(expression)

            # Evaluate safely
            result = eval(sanitized, {"__builtins__": {}}, self.ALLOWED_NAMES)

            return ToolResult(
                success=True,
                output=result,
                metadata={"expression": expression, "sanitized": sanitized},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Calculator error: {str(e)}",
                metadata={"expression": expression},
            )

    def _sanitize_expression(self, expression: str) -> str:
        """Sanitize expression for safe evaluation."""
        # Remove any non-math characters
        allowed_chars = set('0123456789+-*/().,%^ ')
        allowed_words = {'abs', 'round', 'min', 'max', 'sum', 'len', 'pow'}

        # Replace common notation
        expression = expression.replace('^', '**')
        expression = expression.replace('×', '*')
        expression = expression.replace('÷', '/')

        # Validate
        tokens = re.findall(r'[a-zA-Z]+|\S', expression)
        for token in tokens:
            if token.isalpha() and token.lower() not in allowed_words:
                raise ValueError(f"Disallowed term: {token}")

        return expression


class WebSearchTool(Tool):
    """
    Web search tool for retrieving current information.

    Integrates with search APIs for real-time data.

    Example:
        search = WebSearchTool(api_key="your_key")
        result = await search.execute({"query": "latest Python release"})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_engine: str = "duckduckgo",
        max_results: int = 5,
    ):
        super().__init__(
            name="web_search",
            description="Searches the web for current information. Use for recent events, updates, or data not in the knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
        self.api_key = api_key
        self.search_engine = search_engine
        self.max_results = max_results

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        query = input_data.get("query", "")
        num_results = min(input_data.get("num_results", self.max_results), self.max_results)

        try:
            if self.search_engine == "duckduckgo":
                results = await self._duckduckgo_search(query, num_results)
            else:
                results = await self._mock_search(query, num_results)

            return ToolResult(
                success=True,
                output=results,
                metadata={
                    "query": query,
                    "num_results": len(results),
                    "search_engine": self.search_engine,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Search error: {str(e)}",
                metadata={"query": query},
            )

    async def _duckduckgo_search(
        self,
        query: str,
        num_results: int,
    ) -> List[Dict[str, str]]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    })

            return results

        except ImportError:
            logger.warning("duckduckgo_search not installed, using mock search")
            return await self._mock_search(query, num_results)

    async def _mock_search(
        self,
        query: str,
        num_results: int,
    ) -> List[Dict[str, str]]:
        """Mock search for testing."""
        return [
            {
                "title": f"Search result {i+1} for: {query}",
                "snippet": f"This is a mock search result for the query '{query}'. Install duckduckgo_search for real results.",
                "url": f"https://example.com/result{i+1}",
            }
            for i in range(num_results)
        ]


class CodeExecutorTool(Tool):
    """
    Code execution tool for running Python code snippets.

    Safely executes code in a sandboxed environment.

    Example:
        executor = CodeExecutorTool()
        result = await executor.execute({
            "code": "print('Hello')",
            "language": "python"
        })
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        allowed_imports: Optional[List[str]] = None,
    ):
        super().__init__(
            name="code_executor",
            description="Executes Python code snippets. Use for computations, data processing, or testing code.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (currently only 'python' supported)",
                        "default": "python",
                    },
                },
                "required": ["code"],
            },
        )
        self.timeout_seconds = timeout_seconds
        self.allowed_imports = allowed_imports or ['math', 'json', 'datetime', 're', 'collections']

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        code = input_data.get("code", "")
        language = input_data.get("language", "python")

        if language != "python":
            return ToolResult(
                success=False,
                output=None,
                error=f"Unsupported language: {language}",
            )

        try:
            # Validate code safety
            self._validate_code(code)

            # Execute in restricted environment
            output = await self._execute_python(code)

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "language": language,
                    "code_length": len(code),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}",
                metadata={"code": code[:200]},
            )

    def _validate_code(self, code: str):
        """Validate code for safety."""
        # Disallowed patterns
        dangerous_patterns = [
            r'\bimport\s+os\b',
            r'\bimport\s+subprocess\b',
            r'\bimport\s+sys\b',
            r'\bopen\s*\(',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\b',
            r'\bcompile\s*\(',
            r'\bglobals\s*\(',
            r'\blocals\s*\(',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Disallowed code pattern detected")

        # Check imports
        import_pattern = r'import\s+(\w+)'
        imports = re.findall(import_pattern, code)
        for imp in imports:
            if imp not in self.allowed_imports:
                raise ValueError(f"Import not allowed: {imp}")

    async def _execute_python(self, code: str) -> str:
        """Execute Python code safely."""
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Restricted globals
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'isinstance': isinstance,
                'type': type,
            }
        }

        # Allow safe imports
        for module_name in self.allowed_imports:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals)

            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()

            if errors:
                return f"Output:\n{output}\n\nErrors:\n{errors}"

            return output if output else "Code executed successfully (no output)"

        except Exception as e:
            return f"Error: {str(e)}"


class DateTimeTool(Tool):
    """
    Date and time tool for temporal queries.

    Example:
        dt_tool = DateTimeTool()
        result = await dt_tool.execute({"operation": "now"})
    """

    def __init__(self):
        super().__init__(
            name="datetime",
            description="Gets current date/time or performs date calculations.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["now", "today", "format", "diff"],
                        "description": "Operation to perform",
                    },
                    "format": {
                        "type": "string",
                        "description": "Date format string (for 'format' operation)",
                    },
                    "date1": {
                        "type": "string",
                        "description": "First date (for 'diff' operation)",
                    },
                    "date2": {
                        "type": "string",
                        "description": "Second date (for 'diff' operation)",
                    },
                },
                "required": ["operation"],
            },
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        operation = input_data.get("operation", "now")

        try:
            if operation == "now":
                result = datetime.now().isoformat()

            elif operation == "today":
                result = datetime.now().strftime("%Y-%m-%d")

            elif operation == "format":
                fmt = input_data.get("format", "%Y-%m-%d %H:%M:%S")
                result = datetime.now().strftime(fmt)

            elif operation == "diff":
                from datetime import datetime as dt
                date1_str = input_data.get("date1", "")
                date2_str = input_data.get("date2", "")

                date1 = dt.fromisoformat(date1_str)
                date2 = dt.fromisoformat(date2_str)
                diff = (date2 - date1).days
                result = f"{diff} days"

            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}",
                )

            return ToolResult(
                success=True,
                output=result,
                metadata={"operation": operation},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"DateTime error: {str(e)}",
            )


class ToolRegistry:
    """
    Registry for managing available tools.

    Example:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WebSearchTool())

        tools = registry.get_tools()
        tool = registry.get_tool("calculator")
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str):
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tool_names(self) -> List[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get schema for all tools (for LLM tool calling)."""
        return [tool.to_dict() for tool in self._tools.values()]


def create_default_tools() -> List[Tool]:
    """Create the default set of tools."""
    return [
        CalculatorTool(),
        WebSearchTool(),
        CodeExecutorTool(),
        DateTimeTool(),
    ]
