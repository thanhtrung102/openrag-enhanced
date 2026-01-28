"""
Query Planner for Agentic RAG

Decomposes complex queries into execution plans
with retrieval, reasoning, and tool-calling steps.

Supports:
- Query complexity analysis
- Step-by-step decomposition
- Tool selection and routing
- Dynamic replanning
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"          # Single retrieval + answer
    MODERATE = "moderate"      # Multiple retrievals or comparisons
    COMPLEX = "complex"        # Multi-step reasoning, tools needed
    MULTI_HOP = "multi_hop"    # Chain of reasoning required


@dataclass
class QueryAnalysis:
    """Analysis of a user query."""
    query: str
    complexity: QueryComplexity
    sub_questions: List[str]
    required_tools: List[str]
    entities: List[str]
    intent: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "complexity": self.complexity.value,
            "sub_questions": self.sub_questions,
            "required_tools": self.required_tools,
            "entities": self.entities,
            "intent": self.intent,
        }


@dataclass
class PlanStep:
    """A single step in the execution plan."""
    step_type: str
    description: str
    params: Dict[str, Any]
    depends_on: List[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_type": self.step_type,
            "description": self.description,
            "params": self.params,
            "depends_on": self.depends_on or [],
        }


class QueryPlanner:
    """
    Plans execution strategies for complex queries.

    Analyzes queries and generates step-by-step plans
    that the orchestrator can execute.

    Example:
        planner = QueryPlanner(llm_client=llm)

        analysis = await planner.analyze_query(
            "Compare Python and Rust performance for web servers"
        )

        plan = await planner.plan(
            query="Compare Python and Rust performance for web servers",
            available_tools=["web_search", "code_executor"],
        )

        for step in plan:
            print(f"{step['step_type']}: {step['description']}")

    Integration with OpenRAG:
        # In the agentic pipeline
        planner = QueryPlanner(llm_client=openai_client)
        orchestrator = AgenticRAGOrchestrator(
            llm_client=openai_client,
            retriever=retriever,
            planner=planner,
        )

        # Planner is used automatically by orchestrator
        result = await orchestrator.run(query)
    """

    # Patterns for query complexity detection
    COMPARISON_PATTERNS = [
        r'\bcompare\b', r'\bversus\b', r'\bvs\.?\b', r'\bdifference\b',
        r'\bsimilar\b', r'\bbetter\b', r'\bworse\b', r'\badvantages?\b',
    ]

    MULTI_HOP_PATTERNS = [
        r'\bwhy\b.*\bthen\b', r'\bhow\b.*\baffect\b', r'\bwhat\b.*\bcaused\b',
        r'\bif\b.*\bthen\b', r'\brelationship\b.*\bbetween\b',
    ]

    TOOL_PATTERNS = {
        "calculator": [r'\bcalculate\b', r'\bcompute\b', r'\bsum\b', r'\baverage\b', r'\d+\s*[\+\-\*/]\s*\d+'],
        "web_search": [r'\blatest\b', r'\bcurrent\b', r'\brecent\b', r'\btoday\b', r'\bnews\b'],
        "code_executor": [r'\brun\b.*\bcode\b', r'\bexecute\b', r'\btest\b.*\bfunction\b'],
    }

    def __init__(
        self,
        llm_client=None,
        use_llm_planning: bool = True,
        max_sub_questions: int = 5,
    ):
        """
        Initialize the planner.

        Args:
            llm_client: LLM client for complex planning
            use_llm_planning: Whether to use LLM for planning
            max_sub_questions: Maximum sub-questions to generate
        """
        self.llm_client = llm_client
        self.use_llm_planning = use_llm_planning and llm_client is not None
        self.max_sub_questions = max_sub_questions

    async def analyze_query(
        self,
        query: str,
        available_tools: Optional[List[str]] = None,
    ) -> QueryAnalysis:
        """
        Analyze a query to determine complexity and requirements.

        Args:
            query: User query to analyze
            available_tools: List of available tool names

        Returns:
            QueryAnalysis with complexity and requirements
        """
        query_lower = query.lower()

        # Detect complexity
        complexity = self._detect_complexity(query_lower)

        # Detect required tools
        required_tools = self._detect_required_tools(query_lower, available_tools)

        # Extract entities (simple extraction)
        entities = self._extract_entities(query)

        # Determine intent
        intent = self._detect_intent(query_lower)

        # Generate sub-questions if complex
        sub_questions = []
        if complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP]:
            sub_questions = await self._generate_sub_questions(query)

        return QueryAnalysis(
            query=query,
            complexity=complexity,
            sub_questions=sub_questions,
            required_tools=required_tools,
            entities=entities,
            intent=intent,
        )

    async def plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate an execution plan for the query.

        Args:
            query: User query
            context: Optional context information
            available_tools: Available tools for planning

        Returns:
            List of plan steps
        """
        # Analyze the query
        analysis = await self.analyze_query(query, available_tools)

        # Generate plan based on complexity
        if analysis.complexity == QueryComplexity.SIMPLE:
            plan = self._simple_plan(query)

        elif analysis.complexity == QueryComplexity.MODERATE:
            plan = self._moderate_plan(query, analysis)

        elif analysis.complexity == QueryComplexity.COMPLEX:
            if self.use_llm_planning:
                plan = await self._llm_plan(query, analysis, available_tools)
            else:
                plan = self._complex_plan(query, analysis)

        elif analysis.complexity == QueryComplexity.MULTI_HOP:
            if self.use_llm_planning:
                plan = await self._llm_plan(query, analysis, available_tools)
            else:
                plan = self._multi_hop_plan(query, analysis)

        else:
            plan = self._simple_plan(query)

        logger.info(f"Generated {len(plan)} step plan for query (complexity: {analysis.complexity.value})")

        return plan

    def _detect_complexity(self, query_lower: str) -> QueryComplexity:
        """Detect query complexity from patterns."""
        # Check for multi-hop patterns
        for pattern in self.MULTI_HOP_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryComplexity.MULTI_HOP

        # Check for comparison patterns
        comparison_count = sum(
            1 for p in self.COMPARISON_PATTERNS
            if re.search(p, query_lower)
        )
        if comparison_count >= 2:
            return QueryComplexity.COMPLEX
        elif comparison_count >= 1:
            return QueryComplexity.MODERATE

        # Check for multiple question marks or conjunctions
        if query_lower.count('?') > 1 or ' and ' in query_lower:
            return QueryComplexity.MODERATE

        # Check query length
        word_count = len(query_lower.split())
        if word_count > 20:
            return QueryComplexity.MODERATE

        return QueryComplexity.SIMPLE

    def _detect_required_tools(
        self,
        query_lower: str,
        available_tools: Optional[List[str]],
    ) -> List[str]:
        """Detect which tools might be needed."""
        if not available_tools:
            return []

        required = []
        for tool_name, patterns in self.TOOL_PATTERNS.items():
            if tool_name in available_tools:
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        required.append(tool_name)
                        break

        return required

    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query."""
        # Simple capitalized word extraction
        words = query.split()
        entities = []

        for word in words:
            # Skip common words and short words
            clean_word = word.strip('.,?!')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                if clean_word.lower() not in ['the', 'what', 'how', 'why', 'when', 'where', 'which', 'who']:
                    entities.append(clean_word)

        return list(set(entities))

    def _detect_intent(self, query_lower: str) -> str:
        """Detect the primary intent of the query."""
        if any(p in query_lower for p in ['compare', 'versus', 'vs', 'difference']):
            return "comparison"
        elif any(p in query_lower for p in ['how to', 'how do', 'steps to']):
            return "instructional"
        elif any(p in query_lower for p in ['what is', 'define', 'explain']):
            return "definitional"
        elif any(p in query_lower for p in ['why', 'reason', 'cause']):
            return "explanatory"
        elif any(p in query_lower for p in ['list', 'examples', 'types']):
            return "enumerative"
        else:
            return "informational"

    async def _generate_sub_questions(self, query: str) -> List[str]:
        """Generate sub-questions for complex queries."""
        if not self.use_llm_planning:
            return self._rule_based_sub_questions(query)

        prompt = f"""Break down the following complex query into simpler sub-questions that can be answered independently.

Query: {query}

Generate up to {self.max_sub_questions} sub-questions. Format as a numbered list.

Sub-questions:"""

        try:
            response = await self.llm_client.generate(prompt, temperature=0.3)

            # Parse sub-questions from response
            sub_questions = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line and '?' in line:
                    sub_questions.append(line)

            return sub_questions[:self.max_sub_questions]

        except Exception as e:
            logger.error(f"Failed to generate sub-questions: {e}")
            return self._rule_based_sub_questions(query)

    def _rule_based_sub_questions(self, query: str) -> List[str]:
        """Generate sub-questions using rules."""
        sub_questions = []

        # Check for comparisons
        if 'and' in query.lower() or 'vs' in query.lower():
            # Extract items being compared
            parts = re.split(r'\band\b|\bvs\.?\b|\bversus\b', query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                for part in parts[:2]:
                    part = part.strip()
                    if part:
                        sub_questions.append(f"What is {part}?")

        # Add a synthesis question
        if sub_questions:
            sub_questions.append(f"How do these compare?")

        return sub_questions

    def _simple_plan(self, query: str) -> List[Dict[str, Any]]:
        """Generate a simple retrieval + synthesis plan."""
        return [
            {
                "step_type": "retrieve",
                "description": "Search for relevant documents",
                "params": {"query": query, "top_k": 5},
            },
            {
                "step_type": "synthesize",
                "description": "Generate answer from retrieved context",
                "params": {},
            },
        ]

    def _moderate_plan(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> List[Dict[str, Any]]:
        """Generate a moderate complexity plan."""
        plan = []

        # If there are sub-questions, retrieve for each
        if analysis.sub_questions:
            for i, sub_q in enumerate(analysis.sub_questions[:3]):
                plan.append({
                    "step_type": "retrieve",
                    "description": f"Search for: {sub_q[:50]}",
                    "params": {"query": sub_q, "top_k": 3},
                })
        else:
            plan.append({
                "step_type": "retrieve",
                "description": "Search for relevant documents",
                "params": {"query": query, "top_k": 5},
            })

        # Add tool calls if needed
        for tool in analysis.required_tools[:2]:
            plan.append({
                "step_type": "tool_call",
                "description": f"Use {tool} tool",
                "params": {"tool": tool, "input": {"query": query}},
            })

        # Reasoning step for comparisons
        if analysis.intent == "comparison":
            plan.append({
                "step_type": "reason",
                "description": "Compare and contrast findings",
                "params": {"prompt": "Compare the key differences and similarities."},
            })

        # Synthesize
        plan.append({
            "step_type": "synthesize",
            "description": "Generate comprehensive answer",
            "params": {},
        })

        return plan

    def _complex_plan(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> List[Dict[str, Any]]:
        """Generate a complex plan with reasoning steps."""
        plan = []

        # Initial retrieval
        plan.append({
            "step_type": "retrieve",
            "description": "Initial broad search",
            "params": {"query": query, "top_k": 5},
        })

        # Sub-question retrievals
        for sub_q in analysis.sub_questions[:self.max_sub_questions]:
            plan.append({
                "step_type": "retrieve",
                "description": f"Search for: {sub_q[:40]}...",
                "params": {"query": sub_q, "top_k": 3},
            })

        # Tool calls
        for tool in analysis.required_tools:
            plan.append({
                "step_type": "tool_call",
                "description": f"Execute {tool}",
                "params": {"tool": tool, "input": {"query": query}},
            })

        # Reasoning
        plan.append({
            "step_type": "reason",
            "description": "Analyze and connect findings",
            "params": {"prompt": "Analyze how the findings relate to each other."},
        })

        # Synthesize
        plan.append({
            "step_type": "synthesize",
            "description": "Combine all findings into answer",
            "params": {},
        })

        # Reflect
        plan.append({
            "step_type": "reflect",
            "description": "Verify answer completeness",
            "params": {},
        })

        return plan

    def _multi_hop_plan(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> List[Dict[str, Any]]:
        """Generate a multi-hop reasoning plan."""
        plan = []

        # Chain of retrievals
        for i, sub_q in enumerate(analysis.sub_questions):
            plan.append({
                "step_type": "retrieve",
                "description": f"Hop {i+1}: {sub_q[:40]}...",
                "params": {"query": sub_q, "top_k": 3},
            })

            # Intermediate reasoning after each hop
            plan.append({
                "step_type": "reason",
                "description": f"Process hop {i+1} findings",
                "params": {"prompt": f"What did we learn from this step?"},
            })

        # Final synthesis
        plan.append({
            "step_type": "synthesize",
            "description": "Chain reasoning into final answer",
            "params": {},
        })

        return plan

    async def _llm_plan(
        self,
        query: str,
        analysis: QueryAnalysis,
        available_tools: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Generate a plan using LLM."""
        tools_str = ", ".join(available_tools) if available_tools else "none"

        prompt = f"""Create an execution plan to answer the following query.

Query: {query}

Available tools: {tools_str}

Query analysis:
- Complexity: {analysis.complexity.value}
- Intent: {analysis.intent}
- Sub-questions: {analysis.sub_questions}

Generate a step-by-step plan using these step types:
- retrieve: Search for documents (params: query, top_k)
- tool_call: Use a tool (params: tool, input)
- reason: Analyze information (params: prompt)
- synthesize: Generate answer (params: none)
- reflect: Verify answer (params: none)

Format each step as:
STEP: <step_type>
DESCRIPTION: <what this step does>
PARAMS: <JSON params>

Plan:"""

        try:
            response = await self.llm_client.generate(prompt, temperature=0.3)

            # Parse the plan
            plan = self._parse_llm_plan(response)

            if plan:
                return plan
            else:
                # Fall back to rule-based planning
                return self._complex_plan(query, analysis)

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return self._complex_plan(query, analysis)

    def _parse_llm_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM-generated plan."""
        import json

        plan = []
        current_step = {}

        for line in response.split('\n'):
            line = line.strip()

            if line.startswith('STEP:'):
                if current_step:
                    plan.append(current_step)
                step_type = line.replace('STEP:', '').strip().lower()
                current_step = {"step_type": step_type, "params": {}}

            elif line.startswith('DESCRIPTION:'):
                current_step["description"] = line.replace('DESCRIPTION:', '').strip()

            elif line.startswith('PARAMS:'):
                params_str = line.replace('PARAMS:', '').strip()
                try:
                    current_step["params"] = json.loads(params_str)
                except json.JSONDecodeError:
                    current_step["params"] = {}

        if current_step:
            plan.append(current_step)

        return plan

    async def replan(
        self,
        original_query: str,
        original_plan: List[Dict[str, Any]],
        failed_step: int,
        error: str,
        available_tools: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a new plan after a step failure.

        Args:
            original_query: The original query
            original_plan: The plan that was being executed
            failed_step: Index of the failed step
            error: Error message from the failure
            available_tools: Available tools

        Returns:
            New execution plan
        """
        if not self.use_llm_planning:
            # Simple fallback: skip failed step and continue
            return original_plan[failed_step + 1:]

        prompt = f"""A step in the execution plan failed. Create a new plan to complete the task.

Query: {original_query}

Original plan step that failed:
{original_plan[failed_step]}

Error: {error}

Remaining original steps:
{original_plan[failed_step + 1:]}

Create a revised plan that works around this failure. Use the same step format."""

        try:
            response = await self.llm_client.generate(prompt, temperature=0.3)
            new_plan = self._parse_llm_plan(response)

            if new_plan:
                return new_plan

        except Exception as e:
            logger.error(f"Replanning failed: {e}")

        # Fall back to remaining steps
        return original_plan[failed_step + 1:]
