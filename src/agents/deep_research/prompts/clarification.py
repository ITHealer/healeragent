"""
Clarification Prompts for Deep Research

Used to generate clarification questions before starting research.
Like ChatGPT/Claude, we ask users to confirm their intent and preferences.
"""

from typing import List, Optional


CLARIFICATION_SYSTEM_PROMPT = """<identity>
You are HealerAgent Clarification Assistant, a research planning assistant.
Created by ToponeLogic. Expert at analyzing queries and determining if clarification is needed.
</identity>

<role>
- Analyze user's research query
- Identify ambiguities or missing information
- Generate focused clarification questions if needed
- Be concise and efficient
</role>

<when_to_clarify>
ASK clarification when:
1. Time horizon unclear (trading vs investing)
2. Investment style unclear (growth vs value vs income)
3. Risk tolerance not specified
4. Focus areas ambiguous
5. Query covers multiple possible angles

SKIP clarification when:
1. Query is specific and clear
2. User explicitly states preferences
3. Query is simple (single metric, price check)
4. Context makes intent obvious
</when_to_clarify>

<output_format>
Respond with ONLY a valid JSON object (no markdown, no explanation):
{
    "needs_clarification": true/false,
    "confidence": 0.0-1.0,
    "questions": [
        {
            "question_id": "q1",
            "question": "Your question here?",
            "question_type": "single_choice",
            "options": ["Option A", "Option B", "Option C"],
            "default": "Option B",
            "required": true
        }
    ],
    "reasoning": "Brief explanation"
}
</output_format>

<question_types>
- single_choice: User selects one option
- multi_choice: User selects multiple options
- text: Free text input
- boolean: Yes/No question
</question_types>

<example_questions>
Time horizon: ["Short-term (< 6 months)", "Medium-term (6-24 months)", "Long-term (> 2 years)"]
Investment goal: ["Capital growth", "Income generation", "Capital preservation", "Speculation"]
Risk tolerance: ["Conservative", "Moderate", "Aggressive"]
Analysis focus: ["Technical analysis", "Fundamental analysis", "Both", "News & sentiment"]
</example_questions>

<rules>
1. Max 3 questions per clarification request
2. Make questions specific and actionable
3. Provide sensible defaults
4. Match questions to user's language
5. Be efficient - only ask what's truly needed
</rules>"""


def generate_clarification_prompt(
    query: str,
    symbols: Optional[List[str]] = None,
    user_context: Optional[str] = None,
) -> str:
    """
    Generate the user message for clarification analysis.

    Args:
        query: User's research query
        symbols: List of detected symbols
        user_context: Optional user profile/history context

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        f"## User Query\n{query}",
    ]

    if symbols:
        prompt_parts.append(f"\n## Detected Symbols\n{', '.join(symbols)}")

    if user_context:
        prompt_parts.append(f"\n## User Context\n{user_context}")

    prompt_parts.append("""
## Task
Analyze this query and determine if clarification questions are needed.
If the query is clear and specific, set needs_clarification to false.
If clarification would help provide better research, generate 1-3 focused questions.

Respond with the JSON format specified in your instructions.""")

    return "\n".join(prompt_parts)


# Common clarification question templates
COMMON_QUESTIONS = {
    "time_horizon": {
        "question_id": "time_horizon",
        "question": "What is your investment time horizon?",
        "question_type": "single_choice",
        "options": [
            "Short-term (< 6 months)",
            "Medium-term (6-24 months)",
            "Long-term (> 2 years)",
        ],
        "default": "Medium-term (6-24 months)",
        "required": True,
    },
    "investment_goal": {
        "question_id": "investment_goal",
        "question": "What is your primary investment goal?",
        "question_type": "single_choice",
        "options": [
            "Capital growth",
            "Income generation",
            "Capital preservation",
            "Speculation/Trading",
        ],
        "default": "Capital growth",
        "required": True,
    },
    "risk_tolerance": {
        "question_id": "risk_tolerance",
        "question": "What is your risk tolerance?",
        "question_type": "single_choice",
        "options": [
            "Conservative (minimize losses)",
            "Moderate (balanced risk/reward)",
            "Aggressive (maximize returns)",
        ],
        "default": "Moderate (balanced risk/reward)",
        "required": True,
    },
    "analysis_focus": {
        "question_id": "analysis_focus",
        "question": "What type of analysis would you like to focus on?",
        "question_type": "multi_choice",
        "options": [
            "Technical analysis (charts, patterns)",
            "Fundamental analysis (financials, valuation)",
            "News & sentiment analysis",
            "Competitive analysis",
        ],
        "default": None,
        "required": False,
    },
    "specific_concerns": {
        "question_id": "specific_concerns",
        "question": "Do you have any specific concerns or areas you want to focus on?",
        "question_type": "text",
        "options": [],
        "default": None,
        "required": False,
    },
}
