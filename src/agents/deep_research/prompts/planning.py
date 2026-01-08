"""
Planning Prompts for Deep Research

Used by the orchestrator to create research plans.
The plan defines sections, workers, and execution strategy.
"""

from typing import List, Optional, Dict, Any


PLANNING_SYSTEM_PROMPT = """You are a research planning expert. Your job is to create comprehensive research plans for financial analysis queries.

## Your Role
- Analyze the user's research query and clarification answers
- Create a structured research plan with clear sections
- Assign appropriate worker roles to each section
- Estimate execution time and resources

## Available Worker Roles
1. **market_analyst**: Market position, industry trends, competitive landscape
   - Tools: web_search, get_market_news, get_sector_performance

2. **financial_analyst**: Financial statements, ratios, valuation metrics
   - Tools: get_financial_ratios, get_income_statement, get_balance_sheet, get_cash_flow

3. **technical_analyst**: Price charts, technical indicators, patterns
   - Tools: get_stock_price, get_technical_indicators, detect_chart_patterns, get_support_resistance

4. **web_researcher**: News, recent developments, analyst opinions
   - Tools: web_search, get_stock_news, get_company_events

5. **risk_analyst**: Risk assessment, volatility, downside analysis
   - Tools: assess_risk, get_sentiment, get_volume_profile

## Research Plan Structure
Create 3-5 sections that comprehensively cover the query. Each section should:
- Have a clear, specific objective
- Be assigned to the most appropriate worker role
- List the tools needed
- Have an estimated duration (30-120 seconds typically)

## Response Format
Respond with a JSON object:
{
    "title": "Research title",
    "objective": "Clear statement of what this research will accomplish",
    "estimated_duration_min": 5,
    "sections": [
        {
            "id": 1,
            "name": "Section Name",
            "description": "What this section will cover",
            "worker_role": "financial_analyst",
            "tools_needed": ["get_financial_ratios", "get_income_statement"],
            "estimated_duration_sec": 60,
            "priority": 1,
            "dependencies": []
        }
    ],
    "key_questions": [
        "Key question this research will answer"
    ]
}

## Guidelines
1. Start with the most critical sections (priority 1)
2. Consider dependencies between sections
3. Balance depth vs breadth based on query complexity
4. For multi-symbol queries, group by analysis type, not by symbol
5. Ensure the plan addresses all aspects of the user's query
"""


def generate_planning_prompt(
    query: str,
    symbols: List[str],
    clarification_answers: Optional[Dict[str, str]] = None,
    user_context: Optional[str] = None,
) -> str:
    """
    Generate the user message for research planning.

    Args:
        query: User's research query
        symbols: List of symbols to analyze
        clarification_answers: User's answers to clarification questions
        user_context: Optional user profile context

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        f"## Research Query\n{query}",
        f"\n## Symbols to Analyze\n{', '.join(symbols) if symbols else 'No specific symbols'}",
    ]

    if clarification_answers:
        answers_text = "\n".join([
            f"- {k}: {v}" for k, v in clarification_answers.items()
        ])
        prompt_parts.append(f"\n## User Preferences\n{answers_text}")

    if user_context:
        prompt_parts.append(f"\n## User Context\n{user_context}")

    prompt_parts.append("""
## Task
Create a comprehensive research plan for this query.
The plan should:
1. Cover all aspects of the user's question
2. Use appropriate worker roles for each section
3. Be achievable within 5-10 minutes
4. Prioritize the most important analyses

Respond with the JSON format specified in your instructions.""")

    return "\n".join(prompt_parts)


# Template sections for common research types
TEMPLATE_SECTIONS = {
    "stock_analysis": [
        {
            "name": "Price & Technical Analysis",
            "description": "Current price, technical indicators, support/resistance levels",
            "worker_role": "technical_analyst",
            "tools_needed": ["get_stock_price", "get_technical_indicators", "get_support_resistance"],
            "priority": 1,
        },
        {
            "name": "Financial Health",
            "description": "Key financial ratios, profitability, growth metrics",
            "worker_role": "financial_analyst",
            "tools_needed": ["get_financial_ratios", "get_growth_metrics"],
            "priority": 1,
        },
        {
            "name": "Market Position & News",
            "description": "Recent news, market sentiment, industry trends",
            "worker_role": "web_researcher",
            "tools_needed": ["get_stock_news", "web_search"],
            "priority": 2,
        },
        {
            "name": "Risk Assessment",
            "description": "Volatility, risk factors, downside scenarios",
            "worker_role": "risk_analyst",
            "tools_needed": ["assess_risk", "get_sentiment"],
            "priority": 2,
        },
    ],
    "comparison": [
        {
            "name": "Price Comparison",
            "description": "Compare prices, performance, and technical indicators",
            "worker_role": "technical_analyst",
            "tools_needed": ["get_stock_price", "get_stock_performance", "get_technical_indicators"],
            "priority": 1,
        },
        {
            "name": "Fundamental Comparison",
            "description": "Compare financial metrics and valuations",
            "worker_role": "financial_analyst",
            "tools_needed": ["get_financial_ratios", "get_growth_metrics"],
            "priority": 1,
        },
        {
            "name": "Competitive Analysis",
            "description": "Market position, competitive advantages",
            "worker_role": "market_analyst",
            "tools_needed": ["web_search", "get_sector_performance"],
            "priority": 2,
        },
    ],
    "sector_analysis": [
        {
            "name": "Sector Overview",
            "description": "Sector performance, trends, key players",
            "worker_role": "market_analyst",
            "tools_needed": ["get_sector_performance", "get_market_news"],
            "priority": 1,
        },
        {
            "name": "Top Performers",
            "description": "Best performing stocks in sector",
            "worker_role": "technical_analyst",
            "tools_needed": ["get_top_gainers", "stock_screener"],
            "priority": 1,
        },
        {
            "name": "Sector News",
            "description": "Recent developments affecting the sector",
            "worker_role": "web_researcher",
            "tools_needed": ["web_search", "get_market_news"],
            "priority": 2,
        },
    ],
}
