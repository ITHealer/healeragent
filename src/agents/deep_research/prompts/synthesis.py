"""
Synthesis Prompts for Deep Research

Used to combine worker results into coherent final reports.
"""

from typing import List, Dict, Any, Optional


SYNTHESIS_SYSTEM_PROMPT = """<identity>
You are HealerAgent Report Synthesizer, a financial research report writer.
Created by ToponeLogic. Expert at combining multiple research findings into comprehensive reports.
</identity>

<role>
- Combine findings from multiple research workers
- Create coherent narrative from diverse data points
- Highlight key insights and actionable recommendations
- Maintain accuracy - ONLY use data from provided findings
</role>

<report_structure>
1. EXECUTIVE SUMMARY (2-3 paragraphs)
   - Key findings at a glance
   - Main recommendation with confidence level
   - Risk factors to consider

2. DETAILED ANALYSIS (by topic, use available data)
   - Technical Analysis
   - Fundamental Analysis
   - Market Position & News
   - Risk Assessment

3. KEY INSIGHTS (bullet points)
   - Most important takeaways
   - Surprising findings
   - Areas of concern

4. RECOMMENDATIONS
   - Clear action items with specific prices/levels
   - Conditions/triggers to watch
   - Time horizon considerations

5. SOURCES & METHODOLOGY
   - Data sources used
   - Analysis methods applied
</report_structure>

<synthesis_rules>
1. ACCURACY: Only include data from provided findings - NEVER fabricate numbers
2. CLARITY: Use clear language, explain technical terms on first use
3. BALANCE: Present both bullish AND bearish perspectives
4. ACTIONABLE: Specific recommendations with entry/exit levels
5. STRUCTURED: Headers, bullets, tables for readability
6. DIRECT: No flattery or hedging - start with key verdict
</synthesis_rules>

<output_format>
Provide report in clean Markdown format.
Use tables for comparing data across symbols.
Include specific numbers and percentages.
Bold key metrics: **P/E = 25x**
Match user's language for the report.
</output_format>

<language_rules>
Match the language of the original query:
- Vietnamese query → Vietnamese report (keep technical terms in English + explanation)
- English query → English report
- Chinese query → Chinese report
</language_rules>"""


def generate_synthesis_prompt(
    query: str,
    worker_findings: List[Dict[str, Any]],
    clarification_answers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate the user message for report synthesis.

    Args:
        query: Original user query
        worker_findings: List of findings from each worker
        clarification_answers: User's preferences/answers

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        f"## Original Query\n{query}",
    ]

    if clarification_answers:
        answers_text = "\n".join([
            f"- {k}: {v}" for k, v in clarification_answers.items()
        ])
        prompt_parts.append(f"\n## User Preferences\n{answers_text}")

    prompt_parts.append("\n## Research Findings\n")

    for i, finding in enumerate(worker_findings, 1):
        section_name = finding.get("section_name", f"Section {i}")
        worker_role = finding.get("worker_role", "unknown")
        content = finding.get("findings", finding.get("content", "No findings"))
        data = finding.get("data", {})

        prompt_parts.append(f"### {section_name} ({worker_role})")
        prompt_parts.append(content)

        if data:
            prompt_parts.append(f"\n**Data:**\n```json\n{data}\n```")

        prompt_parts.append("")

    prompt_parts.append("""
## Task
Synthesize these findings into a comprehensive research report.
Follow the report structure specified in your instructions.
Ensure all data points are accurately represented.
Provide clear, actionable recommendations based on the evidence.""")

    return "\n".join(prompt_parts)


def generate_executive_summary_prompt(
    query: str,
    full_report: str,
    symbols: List[str],
) -> str:
    """
    Generate prompt for creating executive summary.

    Args:
        query: Original query
        full_report: The complete research report
        symbols: Symbols analyzed

    Returns:
        Formatted prompt string
    """
    return f"""## Task
Create a concise executive summary (3-4 paragraphs) for this research report.

## Original Query
{query}

## Symbols Analyzed
{', '.join(symbols)}

## Full Report
{full_report}

## Requirements
1. Start with the main conclusion/recommendation
2. Highlight 3-5 key findings
3. Mention major risk factors
4. Keep it under 400 words
5. Make it actionable - what should the reader do?

Write the executive summary now:"""


def generate_key_findings_prompt(
    full_report: str,
) -> str:
    """
    Generate prompt for extracting key findings.

    Args:
        full_report: The complete research report

    Returns:
        Formatted prompt string
    """
    return f"""## Task
Extract 5-7 key findings from this research report.

## Full Report
{full_report}

## Requirements
- Each finding should be a single, clear sentence
- Focus on the most important/surprising insights
- Include specific numbers where relevant
- Mix bullish and bearish findings
- Make them actionable/informative

Respond with a JSON array of strings:
["Finding 1", "Finding 2", ...]"""


# Report templates
REPORT_TEMPLATES = {
    "stock_analysis": """# {title}

## Executive Summary
{executive_summary}

---

## Technical Analysis
{technical_section}

## Fundamental Analysis
{fundamental_section}

## Market Position & News
{market_section}

## Risk Assessment
{risk_section}

---

## Key Findings
{key_findings}

## Recommendations
{recommendations}

---

## Sources
{sources}

*Report generated on {date}*
""",

    "comparison": """# {title}

## Executive Summary
{executive_summary}

---

## Comparison Overview
{comparison_table}

## Detailed Analysis

### {symbol1} Analysis
{symbol1_analysis}

### {symbol2} Analysis
{symbol2_analysis}

## Comparative Insights
{comparative_insights}

---

## Recommendations
{recommendations}

---

## Sources
{sources}

*Report generated on {date}*
""",
}
