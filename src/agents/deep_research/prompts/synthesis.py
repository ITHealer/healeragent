"""
Synthesis Prompts for Deep Research

Used to combine worker results into coherent final reports.
"""

from typing import List, Dict, Any, Optional


SYNTHESIS_SYSTEM_PROMPT = """You are a financial research report writer. Your job is to synthesize multiple research findings into a comprehensive, well-structured report.

## Your Role
- Combine findings from multiple research workers
- Create a coherent narrative from diverse data points
- Highlight key insights and actionable recommendations
- Maintain accuracy - only use data from provided findings

## Report Structure
Your report should include:
1. **Executive Summary** (2-3 paragraphs)
   - Key findings at a glance
   - Main recommendation
   - Risk factors to consider

2. **Detailed Analysis** (organized by topic)
   - Technical Analysis (if data available)
   - Fundamental Analysis (if data available)
   - Market Position & News (if data available)
   - Risk Assessment (if data available)

3. **Key Insights** (bullet points)
   - Most important takeaways
   - Surprising findings
   - Areas of concern

4. **Recommendations**
   - Clear action items
   - Conditions/triggers to watch
   - Time horizon considerations

5. **Sources & Methodology**
   - Data sources used
   - Analysis methods applied

## Guidelines
1. **Accuracy**: Only include data from the provided findings. Never make up numbers.
2. **Clarity**: Use clear language, explain technical terms.
3. **Balance**: Present both bullish and bearish perspectives.
4. **Actionable**: Provide clear, specific recommendations.
5. **Structured**: Use headers, bullet points, and tables for readability.

## Response Format
Provide the report in clean Markdown format.
Use tables for comparing data across symbols.
Include specific numbers and percentages where available.
"""


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
