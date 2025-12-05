# File: src/agents/synthesis/synthesis_agent.py
"""
Synthesis Agent - Anthropic Multi-Source Synthesis Pattern

BASED ON:
- Weaviate Context Engineering: "Multi Source Synthesis"
- Anthropic Agent SDK: "Compression Through Parallelization"

RESPONSIBILITIES:
1. Aggregate results from MULTIPLE tool calls
2. Resolve conflicts between sources
3. Extract key insights and patterns
4. Generate coherent, actionable summary

GENERIC APPROACH:
- NO hardcoded query-specific logic
- Adapt to ANY tool combination
- Context-aware synthesis based on data structure
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.tools.base import ToolOutput

class SynthesisAgent(LoggerMixin):
    """
    Synthesis Agent - Generic Multi-Source Data Aggregation
    
    From Anthropic SDK:
    "Subagents use isolated context windows, only send RELEVANT 
    information back to orchestrator rather than full context"
    """
    
    def __init__(self, llm_provider):
        super().__init__()
        self.llm_provider = llm_provider
    
    # ========================================================================
    # MAIN SYNTHESIS METHOD
    # ========================================================================
    
    async def synthesize_results(
        self,
        query: str,
        tool_results: Dict[int, Any],
        language: str = "vi"
    ) -> str:
        """
        Synthesize results from MULTIPLE tool executions
        
        ANTHROPIC PATTERN:
        - Extract relevant excerpts (not full context)
        - Identify patterns and conflicts
        - Generate concise insights
        
        Args:
            query: Original user query
            tool_results: Results from ALL tasks
            language: Response language
            
        Returns:
            Synthesized insights string for LLM context
        """
        
        self.logger.info(f"[SYNTHESIS] Starting synthesis for {len(tool_results)} task results")
        
        # ════════════════════════════════════════════════════════════
        # STEP 1: Organize data by tool type and symbol
        # ════════════════════════════════════════════════════════════
        # organized_data = self._organize_by_tool_and_symbol(tool_results)
        organized_data = self._organize_formatted_results(tool_results)

        if not organized_data:
            self.logger.warning("[SYNTHESIS] No data to synthesize")
            return ""
        
        # ════════════════════════════════════════════════════════════
        # STEP 2: Extract key metrics across all symbols
        # ════════════════════════════════════════════════════════════
        extracted_metrics = self._extract_key_metrics(organized_data)
        
        # ════════════════════════════════════════════════════════════
        # STEP 3: Identify patterns and anomalies
        # ════════════════════════════════════════════════════════════
        # patterns = self._identify_patterns(extracted_metrics)
        patterns = self._identify_patterns(tool_results)

        # ════════════════════════════════════════════════════════════
        # STEP 4: Build synthesis prompt
        # ════════════════════════════════════════════════════════════
        synthesis_prompt = self._build_synthesis_prompt(
            query=query,
            extracted_metrics=organized_data,
            patterns=patterns,
            language=language
        )
        
        # ════════════════════════════════════════════════════════════
        # STEP 5: Generate synthesis via LLM
        # ════════════════════════════════════════════════════════════
        try:
            response = await self.llm_provider.generate_response(
                messages=[
                    {"role": "system", "content": self._get_system_prompt(language)},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.2,  # Low temp for factual synthesis
                max_tokens=1500
            )
            
            if isinstance(response, dict):
                synthesized = response.get("content", "")
            else:
                synthesized = str(response)
            
            self.logger.info(
                f"[SYNTHESIS] ✅ Generated {len(synthesized)} chars from "
                f"{len(extracted_metrics)} symbols, {len(patterns)} patterns"
            )
            
            return synthesized
            
        except Exception as e:
            self.logger.error(f"[SYNTHESIS] Error generating synthesis: {e}", exc_info=True)
            return ""
    
    # ========================================================================
    # DATA ORGANIZATION
    # ========================================================================
    def _organize_formatted_results(
        self,
        tool_results: Dict[str, ToolOutput]
    ) -> Dict[str, Dict[str, str]]:
        """
        Organize results BY SYMBOL using FORMATTED context
        
        Returns:
            {
                "NVDA": {
                    "technical": "Formatted technical analysis...",
                    "news": "Formatted news summary...",
                    "fundamental": "Formatted fundamental data..."
                },
                ...
            }
        """
        
        organized = {}
        
        for tool_name, tool_output in tool_results.items():
            if tool_output.status != 'success':
                continue
            
            # Extract symbols
            symbols = self._extract_symbols_from_output(tool_output.data)
            
            # Get formatted context
            formatted = tool_output.formatted_context
            if not formatted:
                # Fallback to raw
                formatted = json.dumps(tool_output.data, indent=2)
            
            # Categorize tool
            category = self._categorize_tool(tool_name)
            
            # Group by symbol
            for symbol in symbols:
                if symbol not in organized:
                    organized[symbol] = {}
                
                # ✅ Store FORMATTED context, not raw data
                organized[symbol][category] = formatted
        
        return organized

    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tool by type"""
        
        categories = {
            'technical': ['Technical', 'Chart', 'Pattern', 'Indicator', 'Support', 'Resistance', 'Volume'],
            'fundamental': ['Financial', 'Ratio', 'Income', 'Balance', 'CashFlow', 'Growth'],
            'news': ['News', 'Events', 'Earnings', 'Calendar'],
            'risk': ['Risk', 'Sentiment', 'StopLoss', 'Volatility'],
            'price': ['Price', 'Performance', 'Quote'],
            'screener': ['Screener']
        }
        
        for category, keywords in categories.items():
            if any(keyword.lower() in tool_name.lower() for keyword in keywords):
                return category
        
        return 'other'
    
    
    # ========================================================================
    # METRIC EXTRACTION
    # ========================================================================
    
    def _extract_key_metrics(
        self,
        organized_data: Dict[str, Dict[str, List[Dict]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract key metrics for EACH symbol
        
        Returns:
            {
              "AAPL": {
                "symbol": "AAPL",
                "technical": {rsi: 45.2, macd: {...}, ...},
                "fundamental": {pe: 28.5, roe: 0.42, ...},
                "news": {sentiment: "positive", count: 5, ...},
                "risk": {level: "moderate", ...}
              },
              ...
            }
        """
        
        metrics = {}
        
        for symbol, tool_categories in organized_data.items():
            metrics[symbol] = {
                "symbol": symbol,
                "data_available": list(tool_categories.keys())
            }
            
            # Extract from each tool category
            for category, tool_results in tool_categories.items():
                if category == 'technical':
                    metrics[symbol]['technical'] = self._extract_technical_metrics(tool_results)
                
                elif category == 'fundamental':
                    metrics[symbol]['fundamental'] = self._extract_fundamental_metrics(tool_results)
                
                elif category == 'news':
                    metrics[symbol]['news'] = self._extract_news_metrics(tool_results)
                
                elif category == 'risk':
                    metrics[symbol]['risk'] = self._extract_risk_metrics(tool_results)
                
                elif category == 'price':
                    metrics[symbol]['price'] = self._extract_price_metrics(tool_results)
        
        return metrics
    
    def _extract_technical_metrics(self, tool_results: List[Dict]) -> Dict:
        """Extract technical indicator metrics"""
        
        metrics = {}
        
        for result in tool_results:
            data = result.get('data', {})
            
            # RSI
            if 'rsi' in data:
                metrics['rsi'] = data['rsi']
                if isinstance(metrics['rsi'], dict):
                    metrics['rsi'] = metrics['rsi'].get('current', metrics['rsi'].get('value'))
            
            # MACD
            if 'macd' in data:
                metrics['macd'] = data['macd']
            
            # Moving Averages
            for key in ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26']:
                if key in data:
                    metrics[key] = data[key]
            
            # Patterns
            if 'patterns_detected' in data:
                metrics['patterns'] = data['patterns_detected']
            
            # Bollinger Bands
            if 'bollinger_bands' in data:
                metrics['bollinger'] = data['bollinger_bands']
        
        return metrics
    
    def _extract_fundamental_metrics(self, tool_results: List[Dict]) -> Dict:
        """Extract fundamental metrics"""
        
        metrics = {}
        
        for result in tool_results:
            data = result.get('data', {})
            
            # Key ratios
            for key in ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity', 'current_ratio']:
                if key in data:
                    metrics[key] = data[key]
            
            # Growth metrics
            for key in ['revenue_growth', 'earnings_growth', 'eps']:
                if key in data:
                    metrics[key] = data[key]
        
        return metrics
    
    def _extract_news_metrics(self, tool_results: List[Dict]) -> Dict:
        """Extract news/sentiment metrics"""
        
        metrics = {
            'article_count': 0,
            'recent_headlines': [],
            'sentiment': None
        }
        
        for result in tool_results:
            data = result.get('data', {})
            
            if 'articles' in data:
                articles = data['articles']
                metrics['article_count'] += len(articles)
                
                # Get recent headlines (max 3)
                for article in articles[:3]:
                    if isinstance(article, dict):
                        title = article.get('title', article.get('headline', ''))
                        if title:
                            metrics['recent_headlines'].append(title)
            
            if 'sentiment' in data:
                metrics['sentiment'] = data['sentiment']
        
        return metrics
    
    def _extract_risk_metrics(self, tool_results: List[Dict]) -> Dict:
        """Extract risk metrics"""
        
        metrics = {}
        
        for result in tool_results:
            data = result.get('data', {})
            
            for key in ['risk_level', 'volatility', 'beta', 'max_drawdown']:
                if key in data:
                    metrics[key] = data[key]
        
        return metrics
    
    def _extract_price_metrics(self, tool_results: List[Dict]) -> Dict:
        """Extract price metrics"""
        
        metrics = {}
        
        for result in tool_results:
            data = result.get('data', {})
            
            for key in ['current_price', 'change_percent', 'volume', 'high', 'low']:
                if key in data:
                    metrics[key] = data[key]
        
        return metrics
    
    # ========================================================================
    # PATTERN IDENTIFICATION
    # ========================================================================
    
    def _identify_patterns(
        self,
        extracted_metrics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify patterns and anomalies across symbols
        
        Returns list of patterns:
            [
              {
                "type": "oversold_condition",
                "symbols": ["NVDA", "AMD"],
                "description": "RSI < 30",
                "severity": "high"
              },
              ...
            ]
        """
        
        patterns = []
        
        # Pattern 1: Oversold/Overbought conditions
        oversold_symbols = []
        overbought_symbols = []
        
        for symbol, metrics in extracted_metrics.items():
            tech = metrics.get('technical', {})
            rsi = tech.get('rsi')
            
            if rsi is not None:
                try:
                    rsi_val = float(rsi)
                    if rsi_val < 30:
                        oversold_symbols.append(symbol)
                    elif rsi_val > 70:
                        overbought_symbols.append(symbol)
                except (ValueError, TypeError):
                    pass
        
        if oversold_symbols:
            patterns.append({
                "type": "oversold_condition",
                "symbols": oversold_symbols,
                "description": "RSI indicates oversold (< 30)",
                "severity": "high",
                "actionable": True
            })
        
        if overbought_symbols:
            patterns.append({
                "type": "overbought_condition",
                "symbols": overbought_symbols,
                "description": "RSI indicates overbought (> 70)",
                "severity": "medium",
                "actionable": True
            })
        
        # Pattern 2: Positive news + technical setup
        strong_setups = []
        
        for symbol, metrics in extracted_metrics.items():
            news = metrics.get('news', {})
            tech = metrics.get('technical', {})
            
            has_positive_news = news.get('sentiment') == 'positive'
            has_oversold_rsi = False
            
            rsi = tech.get('rsi')
            if rsi:
                try:
                    has_oversold_rsi = float(rsi) < 35
                except (ValueError, TypeError):
                    pass
            
            if has_positive_news and has_oversold_rsi:
                strong_setups.append(symbol)
        
        if strong_setups:
            patterns.append({
                "type": "strong_setup",
                "symbols": strong_setups,
                "description": "Positive news + oversold technical condition",
                "severity": "high",
                "actionable": True
            })
        
        # Pattern 3: High valuation warnings
        high_valuation = []
        
        for symbol, metrics in extracted_metrics.items():
            fund = metrics.get('fundamental', {})
            pe = fund.get('pe_ratio')
            
            if pe:
                try:
                    if float(pe) > 50:
                        high_valuation.append(symbol)
                except (ValueError, TypeError):
                    pass
        
        if high_valuation:
            patterns.append({
                "type": "high_valuation",
                "symbols": high_valuation,
                "description": "P/E ratio > 50 (expensive valuation)",
                "severity": "medium",
                "actionable": False
            })
        
        self.logger.info(f"[SYNTHESIS] Identified {len(patterns)} patterns")
        
        return patterns
    
    # ========================================================================
    # SYNTHESIS PROMPT BUILDING
    # ========================================================================
    
    def _build_synthesis_prompt(
        self,
        query: str,
        organized_formatted: Dict[str, Dict[str, str]],  # ← Formatted
        patterns: List[Dict],
        language: str
    ) -> str:
        """
        Build synthesis prompt with FORMATTED insights
        
        CRITICAL: This prompt now includes domain knowledge from formatter
        """
        
        prompt = f"""
    ORIGINAL QUERY: {query}
    RESPONSE LANGUAGE: {language}

    ═══════════════════════════════════════════════════════════
    FORMATTED DATA ANALYSIS (WITH DOMAIN INSIGHTS)
    ═══════════════════════════════════════════════════════════

    """
        
        for symbol, categories in organized_formatted.items():
            prompt += f"\n{'═' * 60}\n"
            prompt += f"SYMBOL: {symbol}\n"
            prompt += f"{'═' * 60}\n\n"
            
            for category, formatted_text in categories.items():
                prompt += formatted_text  # ← Includes interpretations!
                prompt += "\n\n"
        
        # Add patterns
        if patterns:
            prompt += f"\n{'═' * 60}\n"
            prompt += "IDENTIFIED PATTERNS:\n"
            prompt += f"{'═' * 60}\n\n"
            prompt += self._format_patterns(patterns)
        
        # Instructions
        prompt += f"""

    YOUR TASK:
    1. FILTER & RANK: Use the formatted insights above to identify top candidates
    2. KEY INSIGHTS: Extract 3-5 actionable insights (insights already provided in formatted data)
    3. RECOMMENDATIONS: Top 2-3 symbols with clear rationale

    RULES:
    ✅ Leverage the formatted interpretations provided (e.g., "Oversold", "Positive sentiment")
    ✅ Use SPECIFIC numbers from formatted data
    ✅ Be concise - don't repeat all formatted data
    ❌ Don't say "no data" when formatted insights exist
    ❌ Don't redirect to external sources
    """
        
        return prompt
    
    def _get_system_prompt(self, language: str) -> str:
        """Get system prompt for synthesis"""
        
        if language == "vi":
            return """Bạn là Financial Synthesis AI chuyên tổng hợp dữ liệu từ nhiều nguồn.

NHIỆM VỤ:
- Tổng hợp dữ liệu từ NHIỀU công cụ phân tích
- Xác định mẫu hình và điểm nổi bật
- Đưa ra khuyến nghị CỤ THỂ với SỐ LIỆU

NGUYÊN TẮC:
✅ Dựa vào dữ liệu CÓ SẴN (đã cung cấp)
✅ Số liệu CHÍNH XÁC (RSI 28.3, P/E 15.2)
✅ Kết nối nhiều nguồn (kỹ thuật + tin tức + cơ bản)
❌ KHÔNG bao giờ nói "chưa có dữ liệu"
❌ KHÔNG yêu cầu kiểm tra ngoài hệ thống
"""
        else:
            return """You are a Financial Synthesis AI specialized in multi-source data aggregation.

MISSION:
- Synthesize data from MULTIPLE analysis tools
- Identify patterns and key highlights
- Provide SPECIFIC recommendations with DATA

PRINCIPLES:
✅ Use AVAILABLE data (provided above)
✅ PRECISE numbers (RSI 28.3, P/E 15.2)
✅ Connect multiple sources (technical + news + fundamental)
❌ NEVER say "no data available"
❌ NEVER ask to check external sources
"""