"""
BaseTool - Atomic Tool Interface for Financial Trading Agent
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
import json


# ============================================================================
# Models
# ============================================================================

class ToolParameter(BaseModel):
    """Parameter definition for atomic tool"""
    name: str
    type: str  # "string", "number", "array", "object", "boolean"
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    examples: List[Any] = Field(default_factory=list)
    # examples: List[str] = Field(default_factory=list)


class ToolOutput(BaseModel):
    """Standardized tool output"""
    tool_name: str
    status: str  # "success", "error", "partial"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    formatted_context: Optional[str] = None 
    raw_data: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_success(self) -> bool:
        """Check if execution was successful"""
        return self.status == "success"
    
    def has_data(self) -> bool:
        """Check if output has data"""
        return self.data is not None and len(self.data) > 0


class ToolSchema(BaseModel):
    """
    JSON Schema cho atomic tool
    
    Đây là "contract" mà tool phải tuân thủ:
    - Planning Agent dùng để lập kế hoạch
    - Validation Agent dùng để validate output
    - Tool Registry dùng để discover tools
    """
    name: str
    category: str  # "price", "technical", "fundamentals", "risk", "news", "market", "crypto"
    description: str
    
    # Usage hints cho Planning Agent
    usage_hints: List[str] = Field(
        default_factory=list,
        description="Khi nào nên dùng tool này"
    )
    
    # Capabilities & Limitations (CRITICAL cho Planning)
    capabilities: List[str] = Field(
        default_factory=list,
        description="Tool này CÓ THỂ làm gì"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Tool này KHÔNG THỂ làm gì"
    )
    
    # Input
    parameters: List[ToolParameter] = Field(default_factory=list)
    requires_symbol: bool = True
    
    # Output (Expected fields)
    returns: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected output schema"
    )
    
    # Examples
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance
    typical_execution_time_ms: Optional[int] = 1000
    
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling format
        
        This is used by Planning Agent để hiểu tools
        """
        parameters_obj = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            
            if param.enum:
                prop["enum"] = param.enum
            if param.pattern:
                prop["pattern"] = param.pattern
            if param.examples:
                prop["examples"] = param.examples
            if param.default is not None:
                prop["default"] = param.default
            if param.min_value is not None:
                prop["minimum"] = param.min_value
            if param.max_value is not None:
                prop["maximum"] = param.max_value
            
            parameters_obj["properties"][param.name] = prop
            
            if param.required:
                parameters_obj["required"].append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters_obj,
            "metadata": {
                "category": self.category,
                "capabilities": self.capabilities,
                "limitations": self.limitations,
                "usage_hints": self.usage_hints,
                "requires_symbol": self.requires_symbol
            }
        }
    
    def get_required_fields(self) -> List[str]:
        """Get list of required output fields for validation"""
        return list(self.returns.keys())


# ============================================================================
# BaseTool - Atomic Tool Interface
# ============================================================================

class BaseTool(ABC):
    """
    Base class cho tất cả atomic tools
    
    Mỗi tool PHẢI:
    1. Define schema trong __init__
    2. Implement execute() method
    3. Return ToolOutput
    
    Tool KHÔNG ĐƯỢC:
    1. Phụ thuộc vào output của tool khác
    2. Làm nhiều việc (violate single responsibility)
    3. Có side effects (modify global state)
    4. Cache data internally (dùng external cache service)
    
    Example:
        class GetStockPriceTool(BaseTool):
            def __init__(self):
                super().__init__()
                self.schema = ToolSchema(
                    name="getStockPrice",
                    category="price",
                    description="Get current stock price and basic metrics",
                    capabilities=[
                        "Real-time price data",
                        "Daily change percentage",
                        "Volume data"
                    ],
                    limitations=[
                        "❌ NO technical indicators",
                        "❌ NO historical data",
                        "❌ NO financial ratios"
                    ],
                    parameters=[
                        ToolParameter(
                            name="symbol",
                            type="string",
                            description="Stock symbol (e.g., AAPL, NVDA)",
                            required=True,
                            pattern="^[A-Z]{1,7}$"
                        )
                    ],
                    returns={
                        "symbol": "string",
                        "price": "number",
                        "change": "number",
                        "change_percent": "number",
                        "volume": "number"
                    }
                )
            
            async def execute(self, symbol: str) -> ToolOutput:
                # Fetch from FMP
                data = await self.fmp_service.get_quote(symbol)
                
                return create_success_output(
                    tool_name=self.schema.name,
                    data={
                        "symbol": symbol,
                        "price": data['price'],
                        "change": data['change'],
                        "change_percent": data['changesPercentage'],
                        "volume": data['volume']
                    }
                )
    """
    
    def __init__(self):
        self.schema: Optional[ToolSchema] = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def execute(self, **params) -> ToolOutput:
        """
        Execute the atomic tool
        
        Args:
            **params: Tool parameters as defined in schema
            
        Returns:
            ToolOutput with results or error
        """
        pass
    
    def validate_input(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input parameters against schema
        
        Args:
            params: Input parameters
            
        Returns:
            Validated parameters
            
        Raises:
            ValueError: If validation fails
        """
        if not self.schema:
            return params
        
        validated = {}
        errors = []
        
        for param in self.schema.parameters:
            # Check required
            if param.required and param.name not in params:
                if param.default is not None:
                    validated[param.name] = param.default
                else:
                    errors.append(f"Missing required parameter: {param.name}")
                    continue
            
            value = params.get(param.name, param.default)
            
            if value is None:
                if param.required:
                    errors.append(f"Parameter {param.name} cannot be None")
                continue
            
            # Type validation
            if param.type == "string" and not isinstance(value, str):
                errors.append(f"{param.name} must be string, got {type(value)}")
            elif param.type == "number" and not isinstance(value, (int, float)):
                errors.append(f"{param.name} must be number, got {type(value)}")
            elif param.type == "array" and not isinstance(value, list):
                errors.append(f"{param.name} must be array, got {type(value)}")
            elif param.type == "boolean" and not isinstance(value, bool):
                errors.append(f"{param.name} must be boolean, got {type(value)}")
            
            # Enum validation
            if param.enum and value not in param.enum:
                errors.append(f"{param.name} must be one of {param.enum}, got {value}")
            
            # Pattern validation (for strings)
            if param.type == "string" and param.pattern:
                import re
                if not re.match(param.pattern, str(value)):
                    errors.append(f"{param.name} must match pattern {param.pattern}, got {value}")
            
            # Range validation (for numbers)
            if param.type == "number":
                if param.min_value is not None and value < param.min_value:
                    errors.append(f"{param.name} must be >= {param.min_value}, got {value}")
                if param.max_value is not None and value > param.max_value:
                    errors.append(f"{param.name} must be <= {param.max_value}, got {value}")
            
            validated[param.name] = value
        
        if errors:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")
        
        return validated
    
    def validate_output(self, output: ToolOutput) -> bool:
        """
        Validate output data against schema
        
        Args:
            output: Tool output to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.schema or not output.data:
            return True
        
        # Check if all required fields exist
        required_fields = self.schema.get_required_fields()
        missing_fields = [
            field for field in required_fields
            if field not in output.data
        ]
        
        if missing_fields:
            self.logger.warning(
                f"Output missing required fields: {missing_fields}"
            )
            return False
        
        return True
    
    async def safe_execute(self, **params) -> ToolOutput:
        """
        Execute với error handling & timing
        
        Args:
            **params: Tool parameters
            
        Returns:
            ToolOutput
        """
        start_time = datetime.now()
        tool_name = self.schema.name if self.schema else self.__class__.__name__
        
        try:
            # Validate input
            validated_params = self.validate_input(params)
            
            # Log execution
            self.logger.info(
                f"[{tool_name}] Executing with params: {validated_params}"
            )
            
            # Execute
            result = await self.execute(**validated_params)
            
            # Add execution time
            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result.execution_time_ms = elapsed_ms
            
            # Validate output
            if not self.validate_output(result):
                result.status = "partial"
                result.metadata["validation_warning"] = "Missing expected fields"
            
            self.logger.info(
                f"[{tool_name}] ✅ {result.status.upper()} ({elapsed_ms}ms)"
            )
            
            return result
            
        except ValueError as e:
            # Validation error
            self.logger.error(f"[{tool_name}] ❌ Validation error: {e}")
            return ToolOutput(
                tool_name=tool_name,
                status="error",
                error=f"Validation error: {str(e)}"
            )
            
        except Exception as e:
            # Execution error
            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.logger.error(
                f"[{tool_name}] ❌ Execution error: {e}",
                exc_info=True
            )
            return ToolOutput(
                tool_name=tool_name,
                status="error",
                error=str(e),
                execution_time_ms=elapsed_ms
            )
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema"""
        if not self.schema:
            raise ValueError(f"Tool {self.__class__.__name__} has no schema defined")
        return self.schema


# ============================================================================
# Parallel Execution Utilities
# ============================================================================

async def execute_tools_parallel(
    tools: List[BaseTool],
    params_list: List[Dict[str, Any]]
) -> List[ToolOutput]:
    """
    Execute multiple tools in parallel
    
    Args:
        tools: List of tool instances
        params_list: List of parameters for each tool
        
    Returns:
        List of ToolOutputs in same order as input
        
    Example:
        from src.tools.price.get_stock_price import GetStockPriceTool
        from src.tools.technical.get_technical_indicators import GetTechnicalIndicatorsTool
        
        results = await execute_tools_parallel(
            tools=[GetStockPriceTool(), GetTechnicalIndicatorsTool()],
            params_list=[
                {"symbol": "AAPL"},
                {"symbol": "AAPL", "indicators": ["RSI", "MACD"]}
            ]
        )
    """
    tasks = [
        tool.safe_execute(**params)
        for tool, params in zip(tools, params_list)
    ]
    
    return await asyncio.gather(*tasks)


async def execute_tools_sequential(
    tools: List[BaseTool],
    params_list: List[Dict[str, Any]]
) -> List[ToolOutput]:
    """
    Execute multiple tools sequentially
    
    Use when tools need to run in order (rare for atomic tools)
    """
    results = []
    
    for tool, params in zip(tools, params_list):
        result = await tool.safe_execute(**params)
        results.append(result)
    
    return results


# ============================================================================
# Helper Functions
# ============================================================================

def create_success_output(
    tool_name: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    formatted_context: Optional[str] = None,
    symbols: Optional[List[str]] = None
) -> ToolOutput:
    """
    Helper to create success output
    
    Args:
        tool_name: Name of the tool
        data: Tool output data
        metadata: Optional additional metadata
        formatted_context: Optional human-readable context for LLM
        symbols: Optional list of symbols processed
        
    Returns:
        ToolOutput with status="success"
    """
    meta = metadata or {}
    
    # Add symbols to metadata if provided
    if symbols:
        meta["symbols"] = symbols
    
    return ToolOutput(
        tool_name=tool_name,
        status="success",
        data=data,
        formatted_context=formatted_context,
        metadata=meta
    )


def create_error_output(
    tool_name: str,
    error: Optional[str] = None,
    error_message: Optional[str] = None,  # Alias for error
    error_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ToolOutput:
    """
    Helper to create error output
    
    Args:
        tool_name: Name of the tool
        error: Error message (primary parameter)
        error_message: Alias for error (for backward compatibility)
        error_type: Type of error (validation_error, execution_error, etc.)
        metadata: Optional additional metadata
        
    Returns:
        ToolOutput with status="error"
    """
    # Use error_message as fallback if error not provided
    actual_error = error or error_message or "Unknown error"
    
    meta = metadata or {}
    
    # Add error_type to metadata if provided
    if error_type:
        meta["error_type"] = error_type
    
    return ToolOutput(
        tool_name=tool_name,
        status="error",
        error=actual_error,
        metadata=meta
    )


def create_partial_output(
    tool_name: str,
    data: Dict[str, Any],
    missing_fields: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    formatted_context: Optional[str] = None,
    symbols: Optional[List[str]] = None
) -> ToolOutput:
    """
    Helper to create partial success output
    
    Used when tool returns some data but not all expected fields.
    
    Args:
        tool_name: Name of the tool
        data: Partial data returned
        missing_fields: List of fields that are missing
        metadata: Optional additional metadata
        formatted_context: Optional human-readable context for LLM
        symbols: Optional list of symbols processed
        
    Returns:
        ToolOutput with status="partial"
    """
    meta = metadata or {}
    
    if missing_fields:
        meta["missing_fields"] = missing_fields
    if symbols:
        meta["symbols"] = symbols
    
    return ToolOutput(
        tool_name=tool_name,
        status="partial",
        data=data,
        formatted_context=formatted_context,
        metadata=meta
    )