import asyncio
import itertools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Callable, Any, Optional
from urllib.parse import urlparse

import httpx

from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType

MAX_REQUESTS_PER_ERROR = 15

ThinkingMode = Literal["enable", "disable", "default"]


class AgentResultError(ValueError):
    def __init__(self, message):
        super().__init__(message)


class PartialAgentResultError(ValueError):
    def __init__(self, message, partial_result: dict, append_prompt: str = None):
        super().__init__(message)
        self.partial_result = partial_result
        self.append_prompt = append_prompt


@dataclass
class AgentConfig:
    base_url: str
    api_key: Optional[str] = None
    model_id: str = "gpt-4.1-nano"
    temperature: float = 0.7
    concurrent: int = 30
    timeout: int = 1200  # seconds
    thinking: ThinkingMode = "disable"
    retry: int = 2
    system_proxy_enable: bool = False
    force_json: bool = False
    logger: Optional[logging.Logger] = None
    provider_type: ProviderType = ProviderType.OPENAI


class TotalErrorCounter:
    def __init__(self, logger: logging.Logger, max_errors_count: int = 10):
        self.lock = Lock()
        self.count = 0
        self.logger = logger
        self.max_errors_count = max_errors_count

    def add(self) -> bool:
        """Add error count. Returns True if limit reached."""
        with self.lock:
            self.count += 1
            if self.count > self.max_errors_count:
                self.logger.info(f"Lỗi quá nhiều: {self.count}/{self.max_errors_count}")
            return self.reach_limit()

    def reach_limit(self) -> bool:
        """Check if error limit reached"""
        return self.count > self.max_errors_count


class PromptsCounter:
    """Thread-safe counter for multi-threading"""
    def __init__(self, total: int, logger: logging.Logger):
        self.lock = Lock()
        self.count = 0
        self.total = total
        self.logger = logger

    def add(self):
        """Increment counter"""
        with self.lock:
            self.count += 1
            self.logger.info(f"Multi-thread - Hoàn thành: {self.count}/{self.total}")


def extract_token_info(response_data: dict) -> tuple[int, int, int, int]:
    """
    Extract token info từ API response.
    
    Supports multiple formats:
    1. usage.input_tokens_details.cached_tokens + usage.output_tokens_details.reasoning_tokens
    2. usage.prompt_tokens_details.cached_tokens
    3. usage.prompt_cache_hit_tokens + usage.completion_tokens_details.reasoning_tokens
    
    Returns:
        (input_tokens, cached_tokens, output_tokens, reasoning_tokens)
    """
    if "usage" not in response_data:
        return 0, 0, 0, 0

    usage = response_data["usage"]
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    cached_tokens = 0
    reasoning_tokens = 0
    
    try:
        # Extract cached_tokens - multiple formats
        if "input_tokens_details" in usage and "cached_tokens" in usage["input_tokens_details"]:
            cached_tokens = usage["input_tokens_details"]["cached_tokens"]
        elif "prompt_tokens_details" in usage and "cached_tokens" in usage["prompt_tokens_details"]:
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
        elif "prompt_cache_hit_tokens" in usage:
            cached_tokens = usage["prompt_cache_hit_tokens"]

        # Extract reasoning_tokens - multiple formats
        if "output_tokens_details" in usage and "reasoning_tokens" in usage["output_tokens_details"]:
            reasoning_tokens = usage["output_tokens_details"]["reasoning_tokens"]
        elif "completion_tokens_details" in usage and "reasoning_tokens" in usage["completion_tokens_details"]:
            reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
        
        return input_tokens, cached_tokens, output_tokens, reasoning_tokens
    
    except TypeError as e:
        return -1, -1, -1, -1


class TokenCounter:
    """Thread-safe token counter"""
    def __init__(self, logger: logging.Logger):
        self.lock = Lock()
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.total_tokens = 0
        self.logger = logger

    def add(
        self,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
    ):
        """Add token counts"""
        with self.lock:
            self.input_tokens += input_tokens
            self.cached_tokens += cached_tokens
            self.output_tokens += output_tokens
            self.reasoning_tokens += reasoning_tokens
            self.total_tokens += input_tokens + output_tokens

    def get_stats(self) -> dict:
        """Get current token stats"""
        with self.lock:
            return {
                "input_tokens": self.input_tokens,
                "cached_tokens": self.cached_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        """Reset all counters"""
        with self.lock:
            self.input_tokens = 0
            self.cached_tokens = 0
            self.output_tokens = 0
            self.reasoning_tokens = 0
            self.total_tokens = 0


# Type aliases
PreSendHandlerType = Callable[[str, str], tuple[str, str]]
ResultHandlerType = Callable[[str, str, logging.Logger], Any]
ErrorResultHandlerType = Callable[[str, logging.Logger], Any]


class Agent(LoggerMixin):
    """
    Base Agent class cho LLM interactions.
    
    Features:
    - Automatic retry with exponential backoff
    - Token tracking (separate cached vs billable)
    - Error handling (hard vs soft errors)
    - Concurrent/parallel processing
    - Request/response handlers
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        
        # Setup base URL
        self.baseurl = config.base_url.strip()
        if self.baseurl.endswith("/"):
            self.baseurl = self.baseurl[:-1]
        
        self.domain = urlparse(self.baseurl).netloc
        self.key = config.api_key.strip() if config.api_key else "xx"
        self.model_id = config.model_id.strip()
        self.system_prompt = ""
        self.temperature = config.temperature
        self.max_concurrent = config.concurrent
        self.timeout = httpx.Timeout(connect=5, read=config.timeout, write=300, pool=10)
        self.thinking = config.thinking
        self.retry = config.retry
        self.system_proxy_enable = config.system_proxy_enable
        
        # Use provided logger or default
        if config.logger:
            self.logger = config.logger
        
        # Counters
        self.total_error_counter = TotalErrorCounter(logger=self.logger)
        self.unresolved_error_lock = Lock()
        self.unresolved_error_count = 0
        self.token_counter = TokenCounter(logger=self.logger)

    def _prepare_request_data(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        json_format: bool = False
    ) -> tuple[dict, dict]:
        """Prepare request headers and data"""
        if temperature is None:
            temperature = self.temperature
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }
        
        data = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if json_format:
            data["response_format"] = {"type": "json_object"}
        
        return headers, data

    async def send_async(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        system_prompt: Optional[str] = None,
        retry: bool = True,
        retry_count: int = 0,
        force_json: bool = False,
        pre_send_handler: Optional[PreSendHandlerType] = None,
        result_handler: Optional[ResultHandlerType] = None,
        error_result_handler: Optional[ErrorResultHandlerType] = None,
        best_partial_result: Optional[dict] = None,
    ) -> Any:
        """Async send request to LLM"""
        if system_prompt is None:
            system_prompt = self.system_prompt
        
        if pre_send_handler:
            system_prompt, prompt = pre_send_handler(system_prompt, prompt)
        
        headers, data = self._prepare_request_data(prompt, system_prompt, json_format=force_json)
        
        should_retry = False
        is_hard_error = False
        current_partial_result = None

        try:
            response = await client.post(
                f"{self.baseurl}/chat/completions",
                json=data,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            response_data = response.json()
            result = response_data["choices"][0]["message"]["content"]
            
            # Extract and update token info
            input_tokens, cached_tokens, output_tokens, reasoning_tokens = extract_token_info(response_data)
            self.token_counter.add(input_tokens, cached_tokens, output_tokens, reasoning_tokens)
            
            if retry_count > 0:
                self.logger.info(f"Retry thành công (lần {retry_count}/{self.retry})")
            
            return result if result_handler is None else result_handler(result, prompt, self.logger)

        except AgentResultError as e:
            self.logger.error(f"AI response có lỗi: {e}")
            should_retry = True
        
        except PartialAgentResultError as e:
            self.logger.error(f"Nhận được partial result: {e}")
            current_partial_result = e.partial_result
            should_retry = True
            if e.append_prompt:
                prompt += e.append_prompt
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error (async): {e.response.status_code} - {e.response.text}")
            should_retry = True
            is_hard_error = True
        
        except httpx.RequestError as e:
            self.logger.error(f"Request error (async): {repr(e)}")
            should_retry = True
            is_hard_error = True
        
        except (KeyError, IndexError, ValueError) as e:
            self.logger.error(f"Response format error (async): {repr(e)}")
            should_retry = True
            is_hard_error = True

        # Update partial result
        if current_partial_result:
            best_partial_result = current_partial_result

        # Retry logic
        if should_retry and retry and retry_count < self.retry:
            if is_hard_error:
                if retry_count == 0:
                    if self.total_error_counter.add():
                        self.logger.error("Đạt giới hạn lỗi, không retry")
                        with self.unresolved_error_lock:
                            self.unresolved_error_count += 1
                        return best_partial_result if best_partial_result else (
                            prompt if error_result_handler is None else error_result_handler(prompt, self.logger)
                        )
                elif self.total_error_counter.reach_limit():
                    self.logger.error("Đạt giới hạn lỗi, không retry request này")
                    with self.unresolved_error_lock:
                        self.unresolved_error_count += 1
                    return best_partial_result if best_partial_result else (
                        prompt if error_result_handler is None else error_result_handler(prompt, self.logger)
                    )

            self.logger.info(f"Đang retry lần {retry_count + 1}/{self.retry}...")
            await asyncio.sleep(0.5)
            return await self.send_async(
                client, prompt, system_prompt, retry=True, retry_count=retry_count + 1,
                force_json=force_json, pre_send_handler=pre_send_handler,
                result_handler=result_handler, error_result_handler=error_result_handler,
                best_partial_result=best_partial_result
            )
        else:
            if should_retry:
                self.logger.error("Tất cả retry đã thất bại")
                with self.unresolved_error_lock:
                    self.unresolved_error_count += 1

            if best_partial_result:
                self.logger.info("Sử dụng partial result")
                return best_partial_result

            return prompt if error_result_handler is None else error_result_handler(prompt, self.logger)

    def send(
        self,
        client: httpx.Client,
        prompt: str,
        system_prompt: Optional[str] = None,
        retry: bool = True,
        retry_count: int = 0,
        force_json: bool = False,
        pre_send_handler: Optional[PreSendHandlerType] = None,
        result_handler: Optional[ResultHandlerType] = None,
        error_result_handler: Optional[ErrorResultHandlerType] = None,
        best_partial_result: Optional[dict] = None,
    ) -> Any:
        """Sync send request to LLM"""
        if system_prompt is None:
            system_prompt = self.system_prompt
        
        if pre_send_handler:
            system_prompt, prompt = pre_send_handler(system_prompt, prompt)

        headers, data = self._prepare_request_data(prompt, system_prompt, json_format=force_json)
        
        should_retry = False
        is_hard_error = False
        current_partial_result = None

        try:
            response = client.post(
                f"{self.baseurl}/chat/completions",
                json=data,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            response_data = response.json()
            result = response_data["choices"][0]["message"]["content"]

            # Extract and update token info
            input_tokens, cached_tokens, output_tokens, reasoning_tokens = extract_token_info(response_data)
            self.token_counter.add(input_tokens, cached_tokens, output_tokens, reasoning_tokens)

            if retry_count > 0:
                self.logger.info(f"Retry thành công (lần {retry_count}/{self.retry})")

            return result if result_handler is None else result_handler(result, prompt, self.logger)
        
        except AgentResultError as e:
            self.logger.error(f"AI response có lỗi: {e}")
            should_retry = True
        
        except PartialAgentResultError as e:
            self.logger.error(f"Nhận được partial result: {e}")
            current_partial_result = e.partial_result
            should_retry = True
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error (sync): {e.response.status_code} - {e.response.text}")
            should_retry = True
            is_hard_error = True
        
        except httpx.RequestError as e:
            self.logger.error(f"Request error (sync): {repr(e)}")
            should_retry = True
            is_hard_error = True
        
        except (KeyError, IndexError, ValueError) as e:
            self.logger.error(f"Response format error (sync): {repr(e)}")
            should_retry = True
            is_hard_error = True

        if current_partial_result:
            best_partial_result = current_partial_result

        # Retry logic (same as async)
        if should_retry and retry and retry_count < self.retry:
            if is_hard_error:
                if retry_count == 0:
                    if self.total_error_counter.add():
                        self.logger.error("Đạt giới hạn lỗi, không retry")
                        with self.unresolved_error_lock:
                            self.unresolved_error_count += 1
                        return best_partial_result if best_partial_result else (
                            prompt if error_result_handler is None else error_result_handler(prompt, self.logger)
                        )
                elif self.total_error_counter.reach_limit():
                    self.logger.error("Đạt giới hạn lỗi, không retry request này")
                    with self.unresolved_error_lock:
                        self.unresolved_error_count += 1
                    return best_partial_result if best_partial_result else (
                        prompt if error_result_handler is None else error_result_handler(prompt, self.logger)
                    )

            self.logger.info(f"Đang retry lần {retry_count + 1}/{self.retry}...")
            time.sleep(0.5)
            return self.send(
                client, prompt, system_prompt, retry=True, retry_count=retry_count + 1,
                force_json=force_json, pre_send_handler=pre_send_handler,
                result_handler=result_handler, error_result_handler=error_result_handler,
                best_partial_result=best_partial_result
            )
        else:
            if should_retry:
                self.logger.error("Tất cả retry đã thất bại")
                with self.unresolved_error_lock:
                    self.unresolved_error_count += 1

            if best_partial_result:
                self.logger.info("Sử dụng partial result")
                return best_partial_result

            return prompt if error_result_handler is None else error_result_handler(prompt, self.logger)

    async def send_prompts_async(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        force_json: bool = False,
        pre_send_handler: Optional[PreSendHandlerType] = None,
        result_handler: Optional[ResultHandlerType] = None,
        error_result_handler: Optional[ErrorResultHandlerType] = None,
    ) -> list[Any]:
        """Send multiple prompts concurrently (async)"""
        max_concurrent = self.max_concurrent if max_concurrent is None else max_concurrent
        total = len(prompts)
        
        self.logger.info(f"Async batch: {total} requests, concurrency: {max_concurrent}")
        self.total_error_counter.max_errors_count = len(prompts) // MAX_REQUESTS_PER_ERROR
        
        # Reset counters
        self.unresolved_error_count = 0
        self.token_counter.reset()
        
        count = 0
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        
        limits = httpx.Limits(
            max_connections=self.max_concurrent * 2,
            max_keepalive_connections=self.max_concurrent,
        )

        async with httpx.AsyncClient(trust_env=False, verify=False, limits=limits) as client:
            async def send_with_semaphore(p_text: str):
                async with semaphore:
                    result = await self.send_async(
                        client=client, prompt=p_text, system_prompt=system_prompt,
                        force_json=force_json, pre_send_handler=pre_send_handler,
                        result_handler=result_handler, error_result_handler=error_result_handler,
                    )
                    nonlocal count
                    count += 1
                    self.logger.info(f"Async - Hoàn thành {count}/{total}")
                    return result

            for p_text in prompts:
                task = asyncio.create_task(send_with_semaphore(p_text))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=False)

            # Log final stats
            self.logger.info(f"Async batch hoàn thành. Lỗi chưa giải quyết: {self.unresolved_error_count}")
            
            token_stats = self.token_counter.get_stats()
            if token_stats["input_tokens"] >= 0:
                self.logger.info(
                    f"Token - Input: {token_stats['input_tokens']/1000:.2f}K (cached: {token_stats['cached_tokens']/1000:.2f}K), "
                    f"Output: {token_stats['output_tokens']/1000:.2f}K (reasoning: {token_stats['reasoning_tokens']/1000:.2f}K), "
                    f"Total: {token_stats['total_tokens']/1000:.2f}K"
                )

            return results

    def send_prompts(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        json_format: bool = False,
        pre_send_handler: Optional[PreSendHandlerType] = None,
        result_handler: Optional[ResultHandlerType] = None,
        error_result_handler: Optional[ErrorResultHandlerType] = None,
    ) -> list[Any]:
        """Send multiple prompts using threading (sync)"""
        self.logger.info(f"Sync batch: {len(prompts)} requests, concurrency: {self.max_concurrent}")
        self.total_error_counter.max_errors_count = len(prompts) // MAX_REQUESTS_PER_ERROR
        
        # Reset counters
        self.unresolved_error_count = 0
        self.token_counter.reset()
        
        counter = PromptsCounter(len(prompts), self.logger)
        
        # Prepare iterators
        system_prompts = itertools.repeat(system_prompt, len(prompts))
        json_formats = itertools.repeat(json_format, len(prompts))
        counters = itertools.repeat(counter, len(prompts))
        pre_send_handlers = itertools.repeat(pre_send_handler, len(prompts))
        result_handlers = itertools.repeat(result_handler, len(prompts))
        error_result_handlers = itertools.repeat(error_result_handler, len(prompts))
        
        limits = httpx.Limits(
            max_connections=self.max_concurrent * 2,
            max_keepalive_connections=self.max_concurrent,
        )
        
        with httpx.Client(trust_env=False, verify=False, limits=limits) as client:
            clients = itertools.repeat(client, len(prompts))
            
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                results_iterator = executor.map(
                    self._send_prompt_count,
                    clients, prompts, system_prompts, json_formats, counters,
                    pre_send_handlers, result_handlers, error_result_handlers
                )
                output_list = list(results_iterator)

        # Log final stats
        self.logger.info(f"Sync batch hoàn thành. Lỗi chưa giải quyết: {self.unresolved_error_count}")
        
        token_stats = self.token_counter.get_stats()
        if token_stats["input_tokens"] >= 0:
            self.logger.info(
                f"Token - Input: {token_stats['input_tokens']/1000:.2f}K (cached: {token_stats['cached_tokens']/1000:.2f}K), "
                f"Output: {token_stats['output_tokens']/1000:.2f}K (reasoning: {token_stats['reasoning_tokens']/1000:.2f}K), "
                f"Total: {token_stats['total_tokens']/1000:.2f}K"
            )

        return output_list

    def _send_prompt_count(
        self,
        client: httpx.Client,
        prompt: str,
        system_prompt: Optional[str],
        force_json: bool,
        count: PromptsCounter,
        pre_send_handler: Optional[PreSendHandlerType],
        result_handler: Optional[ResultHandlerType],
        error_result_handler: Optional[ErrorResultHandlerType]
    ) -> Any:
        """Helper for threading - send and count"""
        result = self.send(
            client, prompt, system_prompt, force_json=force_json,
            pre_send_handler=pre_send_handler,
            result_handler=result_handler,
            error_result_handler=error_result_handler,
        )
        count.add()
        return result