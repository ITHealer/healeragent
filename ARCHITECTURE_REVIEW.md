# HealerAgent Architecture Review
## So sánh với Claude AI, ChatGPT, và Manus AI

**Ngày review:** 2025-12-27
**Reviewer:** Architecture Analysis Agent

---

## 1. TỔNG QUAN KIẾN TRÚC HEALERAGENT

### 1.1 Các thành phần chính

```
┌─────────────────────────────────────────────────────────────────┐
│                     HEALERAGENT ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Planning   │───▶│    Task      │───▶│  Validation  │     │
│  │    Agent     │    │   Executor   │    │    Agent     │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                   │              │
│         │                   │                   │              │
│  ┌──────▼──────────────────▼───────────────────▼──────┐       │
│  │                    TOOL REGISTRY                    │       │
│  │              (31+ Atomic Financial Tools)           │       │
│  └────────────────────────────────────────────────────┘       │
│                            │                                   │
│  ┌─────────────────────────▼──────────────────────────┐       │
│  │                   MEMORY SYSTEM                     │       │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────┐  │       │
│  │  │  Working   │ │ Short-term │ │   Long-term    │  │       │
│  │  │  Memory    │ │   Memory   │ │    Memory      │  │       │
│  │  └────────────┘ └────────────┘ └────────────────┘  │       │
│  └────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 7-Phase Pipeline

```
Phase 0: Initialize → Phase 1: Load Context → Phase 2: Planning
    ↓
Phase 3: Execute → Phase 4: Validation → Phase 5: Context Assembly
    ↓
Phase 6: Response Generation → Phase 7: Cleanup & Persistence
```

---

## 2. SO SÁNH VỚI CÁC HỆ THỐNG HIỆN ĐẠI

### 2.1 So sánh với Claude AI (Extended Thinking)

#### Claude's Approach:
```
User Query → [THINKING Block] → Tool 1 → [THINKING Block] → Tool 2 → Response
            "User wants technical    "Price dropped 3%,
             analysis, need price     let me check RSI
             first, then RSI..."      if it's oversold..."
```

#### HealerAgent's Approach:
```
User Query → Classification → Plan All Tools → Execute All → Response
             (1 LLM call)     (1 LLM call)     (No thinking)
```

#### ❌ **GAP 1: Thiếu Interleaved Thinking (Chain-of-Thought)**

**Vấn đề:** HealerAgent plan tất cả tools trước, không có khả năng "suy nghĩ" giữa các tool calls.

**Claude làm gì:**
- Có thinking blocks giữa mỗi tool call
- Có thể thay đổi quyết định dựa trên kết quả tool trước
- Reasoning visible và traceable

**HealerAgent thiếu:**
- `thinkTool` có nhưng không được integrate vào agent loop
- Không có cơ chế re-planning sau khi có tool result
- Planning Agent quyết định tất cả upfront

**Đề xuất:**
```python
# Thêm ReAct-style loop
class ReactiveTaskExecutor:
    async def execute_with_thinking(self, task):
        for step in task.steps:
            # Think before each action
            thought = await self.think(context, step)

            # Execute action
            result = await self.execute_tool(step.tool)

            # Think after action - may replan
            should_continue, new_plan = await self.reflect(thought, result)

            if not should_continue:
                return self.synthesize(results)
```

---

### 2.2 So sánh với ChatGPT (Memory System)

#### ChatGPT Memory Architecture:
```
┌─────────────────────────────────────────────────────────────┐
│                   ChatGPT Memory System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Session Metadata│  │   User Memory   │                   │
│  │   (Ephemeral)   │  │   (Long-term)   │                   │
│  │ Device, location│  │  33 stored facts│                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Chat Summaries │  │    Current      │                   │
│  │  (Lightweight)  │  │  Conversation   │                   │
│  │ ~15 past chats  │  │ (Sliding Window)│                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### HealerAgent Memory Architecture:
```
┌─────────────────────────────────────────────────────────────┐
│                   HealerAgent Memory System                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Working Memory  │  │  Core Memory    │                   │
│  │  (Per Request)  │  │  (Long-term)    │                   │
│  │  Symbols, Plan  │  │  User Profile   │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Short-term Mem  │  │   Long-term     │                   │
│  │ (Vector DB)     │  │   Memory        │                   │
│  │ Last 10 turns   │  │ (Vector DB)     │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                              │
│         ❌ MISSING: Chat Summaries Layer                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### ❌ **GAP 2: Thiếu Chat Summaries Layer**

**Vấn đề:** HealerAgent chỉ giữ 10 turns gần nhất, không có summaries của các cuộc hội thoại cũ.

**ChatGPT làm gì:**
- Tạo lightweight summaries (~15 past conversations)
- Mỗi summary chứa: key topics, outcomes, user preferences learned
- Cho phép recall context từ nhiều tuần trước

**HealerAgent thiếu:**
- Không có `ConversationSummary` model
- Không có periodic summarization job
- Long-term memory chỉ lưu "important" conversations (score >= 0.7)

**Đề xuất:**
```python
class ConversationSummarizer:
    """Periodically summarize conversations for long-term recall"""

    async def summarize_session(self, session_id: str) -> Summary:
        """Create summary when session ends or periodically"""
        messages = await self.get_session_messages(session_id)

        summary = await self.llm.generate(
            prompt=self.SUMMARY_PROMPT,
            messages=messages
        )

        return Summary(
            session_id=session_id,
            key_topics=summary.topics,
            symbols_discussed=summary.symbols,
            user_preferences_learned=summary.preferences,
            outcomes=summary.outcomes,
            created_at=datetime.now()
        )
```

---

### 2.3 So sánh với ChatGPT (Context Compaction)

#### ChatGPT's Auto-Compaction:
```
Context: 70% full
    ↓
New message → Context: 85% full
    ↓
Another message → Context: 95% full ⚠️
    ↓
TRIGGER AUTO-COMPACT
    ↓
┌──────────────────────────────────────────┐
│         COMPACTION PROCESS:              │
│ 1. Stop current work (~1s)               │
│ 2. Analyze conversation (~2-5s)          │
│ 3. Generate summary (~3-10s)             │
│ 4. Replace & continue (~1s)              │
│                                          │
│ Total interruption: 7-17 seconds         │
└──────────────────────────────────────────┘
    ↓
Context reduced to 30%
    ↓
Continue with compressed context
```

#### ❌ **GAP 3: Thiếu Auto Context Compaction**

**Vấn đề:** HealerAgent có `max_tokens` limit nhưng chỉ có eviction (xóa), không có compaction (nén).

**HealerAgent hiện tại:**
```python
# working_memory.py:188
while self._total_tokens + token_count > self.max_tokens:
    if not self._evict_lowest_priority():  # ← Chỉ XÓA, không NÉN
        return ""
```

**Cần thêm:**
```python
class ContextCompactor:
    """Compress context when approaching token limit"""

    COMPACTION_THRESHOLD = 0.85  # Trigger at 85%
    TARGET_RATIO = 0.30  # Reduce to 30%

    async def check_and_compact(self, working_memory: WorkingMemory):
        usage_ratio = working_memory._total_tokens / working_memory.max_tokens

        if usage_ratio > self.COMPACTION_THRESHOLD:
            # 1. Get all entries
            entries = working_memory.get_all_entries()

            # 2. Generate summary
            summary = await self.summarize_context(entries)

            # 3. Clear old entries
            working_memory.clear_all()

            # 4. Add compressed summary
            working_memory.add_intermediate(
                label="context_summary",
                data=summary,
                priority=Priority.CRITICAL
            )
```

---

### 2.4 So sánh với Manus AI / Galileo (Tool Selection Scoring)

#### Galileo's Tool Selection Evaluation:
```
┌─────────────────────────────────────────────────────────────┐
│              TOOL SELECTION SCORING                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Input → LLM Planner → Develops Plans → Tools A/B/C   │
│                                    │                        │
│                                    ↓                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │            EVALUATION CRITERIA:                     │    │
│  │  • Were correct tools selected?                     │    │
│  │  • Were arguments correct?                          │    │
│  │  • Did tool encounter errors?                       │    │
│  │  • Did selection meet user goals?                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Tool response sent to planner → Repeat until final action │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### ❌ **GAP 4: Thiếu Tool Selection Scoring/Evaluation**

**Vấn đề:** HealerAgent có `ValidationAgent` nhưng chỉ validate output, không score tool selection.

**HealerAgent hiện tại:**
```python
# validation_agent.py - Chỉ validate OUTPUT
class ValidationAgent:
    def validate_output(self, output: Dict) -> ValidationResult:
        # Check schema, types, ranges
        pass
```

**Thiếu:**
```python
class ToolSelectionEvaluator:
    """Evaluate if correct tools were selected"""

    async def evaluate_selection(
        self,
        query: str,
        selected_tools: List[ToolCall],
        available_tools: List[Tool],
        execution_results: Dict
    ) -> ToolSelectionScore:

        return ToolSelectionScore(
            # Did we select the right tools?
            tool_accuracy=self._calculate_tool_accuracy(query, selected_tools),

            # Were parameters correct?
            param_accuracy=self._calculate_param_accuracy(selected_tools, results),

            # Did tools achieve user goal?
            goal_achievement=self._calculate_goal_achievement(query, results),

            # Were there avoidable errors?
            error_rate=self._calculate_error_rate(results),

            # Feedback for improvement
            suggestions=self._generate_improvement_suggestions()
        )
```

---

### 2.5 So sánh với Memory Distiller Pattern

#### Memory Distiller Flow (Best Practice):
```
Conversation Complete
        ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Memory Distiller│───▶│  LLM Analyzer   │───▶│  Memory Store   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        │              Decision Process:              │
        │              1. Is this worth              │
        │                 remembering?               │
        │              2. Is it a stable fact?       │
        │              3. Is it user-specific?       │
        │                      │                      │
        │                      ↓                      │
        │              [Worth Remembering]            │
        │                      │                      │
        │              Create Observation Object      │
        │                      │                      │
        │                      ↓                      │
        │              Store Memory                   │
        │              Format:                        │
        │              - User's name is X             │
        │              - Works on project Y           │
        │              - Prefers Z approach           │
        │                      │                      │
        │              [Not Worth Saving]             │
        │                      │                      │
        │              Discard (chit-chat)            │
        └──────────────────────────────────────────────┘
```

#### ✅ **HealerAgent có implement pattern này**

**MemoryUpdateAgent** (`memory_update_agent.py`) thực hiện:
1. Extract user info từ conversation
2. Check if worth remembering (confidence threshold)
3. Consolidate với existing memories
4. Store với appropriate action (ADD/UPDATE/DELETE/MERGE/NOOP)

**Điểm tốt:**
- Có confidence scoring (reject < 0.6)
- Có consolidation để tránh duplicate
- Có relevance decay cleanup

**Điểm cần cải thiện:**
- Extraction prompt quá dài (~100 lines) - có thể split
- Không có periodic background distillation
- Chỉ chạy khi có explicit user info, không passive observe

---

## 3. ĐÁNH GIÁ IMPLEMENTATION QUALITY

### 3.1 Planning Agent (planning_agent.py)

#### ✅ Điểm tốt:
- 3-stage flow rõ ràng (Classify → Load Tools → Create Plan)
- Progressive disclosure (chỉ load 7-15 tools thay vì 31)
- Model capability detection (Basic/Intermediate/Advanced)
- Working memory integration cho symbol continuity
- JSON recovery logic cho truncated responses

#### ❌ Vấn đề:
1. **Prompt quá dài** - `_stage3_create_plan` prompt ~200 lines
2. **Hardcoded examples** trong prompt - khó maintain
3. **Override logic phức tạp** - nhiều if/else để fix edge cases

```python
# Ví dụ override phức tạp (lines 178-204)
if query_type == "conversational":
    if symbols:
        should_override = True
        query_type = "stock_specific"
    elif any(cat in STRONG_FINANCIAL_CATEGORIES for cat in categories):
        should_override = True
        if "discovery" in categories:
            query_type = "screener"
        else:
            query_type = "stock_specific"
```

**Recommendation:** Tách prompt templates ra file riêng, dùng Jinja2 hoặc similar.

---

### 3.2 Task Executor (task_executor.py)

#### ✅ Điểm tốt:
- Auto-detect dependencies từ `<FROM_TASK_N>` placeholders
- Symbol expansion cho multi-symbol queries
- Graceful skip khi dependency trả về 0 symbols (v2.2 fix)
- Exponential backoff retry
- Result aggregation cho batch results

#### ❌ Vấn đề:
1. **Không có circuit breaker** - nếu external API down, vẫn retry
2. **Không có rate limiting** - có thể overload API
3. **MAX_SYMBOLS_PER_TASK = 5** - hardcoded, không configurable per tool

```python
# Cần thêm
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.last_failure = None

    def allow_request(self) -> bool:
        if self.failures >= self.failure_threshold:
            if time.time() - self.last_failure < self.reset_timeout:
                return False
        return True
```

---

### 3.3 Memory System

#### ✅ Điểm tốt:
- Dual memory (short-term + long-term) với vector search
- Working memory với token management
- Symbol continuity across turns (5-turn TTL)
- Memory consolidation (ADD/UPDATE/DELETE/MERGE/NOOP)
- Thread-safe operations

#### ❌ Vấn đề:
1. **search_archival_memory có bug** - dùng `self.memory_manager` nhưng class là `MemoryManager`

```python
# Bug ở line 531-535
async def search_archival_memory(...):
    memories = await self.memory_manager.search_relevant_memory(...)  # ← BUG!
    # Đây là method của MemoryManager, không thể gọi self.memory_manager
```

2. **Không có memory size limit** - short-term có thể grow indefinitely
3. **Cleanup chỉ manual** - không có background job

---

### 3.4 Tool System

#### ✅ Điểm tốt:
- BaseTool với schema validation
- ToolRegistry với singleton pattern
- Parallel execution support
- Comprehensive financial tool coverage (31+ tools)

#### ❌ Vấn đề:
1. **Không có tool versioning** - khó rollback nếu có breaking change
2. **Không có tool health check** - không biết tool nào đang available
3. **Không có usage analytics** - không track tool performance

---

## 4. SUMMARY: NHỮNG GÌ ĐANG THIẾU

### 4.1 Critical Gaps (Cần fix ngay)

| Gap | Mô tả | Priority |
|-----|-------|----------|
| **Interleaved Thinking** | Không có thinking giữa tool calls | HIGH |
| **Context Compaction** | Chỉ evict, không compress | HIGH |
| **Chat Summaries** | Không có summary layer | MEDIUM |
| **Bug in search_archival_memory** | Self-reference bug | HIGH |

### 4.2 Nice-to-Have Improvements

| Improvement | Benefit |
|-------------|---------|
| Tool Selection Scoring | Đánh giá và cải thiện tool selection |
| Circuit Breaker | Prevent cascade failures |
| Background Memory Distillation | Passive learning từ conversations |
| Prompt Templates | Easier maintenance |
| Tool Health Monitoring | Proactive issue detection |

---

## 5. RECOMMENDED ARCHITECTURE IMPROVEMENTS

### 5.1 Add ReAct Loop for Thinking

```python
class ReActExecutor:
    """
    ReAct (Reasoning + Acting) pattern implementation

    Loop: Think → Act → Observe → Think → Act → ...
    """

    async def execute(self, query: str, plan: TaskPlan):
        context = []

        for task in plan.tasks:
            # THINK: Reason about next action
            thought = await self.think(query, context, task)

            # ACT: Execute tool
            observation = await self.act(task, thought)

            # OBSERVE: Process result
            context.append({
                "thought": thought,
                "action": task.description,
                "observation": observation
            })

            # Check if need to replan
            if self.should_replan(observation):
                new_plan = await self.replan(query, context)
                return await self.execute(query, new_plan)

        return await self.synthesize(context)
```

### 5.2 Add Context Compaction

```python
class ContextCompactionManager:
    """
    Auto-compact context when approaching limits
    """

    THRESHOLD = 0.85
    TARGET = 0.30

    async def maybe_compact(self, working_memory: WorkingMemory):
        if self.should_compact(working_memory):
            summary = await self.generate_summary(working_memory)
            working_memory.replace_with_summary(summary)
```

### 5.3 Add Chat Summaries Layer

```python
class ChatSummaryManager:
    """
    Maintain lightweight summaries of past conversations
    """

    MAX_SUMMARIES = 15

    async def create_summary(self, session_id: str):
        messages = await self.get_messages(session_id)
        summary = await self.llm.summarize(messages)
        await self.store_summary(session_id, summary)
        await self.prune_old_summaries()
```

---

## 6. CONCLUSION

### Điểm mạnh của HealerAgent:
1. ✅ 7-phase pipeline rõ ràng, có structure
2. ✅ Dual memory system với vector search
3. ✅ Memory consolidation thông minh
4. ✅ Tool registry với progressive disclosure
5. ✅ Symbol continuity across turns

### Điểm cần cải thiện:
1. ❌ Thiếu interleaved thinking (critical)
2. ❌ Thiếu context compaction (critical)
3. ❌ Thiếu chat summaries layer
4. ❌ Thiếu tool selection evaluation
5. ❌ Có bug trong search_archival_memory

### So với best practices:
- **vs Claude:** Thiếu thinking blocks giữa tool calls
- **vs ChatGPT:** Thiếu chat summaries và context compaction
- **vs Manus AI:** Thiếu tool selection scoring

**Overall Assessment:** HealerAgent có foundation tốt nhưng cần bổ sung các features quan trọng để đạt production-grade như các hệ thống AI hàng đầu.

---

## 7. NEXT STEPS

1. **Immediate:** Fix bug `search_archival_memory`
2. **Short-term:** Implement context compaction
3. **Medium-term:** Add ReAct-style thinking loop
4. **Long-term:** Add chat summaries và tool selection scoring
