# CMPSC 441 Final Project - AI Dungeon Master

## Overview

This project implements an AI-powered Dungeons & Dragons Dungeon Master (DM) system. Players interact through a command-line interface where a large language model acts as the DM, narrating a procedurally generated adventure, resolving combat mechanics, roleplaying NPCs and enemies, and maintaining a persistent AI companion (teammate). The system is built entirely with local LLMs via Ollama and integrates multiple AI methods including RAG, structured output generation, multi-agent orchestration, tool calling via MCP, and conversation summarization.

---

## Section 1 - Base System Functionality

The following scenarios are handled by the system. Each is described in detail and linked to the AI methods applied in later sections.

### Scenario 1: Character Creation
The player provides their name and selects a class (fighter, wizard, rogue, paladin, ranger, cleric, barbarian, bard, druid, monk, sorcerer, or warlock). The system rolls stats using the 4d6-drop-lowest method, derives HP and MP from class and ability scores, and assigns class-appropriate skill lists. A random AI companion (teammate) is also created with their own full character sheet, class, and skills.

### Scenario 2: Dynamic World Narration
The DM agent generates an immersive opening scene and responds to all player actions with narrative prose. It describes the environment, introduces story elements, and reacts appropriately to player choices - without ever deciding what the player does next. The DM maintains full conversation history and context throughout the session.

### Scenario 3: Procedural Dungeon Room Generation
Using the `/room <theme>` command, the player can trigger generation of a new dungeon room. The LLM produces a structured room object (name, description, enemies, items, exits, and danger level) that the DM then narrates dramatically. Room data is validated via a Pydantic schema to ensure it always contains the expected fields.

### Scenario 4: Turn-Based Combat
When combat begins, the system resolves attack rolls, damage calculation, and HP tracking automatically using MCP-hosted tools. The DM handles the player's attack turn, then delegates to the enemy sub-agent for the enemy's counterattack. HP totals are displayed after every exchange. Combat ends when either combatant reaches 0 HP.

### Scenario 5: Enemy Agent (Sub-Agent Delegation)
When the DM embeds an `[ENEMY_TURN: ...]` tag in its response, a dedicated enemy sub-agent takes over. The enemy agent follows a strict multi-step sequence: roll to attack, roll for damage, call `calculate_damage`, update the player's HP if the attack lands, and finally narrate its action in character. The enemy's personality and stats are drawn from a generated `EnemySheet`.

### Scenario 6: NPC Interaction
When the DM embeds an `[NPC_TURN: ...]` tag, an NPC sub-agent responds in character. The NPC's name, role, personality, knowledge, and greeting are defined in a generated `NPCSheet`. The NPC speaks only from its own perspective and knowledge - it does not resolve mechanics or break character.

### Scenario 7: AI Companion (Teammate Agent)
A randomly generated AI companion character accompanies the player throughout the adventure. The teammate reacts to DM narration, engages in dialogue with the player, and can carry on multi-turn side conversations. The companion is intentionally restricted to roleplay only - it has no tool access and always defers mechanical decisions to the DM.

### Scenario 8: Conversation Summarization
When the DM conversation history or teammate history exceeds 20 messages, the system automatically summarizes all prior exchanges into a single compressed entry. This preserves critical facts (HP values, items, decisions, locations, characters met) while preventing context window overflow across long sessions.

### Scenario 9: Rule-Aware Action Resolution
Every player action is sent to the DM along with a RAG-retrieved snippet of D&D 5e rules relevant to that action. This means that when the player attacks, casts a spell, or uses a skill, the DM has authoritative rule context injected into its prompt, leading to mechanically consistent rulings.

### Scenario 10: Skills Reference
The `/skills` command displays the player's and teammate's current skill lists at any time during the session, without interrupting the flow of the game.

---

## Section 2 - Prompt Engineering and Model Parameter Choice

### Model and Temperature Settings

The system uses `llama3.1:8b` via Ollama for all LLM calls. Three separate `ChatOllama` instances are created with different temperatures tuned to each agent's role:

| Agent | Temperature | Rationale |
|---|---|---|
| `llm_dm` (Dungeon Master) | `0.75` | Balanced creativity and consistency. High enough to produce vivid, varied narrative; low enough to follow structural rules reliably. |
| `llm_sub` (Enemy / NPC sub-agents) | `0.85` | Slightly higher for more expressive, character-driven responses from enemies and NPCs. |
| `llm_teammate` (Companion) | `0.90` | Highest temperature for the most spontaneous and human-like companion dialogue. |
| Room generation (`ollama_chat`) | `0.9` | High creativity encouraged for procedural world-building where variety is desirable. |
| Summarization (`ollama_chat`) | `0.2` | Very low - factual accuracy and compression are critical; creativity is unwanted here. |

### Prompt Design

**DM System Prompt (`DM_SYSTEM_PROMPT`):**
The DM prompt is the most detailed in the system. Key design decisions include:
- Explicit "player agency" rules that prohibit the DM from narrating the player's thoughts, feelings, or decisions. This prevents the LLM from overstepping and taking control of the player's character.
- Delegation syntax (`[ENEMY_TURN: ...]`, `[NPC_TURN: ...]`) is defined precisely so the DM knows how and when to hand off to sub-agents.
- Tool usage order for combat is spelled out step-by-step to enforce correct sequencing (roll → damage → update HP → narrate).
- The DM is explicitly forbidden from outputting raw JSON or tool call syntax in its visible response.

**Enemy Agent Prompt (`ENEMY_AGENT_PROMPT`):**
The enemy prompt enforces a numbered multi-step sequence for every attack turn. Steps are labeled and the model is told to note integer results from each `roll_dice` call before proceeding to `calculate_damage`. Passing `None` to any tool argument is explicitly prohibited. This addresses the core failure mode where smaller models skip prerequisite steps and call tools with missing arguments.

**NPC Agent Prompt (`NPC_AGENT_PROMPT`):**
Kept minimal - the NPC should stay in character, speak from its own limited knowledge, and not touch mechanics. Brevity in the prompt reflects brevity expected in the output.

**Teammate Prompt (`make_teammate_prompt`):**
Uses the player's actual name (injected at runtime) so the companion always addresses the player correctly. Contains explicit negative examples of wrong behavior (narrating the player's thoughts) paired with correct alternatives. Tool use is absolutely prohibited through both prompt instruction and by passing an empty tools list to the agent.

**Player Action Context Prompt:**
Every player action submitted to the DM is wrapped in a structured message containing:
1. Any prior side conversation between the player and teammate
2. RAG-retrieved D&D rules relevant to the action
3. A live HP snapshot for all parties
4. The raw player action text
5. Closing instructions reminding the DM not to narrate the player's next move

This layered context construction is the primary mechanism by which the system achieves rule-consistent narration.

---

## Section 3 - Tools Usage

The system integrates tool calling via the **Model Context Protocol (MCP)**, using `fastmcp` to define a local stdio MCP server that exposes four game tools. These are loaded into the DM and enemy agents via `langchain_mcp_adapters`.

### Tools Defined

**`roll_dice(n_dice, sides, modifier)`**
Simulates any dice roll in the D&D system. Used for attack rolls (1d20 + modifier), damage rolls (variable dice by weapon), and initiative. Returns a human-readable string with the individual rolls and total.

**`calculate_damage(base_damage, armor_class, attack_roll)`**
Compares the attack roll against the target's armor class and returns whether the attack hit, and if so, how much damage was dealt. Uses `Optional[int]` for all parameters with a guard clause that returns a descriptive error if any argument is `None`, preventing crashes when the LLM skips a prerequisite step.

**`get_character_stat(entity, stat)`**
Allows agents to query the game state for any stat on `player`, `teammate`, `active_enemy`, or `active_npc`. Returns a prompt directing the agent to check the provided game state context (the actual values are passed in the agent's system/user messages).

**`update_stats(entity, field, new_value)`**
Writes a new integer value to a stat in the shared JSON state file. Used after every successful attack to keep HP accurate across agents. Reads and writes a temp file (`DND_STATE_FILE`) that is also polled by the main loop after each DM turn, so changes made inside the MCP server propagate back to the Python game state.

### State Synchronization

Because the MCP server runs as a subprocess with its own memory, game state is shared via a temporary JSON file. After each DM agent invocation, the main loop re-reads this file and merges any changes back into the in-process `game_state` dict. This two-way sync ensures HP changes made by the enemy agent are visible to the DM on its next turn.

### Integration Libraries

- `mcp` - MCP client/server protocol
- `fastmcp` - high-level MCP server definition
- `langchain_mcp_adapters` - bridges MCP tools into LangChain/LangGraph tool format
- `langgraph.prebuilt.create_react_agent` - ReAct agent loop that handles tool calls iteratively

---

## Section 4 - Planning & Reasoning

### Multi-Agent Orchestration

The system implements a **hierarchical multi-agent architecture** with three distinct reasoning agents:

1. **DM Agent** - the top-level planner. It reads the full game state, interprets the player's action, selects appropriate tools to call, and decides whether to delegate to a sub-agent.
2. **Enemy Sub-Agent** - a specialized short-lived agent invoked only during enemy turns. It reasons through a fixed attack sequence using ReAct-style tool chaining: roll attack → roll damage → calculate hit → update HP → narrate.
3. **Teammate Agent** - a persistent companion agent that maintains its own conversation history and reasons about how to react to DM narration and player dialogue independently of the DM.

### ReAct Tool-Chaining (Chain-of-Thought via Tool Sequencing)

The DM and enemy agents use `create_react_agent` from LangGraph, which implements the **ReAct (Reasoning + Acting)** pattern. The agent alternates between reasoning steps and tool calls, receiving tool results before deciding what to do next. For example, during combat:

```
Thought: I need to roll to attack.
Action: roll_dice(n_dice=1, sides=20, modifier=3)
Observation: Rolled 1d20+3: [14] = 17
Thought: 17 vs AC 13 - this hits. Now I need damage.
Action: roll_dice(n_dice=1, sides=8, modifier=2)
Observation: Rolled 1d8+2: [5] = 7
Thought: 7 damage dealt. Update the enemy HP.
Action: update_stats(entity="active_enemy", field="hp", new_value=8)
...
```

This multi-step reasoning chain is enforced by both the agent framework and the explicit step-by-step instructions in each agent's system prompt.

### Delegation Tag Parsing

The DM uses a structured delegation pattern - embedding `[ENEMY_TURN: ...]` or `[NPC_TURN: ...]` tags when it needs a sub-agent to act. The `resolve_delegation_tags` function intercepts these tags using regex, constructs a context-rich prompt for the sub-agent (including the relevant character sheet and situation), invokes the sub-agent asynchronously, and splices the result back into the DM's output before it reaches the player. This gives the system a lightweight form of task decomposition and planning.

### Conversation Coherence via Summarization

Long-running sessions would eventually overflow the model's context window. The `summarize_history` function addresses this by compressing conversation history into a single structured summary when it exceeds `SUMMARY_THRESHOLD` (20 messages). The summarizer is prompted at `temperature=0.2` to be factual and preserve game-critical details (HP, inventory, decisions, locations). Both the DM and teammate maintain their own independent summarization cycles.

---

## Section 5 - RAG Implementation

### Vector Store

At startup, `build_vector_store()` constructs an in-memory **ChromaDB** collection using cosine similarity as the distance metric. Eighteen curated D&D 5e rules facts are embedded using the `nomic-embed-text` model via Ollama and stored as document vectors.

### Retrieval

On every player action, `retrieve_context(collection, user_input, n=3)` embeds the player's input using the same `nomic-embed-text` model and retrieves the three most semantically similar rules from the knowledge base. These are injected into the DM's prompt under a `[DnD rules context]` header before the action is resolved.

### Why This Matters Per Scenario

| Scenario | RAG Benefit |
|---|---|
| Combat (attack rolls) | Retrieves AC, attack roll, and critical hit rules so the DM applies them correctly |
| Spellcasting | Retrieves spell slot and concentration rules relevant to the spell used |
| Skill checks | Retrieves saving throw and ability check rules to guide resolution |
| Exploration | Retrieves passive perception and rest rules when relevant |

### Design Choice: Embedded Rules vs. External Retrieval

Rather than fetching rules from an external D&D API or document corpus, the knowledge base is a curated, hand-selected set of 18 foundational rules. This keeps the system fully offline and fast, while covering the most common game mechanics a player will invoke. The semantic search ensures the most relevant rules surface regardless of how the player phrases their action.

---

## Section 6 - Additional Tools / Innovation

### Structured Output Generation with Pydantic

A key innovation in this system is the use of **Pydantic models as JSON schemas** passed directly to the Ollama API's `format` parameter. This forces the LLM to produce output that strictly conforms to a defined schema, enabling reliable parsing without regex heuristics.

Three schemas are used:

- **`DungeonRoom`** - used by `generate_room()` to produce validated room data (name, description, enemies, items, exits, danger level). Danger level is constrained to a `Literal` enum.
- **`EnemySheet`** - used by `create_enemy()` to generate a complete, stats-accurate monster entry.
- **`NPCSheet`** - used by `create_npc()` to produce a character with a defined personality, knowledge scope, and greeting line.

This is called with `format=ModelClass.model_json_schema()` and validated with `ModelClass.model_validate_json(resp.message.content)`, making the generation pipeline type-safe and crash-resistant.

### Response Sanitization Pipeline

The `sanitize()` function is a creative defensive layer that strips leaked internal artifacts (raw JSON blobs, bracket templates, tool call headers) from any agent output before it reaches the player. It uses:
- Regex substitution for known patterns (`_JSON_BLOB`, `_BRACKET_TMPL`, `_TOOL_CALL_HDR`)
- Line-by-line brace depth tracking to strip multi-line JSON objects
- Collapsing of excess blank lines

This was essential for a smooth user experience with a small local model that occasionally leaks tool call syntax into its narrative output.

### Teammate Address Detection

The `player_is_addressing_teammate()` function uses keyword and name matching to detect when the player is speaking to their companion rather than taking a world action. When detected, the input is routed to `teammate.converse()` instead of the DM, enabling natural multi-turn side dialogues between the player and their AI companion without confusing the DM agent.

---

## Section 7 - Code Quality & Modular Design

### Module Structure

The codebase is organized into clearly separated, single-responsibility sections, each marked with a header comment:

| Section | Responsibility |
|---|---|
| Pydantic Models | Data schemas for all game entities |
| Global Game State | Shared mutable state dictionary |
| RAG Knowledge Base | Embedding data and retrieval constants |
| RAG Helpers | `get_embedding`, `build_vector_store`, `retrieve_context` |
| Structured Output Helpers | `generate_room` |
| Character / Enemy / NPC Creation | Stat rolling, HP/MP calculation, LLM-driven creation |
| Conversation Summarization | `summarize_history` |
| Agent Prompts | All system prompts as named constants |
| Response Sanitizer | `sanitize()` |
| Sub-Agent Runner | `run_sub_agent()` |
| TeammateAgent class | Encapsulated companion logic with history management |
| Character Creation Flow | Interactive CLI onboarding |
| Delegation Tag Parser | `resolve_delegation_tags()` |
| Teammate Address Detector | `player_is_addressing_teammate()` |
| Main Agent Loop | `run_agent()` - top-level game loop |
| Entry Point | `build_vector_store` + `asyncio.run` |

### Key Design Practices

- **Separation of concerns** - Each agent (DM, enemy, NPC, teammate) has its own prompt, its own `ChatOllama` instance with its own temperature, and its own invocation path. No agent shares state directly with another; all coordination goes through `game_state` and the MCP state file.
- **Pydantic everywhere** - All structured data produced by the LLM is validated through Pydantic models before use, preventing malformed game state from propagating.
- **Async throughout** - The entire agent loop and all sub-agent calls are `async`, enabling non-blocking I/O for the MCP stdio transport.
- **Defensive error handling** - Tool functions return descriptive error strings rather than raising exceptions where possible. The `calculate_damage` guard against `None` arguments is an example of this defensive posture.
- **Named constants** - Prompts (`DM_SYSTEM_PROMPT`, `ENEMY_AGENT_PROMPT`, etc.), model names (`MODEL`, `EMBED_MODEL`), and thresholds (`SUMMARY_THRESHOLD`) are all defined as module-level constants, not inline strings.

### Environment & Dependencies

The project uses a Python virtual environment (`.venv`) and relies on the following key packages:
- `ollama` - local LLM inference and embeddings
- `chromadb` - in-memory vector store for RAG
- `langchain-ollama`, `langgraph` - ReAct agent framework
- `langchain-mcp-adapters`, `mcp`, `fastmcp` - MCP tool protocol
- `pydantic` - data validation and structured output schemas
- `asyncio` - async orchestration of all agents