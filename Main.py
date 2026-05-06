import asyncio
import chromadb
import random
import sys
import re
import warnings
import json
from ollama import embeddings, chat as ollama_chat
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL      = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

SUMMARY_THRESHOLD = 20


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DungeonRoom(BaseModel):
    room_name: str
    description: str
    enemies: list[str]
    items: list[str]
    exits: list[Literal["north", "south", "east", "west", "up", "down"]]
    danger_level: Literal["safe", "low", "medium", "high", "deadly"]

class CharacterSheet(BaseModel):
    name: str
    character_class: str
    hp: int
    mp: int
    strength: int
    dexterity: int
    constitution: int
    intelligence: int
    wisdom: int
    charisma: int
    skills: list[str]

class EnemySheet(BaseModel):
    name: str
    enemy_type: str
    hp: int
    armor_class: int
    attack_bonus: int
    damage_dice: str
    description: str
    personality: str

class NPCSheet(BaseModel):
    name: str
    role: str
    personality: str
    knowledge: str
    greeting: str


# ---------------------------------------------------------------------------
# Global game state
# ---------------------------------------------------------------------------

game_state: dict = {
    "player":         None,
    "teammate":       None,
    "active_enemy":   None,
    "active_npc":     None,
    "player_skills":  [],
    "teammate_skills": [],
}


# ---------------------------------------------------------------------------
# RAG Knowledge Base
# ---------------------------------------------------------------------------

DND_KNOWLEDGE = [
    "Dungeons and Dragons uses a d20 system where players roll a 20-sided die for most checks. Higher rolls are better, and a natural 20 is a critical success.",
    "Armor Class (AC) represents how hard a creature is to hit. Attack rolls must meet or exceed the target's AC to deal damage.",
    "Hit Points (HP) represent a character's health. When HP reaches 0 the character falls unconscious and begins making death saving throws.",
    "Spell slots are the resource used to cast spells. A 1st-level wizard has two 1st-level spell slots. Higher-level spells require higher-level slots.",
    "Advantage means rolling two d20s and taking the higher result. Disadvantage means rolling two d20s and taking the lower result.",
    "Ability scores are Strength, Dexterity, Constitution, Intelligence, Wisdom, and Charisma. Each modifier is calculated as (score - 10) / 2 rounded down.",
    "A Fighter is a martial class focused on weapon attacks. They get Extra Attack at level 5, allowing two attacks per turn.",
    "A Wizard is a spellcasting class that prepares spells from a spellbook. They have low HP but powerful offensive and utility spells.",
    "A Rogue excels at stealth and dealing Sneak Attack damage when they have advantage or an ally adjacent to the target.",
    "Saving throws are defensive rolls made to resist spells, traps, and other effects. Proficiency in a save adds your proficiency bonus.",
    "The Dungeon Master describes the world, controls NPCs, and adjudicates rules.",
    "Concentration spells require focus. Taking damage forces a Constitution saving throw (DC 10 or half damage) to maintain the spell.",
    "Short rests last 1 hour and allow spending Hit Dice to recover HP. Long rests last 8 hours and fully restore HP and most abilities.",
    "Critical hits occur on a natural 20. The attack deals double dice damage.",
    "Initiative determines turn order in combat. Everyone rolls d20 plus their Dexterity modifier.",
    "Passive Perception equals 10 plus Wisdom modifier plus Perception proficiency.",
    "The Paladin's Divine Smite expends a spell slot after a hit to deal extra radiant damage.",
    "Grappling replaces one attack. You make Strength (Athletics) vs the target's Strength or Dexterity (Acrobatics).",
]

SKILLS_BY_CLASS = {
    "fighter":   ["Athletics", "Intimidation", "Weapon Mastery", "Second Wind", "Action Surge"],
    "wizard":    ["Arcana", "History", "Spellcasting", "Magic Missile", "Shield"],
    "rogue":     ["Stealth", "Sleight of Hand", "Sneak Attack", "Cunning Action", "Evasion"],
    "paladin":   ["Persuasion", "Athletics", "Divine Smite", "Lay on Hands", "Aura of Protection"],
    "ranger":    ["Survival", "Perception", "Favored Enemy", "Natural Explorer", "Volley"],
    "cleric":    ["Medicine", "Religion", "Turn Undead", "Divine Intervention", "Healing Word"],
    "barbarian": ["Athletics", "Intimidation", "Rage", "Reckless Attack", "Danger Sense"],
    "bard":      ["Performance", "Persuasion", "Bardic Inspiration", "Jack of All Trades", "Countercharm"],
    "druid":     ["Nature", "Perception", "Wild Shape", "Wildfire Spirit", "Entangle"],
    "monk":      ["Acrobatics", "Stealth", "Flurry of Blows", "Stunning Strike", "Ki Points"],
    "sorcerer":  ["Arcana", "Persuasion", "Metamagic", "Sorcery Points", "Tides of Chaos"],
    "warlock":   ["Arcana", "Deception", "Eldritch Blast", "Hex", "Pact of the Blade"],
}

VALID_CLASSES = list(SKILLS_BY_CLASS.keys())

TEAMMATE_NAMES = [
    "Aldric", "Mira", "Fenwick", "Sable", "Torvin",
    "Lira", "Draven", "Kessa", "Brom", "Nyx",
]


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    return embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

def build_vector_store() -> chromadb.Collection:
    client = chromadb.Client()
    try:
        client.delete_collection("dnd_knowledge")
    except Exception:
        pass
    collection = client.create_collection("dnd_knowledge", metadata={"hnsw:space": "cosine"})
    docs, ids, embeds = [], [], []
    for i, doc in enumerate(DND_KNOWLEDGE):
        docs.append(doc)
        ids.append(f"doc_{i}")
        embeds.append(get_embedding(doc))
    collection.add(documents=docs, embeddings=embeds, ids=ids)
    return collection

def retrieve_context(collection: chromadb.Collection, query: str, n: int = 3) -> str:
    results = collection.query(query_embeddings=[get_embedding(query)], n_results=n)
    return "\n".join(f"- {c}" for c in results["documents"][0])


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------

def generate_room(theme: str) -> DungeonRoom:
    resp = ollama_chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a DnD dungeon designer. Output valid JSON only, no extra text."},
            {"role": "user",   "content": f"Generate a dungeon room with theme: {theme}"},
        ],
        format=DungeonRoom.model_json_schema(),
        options={"temperature": 0.9},
    )
    return DungeonRoom.model_validate_json(resp.message.content)


# ---------------------------------------------------------------------------
# Character / Enemy / NPC creation
# ---------------------------------------------------------------------------

def _roll_stat() -> int:
    rolls = sorted(random.randint(1, 6) for _ in range(4))
    return sum(rolls[1:])

def _hp_for_class(char_class: str, con_mod: int) -> int:
    hit_dice = {
        "barbarian": 12, "fighter": 10, "paladin": 10, "ranger": 10,
        "monk": 8, "rogue": 8, "cleric": 8, "druid": 8, "bard": 8,
        "warlock": 8, "wizard": 6, "sorcerer": 6,
    }
    return hit_dice.get(char_class, 8) + con_mod

def _mp_for_class(char_class: str, wis_mod: int, int_mod: int) -> int:
    casters = {"wizard", "sorcerer", "warlock", "cleric", "druid", "bard", "paladin", "ranger"}
    return (10 + max(wis_mod, int_mod)) if char_class in casters else 0

def create_character(name: str, char_class: str) -> CharacterSheet:
    char_class = char_class.lower().strip()
    stats = {s: _roll_stat() for s in
             ("strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma")}
    con_mod = (stats["constitution"] - 10) // 2
    wis_mod = (stats["wisdom"]       - 10) // 2
    int_mod = (stats["intelligence"] - 10) // 2
    sheet = CharacterSheet(
        name=name,
        character_class=char_class,
        hp=_hp_for_class(char_class, con_mod),
        mp=_mp_for_class(char_class, wis_mod, int_mod),
        skills=SKILLS_BY_CLASS.get(char_class, ["Perception", "Athletics"]),
        **stats,
    )
    return sheet

def create_enemy(name: str, enemy_type: str, danger_level: str) -> EnemySheet:
    hp_map  = {"safe": 8,  "low": 15, "medium": 30, "high": 55,    "deadly": 90}
    ac_map  = {"safe": 10, "low": 12, "medium": 14, "high": 16,    "deadly": 18}
    ab_map  = {"safe": 1,  "low": 2,  "medium": 4,  "high": 6,     "deadly": 8}
    dmg_map = {"safe": "1d4", "low": "1d6", "medium": "1d8+2", "high": "2d6+3", "deadly": "3d6+5"}
    resp = ollama_chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a DnD monster creator. Output valid JSON only, no extra text."},
            {"role": "user", "content": (
                f"Create an enemy named '{name}' of type '{enemy_type}' at '{danger_level}' danger. "
                f"Stats: hp={hp_map[danger_level]}, armor_class={ac_map[danger_level]}, "
                f"attack_bonus={ab_map[danger_level]}, damage_dice={dmg_map[danger_level]}. "
                "Write a vivid description and a one-word personality trait."
            )},
        ],
        format=EnemySheet.model_json_schema(),
        options={"temperature": 0.85},
    )
    sheet = EnemySheet.model_validate_json(resp.message.content)
    game_state["active_enemy"] = sheet.model_dump()
    return sheet

def create_npc(name: str, role: str, context: str) -> NPCSheet:
    resp = ollama_chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a DnD NPC creator. Output valid JSON only, no extra text."},
            {"role": "user", "content": (
                f"Create an NPC named '{name}' who is a '{role}'. "
                f"Context: {context}. "
                "Give them a distinct personality, what they know, and a greeting line."
            )},
        ],
        format=NPCSheet.model_json_schema(),
        options={"temperature": 0.85},
    )
    sheet = NPCSheet.model_validate_json(resp.message.content)
    game_state["active_npc"] = sheet.model_dump()
    return sheet

def update_stats(entity: str, field: str, new_value) -> str:
    target = game_state.get(entity)
    if target is None:
        return f"Error: entity '{entity}' not found."
    if field not in target:
        return f"Error: field '{field}' not found on '{entity}'."
    old = target[field]
    target[field] = new_value
    return f"[STAT UPDATE] {entity}.{field}: {old} -> {new_value}"


# ---------------------------------------------------------------------------
# Conversation summarization
# ---------------------------------------------------------------------------

def summarize_history(history: list[dict], label: str) -> list[dict]:
    if len(history) <= 2:
        return history
    system_msg = history[0]
    body_text  = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[1:])
    resp = ollama_chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a concise DnD session summarizer. "
                "Preserve: HP values, items, decisions, locations, characters met. "
                "Be factual and brief. Output only the summary, no preamble."
            )},
            {"role": "user", "content": body_text},
        ],
        options={"temperature": 0.2},
    )
    return [
        system_msg,
        {
            "role": "assistant",
            "content": (
                f"[SESSION SUMMARY -- {label}]\n{resp.message.content}\n"
                "[End of summary. Continue from this point.]"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

DM_SYSTEM_PROMPT = """You are the Dungeon Master of a DnD 5e campaign.

== YOUR ROLE ==
- Describe the world, drive the narrative, and adjudicate rules impartially.
- You are NOT a helpful assistant. You are a neutral narrator and referee.
- NEVER output raw JSON, tool call syntax, or bracket templates in your visible response.
  Tool calls happen silently. Your visible output is ONLY natural narrative prose.

== PLAYER AGENCY — THE MOST IMPORTANT RULE ==
The player controls their own character COMPLETELY. You NEVER:
- Narrate what the player character thinks, feels, decides, or does.
- Assume the player takes an action and narrate its result before they choose it.
- Move the story forward based on what you think the player WOULD do.
- Put words, plans, reactions, or intentions in the player's mouth or head.

You describe what the WORLD does and what NPCs/enemies do.
Then you STOP and wait for the player to tell you their next action.

When the player states an action (e.g. "I roll perception", "I walk over to him"):
  1. Resolve it mechanically if needed (roll dice, apply modifiers silently).
  2. Narrate only what the player's character perceives or what happens as a result.
  3. End with the world in a state of waiting — never assume the player's next move.

== CHARACTER CREATION ==
Character creation is handled BEFORE you receive your first message.
Both the player and teammate already have full character sheets.
Do NOT ask the player to choose a name or class. Simply begin the adventure.

== SUB-AGENTS UNDER YOUR AUTHORITY ==
You oversee an enemy agent and an NPC agent. Delegate using ONLY these exact tags:

  To delegate an enemy turn:    [ENEMY_TURN: <situation in one sentence>]
  To delegate an NPC response:  [NPC_TURN: <what the players said/asked>]

These tags are intercepted and replaced before output reaches the player.

== TEAMMATE ==
A second player character (the teammate) is present. They are NOT under your authority.
- Treat their inputs exactly like the main player's inputs.
- Do NOT proceed with the story while the two players are talking to each other.
- The teammate's choices influence the story, but you still cannot act FOR either player.

== TOOL USAGE ==
You have access to: roll_dice, calculate_damage, get_character_stat.

Correct attack resolution order for the PLAYER attacking (silent until step 5):
  1. Call roll_dice(n_dice=1, sides=20, modifier=<relevant modifier>) silently.
  2. Call calculate_damage(base_damage=<damage>, armor_class=<enemy AC>,
     attack_roll=<step 1 result>) silently.
  3. If it hit, call update_stats(entity="active_enemy", field="hp",
     new_value=<enemy current hp minus damage>) silently.
  4. If enemy HP reaches 0, narrate its death.
  5. Narrate the outcome in plain prose, then show updated HP totals.

After the player acts in combat, always delegate the enemy turn with [ENEMY_TURN: ...].

During combat, show player HP and enemy HP after every exchange.
Hide HP during exploration and conversation.
Always end your turn with an open question or prompt — NEVER assume the player's next choice.
"""

ENEMY_AGENT_PROMPT = """You are roleplaying as a specific enemy in a DnD encounter.
Your character sheet and the current situation will be provided.

Your turn proceeds in this EXACT order — you MUST complete each step before moving to the next:
  1. Call roll_dice(n_dice=1, sides=20, modifier=<your attack_bonus>) to get your attack roll.
     Note the total integer from the result — this is your attack_roll.
  2. Call roll_dice again with your damage dice (e.g. n_dice=1, sides=6, modifier=0) to get your damage.
     Note the total integer from the result — this is your base_damage.
  3. ONLY AFTER steps 1 and 2 are complete, call calculate_damage(
       base_damage=<integer from step 2>,
       armor_class=<player's armor_class as an integer>,
       attack_roll=<integer from step 1>
     ). NEVER pass None to any argument.
  4. If the attack hit, call update_stats(entity="player", field="hp",
     new_value=<player current hp minus damage>) to apply the damage.
  5. Narrate your action in 1-2 vivid sentences as the enemy acting — not as a narrator.

Additional rules:
- Do NOT ask questions. Simply act.
- Do NOT output JSON or tool call syntax in your narration.
- Do NOT call calculate_damage before you have real integers from both roll_dice calls.
- Do NOT skip update_stats when you land a hit. HP must always be kept accurate.
- Your own HP is tracked by the DM. You do not call update_stats on yourself.
"""

NPC_AGENT_PROMPT = """You are roleplaying as a specific NPC in a DnD world.
Your name, role, personality, and knowledge will be provided.
- Speak ONLY as your character. Never break character.
- You know only what your character sheet says you know.
- Keep responses brief and conversational.
- Do NOT resolve game mechanics. Simply talk as your character would.
- Do NOT output JSON or tool syntax.
"""

def make_teammate_prompt(player_name: str) -> str:
    return f"""You are a player character in a DnD adventure — the main player's companion.
Your character sheet will be given to you at the start. Play as a human player would at a tabletop game.

== HOW TO BEHAVE ==
- Speak as your character in 1-2 sentences. You are not the storyteller.
- React to what is happening in the world around you.
- Always refer to yourself using "I" and "me" — never "you" when meaning yourself.
- Always refer to the main player by their name: {player_name}. Never call them "you".
- You do NOT control the story or invent facts the DM has not established.

== WHAT YOU MUST NEVER DO ==
- NEVER narrate {player_name}'s thoughts, feelings, or internal monologue.
  Wrong: "{player_name} thinks to themselves..."
  Wrong: "You feel a sense of dread..."
  Right: "I have a bad feeling about this. What do you make of it, {player_name}?"
- NEVER describe what {player_name} does or decides. Only {player_name} controls {player_name}.
- NEVER refer to yourself as "you". You are "I". {player_name} is "{player_name}".

== TOOL / MECHANIC RULES — ABSOLUTE ==
- You have NO access to any tools whatsoever — not roll_dice, not get_character_stat, not any other.
- Never call a tool, reference a tool result, or output anything resembling JSON or code.
- Never look up or report stat numbers. You are a player character, not a rules referee.
- For any action needing a roll or stat check, express your intent in plain speech and defer to the DM:
  Correct: "I want to try reading his body language — DM, can you call for a check?"
  Wrong:   {{"name": "roll_dice", ...}}
  Wrong:   "The stat result is: {{\"result\": \"12/12\"}}"
"""


# ---------------------------------------------------------------------------
# Response sanitizer -- strips leaked JSON blobs line-by-line
# ---------------------------------------------------------------------------

_JSON_BLOB     = re.compile(r'\{[^{}]*\}', re.DOTALL)   # any {...} object
_JSON_OPEN     = re.compile(r'^\s*\{\s*"')              # line opening a JSON object
_BRACKET_TMPL  = re.compile(r'\[(?:Silent tool call|Receive result)[^\]]*\]', re.IGNORECASE)
_TOOL_CALL_HDR = re.compile(r'^\s*Tool(?:\s+call)?:\s*', re.IGNORECASE | re.MULTILINE)
# Preamble lines that introduce leaked tool results
_RESULT_PREAMBLE = re.compile(r'^[^\n]*\bstat(?:s)?\b[^\n]*result[^\n]*$', re.IGNORECASE | re.MULTILINE)

def sanitize(text: str) -> str:
    """Remove any leaked raw tool-call artifacts from an agent response."""
    # First pass: regex-kill known patterns
    text = _JSON_BLOB.sub("", text)
    text = _BRACKET_TMPL.sub("", text)
    text = _TOOL_CALL_HDR.sub("", text)
    text = _RESULT_PREAMBLE.sub("", text)

    # Second pass: strip any lines that open a JSON object (catches multi-line blobs)
    lines = text.splitlines()
    clean_lines = []
    brace_depth = 0
    for line in lines:
        stripped = line.strip()
        if brace_depth == 0 and _JSON_OPEN.match(stripped):
            # Start of a JSON block — begin skipping
            brace_depth += stripped.count('{') - stripped.count('}')
            if brace_depth <= 0:
                brace_depth = 0  # single-line object, done
            continue
        if brace_depth > 0:
            brace_depth += stripped.count('{') - stripped.count('}')
            if brace_depth <= 0:
                brace_depth = 0
            continue
        clean_lines.append(line)

    text = "\n".join(clean_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sub-agent runner
# ---------------------------------------------------------------------------

async def run_sub_agent(
    agent_type: Literal["enemy", "npc"],
    context: str,
    tools: list,
    llm: ChatOllama,
) -> str:
    system = ENEMY_AGENT_PROMPT if agent_type == "enemy" else NPC_AGENT_PROMPT
    agent  = create_react_agent(llm, tools)
    resp   = await agent.ainvoke({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": context},
        ]
    })
    return sanitize(resp["messages"][-1].content)


# ---------------------------------------------------------------------------
# Teammate agent
# ---------------------------------------------------------------------------

class TeammateAgent:
    def __init__(self, llm: ChatOllama, tools: list, player_name: str = "the player"):
        self.llm     = llm
        # The teammate is a roleplaying agent only — it has NO tool access.
        # Using an empty tools list prevents the LLM from ever attempting tool calls
        # like roll_dice(entity=..., skill=...) which crash with validation errors.
        self.agent   = create_react_agent(llm, [])
        self.history: list[dict] = [
            {"role": "system", "content": make_teammate_prompt(player_name)}
        ]

    def set_character(self, sheet: CharacterSheet):
        game_state["teammate"]        = sheet.model_dump()
        game_state["teammate_skills"] = sheet.skills
        self.history.append({
            "role": "user",
            "content": (
                f"Your character sheet:\n{json.dumps(sheet.model_dump(), indent=2)}\n"
                "Remember this is who you are. The adventure is about to begin."
            ),
        })

    def _maybe_summarize(self):
        if len(self.history) >= SUMMARY_THRESHOLD:
            self.history = summarize_history(self.history, "Teammate")

    async def react(self, dm_narration: str) -> str:
        self._maybe_summarize()
        self.history.append({"role": "user", "content": f"[DM narrates]: {dm_narration}"})
        resp  = await self.agent.ainvoke({"messages": self.history})
        reply = sanitize(resp["messages"][-1].content)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    async def converse(self, player_message: str) -> str:
        self._maybe_summarize()
        self.history.append({"role": "user", "content": f"[Player says to you]: {player_message}"})
        resp  = await self.agent.ainvoke({"messages": self.history})
        reply = sanitize(resp["messages"][-1].content)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def is_asking_player(self, text: str) -> bool:
        stripped = text.strip()
        sentences = [s.strip() for s in stripped.replace("?", "?.").split(".") if s.strip()]
        if any("?" in s for s in sentences):
            return True
        address_phrases = [
            "what do you think", "what say you", "shall we", "should we",
            "do you want", "your call", "up to you", "what would you like",
        ]
        lower = stripped.lower()
        return any(p in lower for p in address_phrases)


# ---------------------------------------------------------------------------
# Character creation flow
# ---------------------------------------------------------------------------

def character_creation_flow(teammate: TeammateAgent) -> tuple[CharacterSheet, CharacterSheet]:
    print("\nDM: Welcome, adventurer. Before we begin, tell me your name.\n")

    player_name = ""
    while not player_name:
        raw = input("You: ").strip()
        lower = raw.lower()
        has_class_inline = any(
            f", {cls}" in lower or lower.endswith(f" {cls}") or lower == cls
            for cls in VALID_CLASSES
        )
        if has_class_inline:
            print("DM: Please give me just your name -- we will discuss your class next.\n")
        elif raw:
            player_name = raw

    print(f"\nDM: Well met, {player_name}. What is your class?")
    print(f"    Choose from: {', '.join(VALID_CLASSES)}\n")

    player_class = ""
    while player_class not in VALID_CLASSES:
        raw = input(f"{player_name} (you): ").strip().lower().split()
        raw = raw[0] if raw else ""
        if raw in VALID_CLASSES:
            player_class = raw
        else:
            print(f"DM: '{raw}' is not a known class. Choose from: {', '.join(VALID_CLASSES)}\n")

    player_sheet = create_character(player_name, player_class)
    game_state["player"]        = player_sheet.model_dump()
    game_state["player_skills"] = player_sheet.skills

    print(f"\n  {player_name} the {player_class.capitalize()}")
    print(f"  HP {player_sheet.hp}  |  MP {player_sheet.mp}")
    print(f"  STR {player_sheet.strength}  DEX {player_sheet.dexterity}  CON {player_sheet.constitution}  "
          f"INT {player_sheet.intelligence}  WIS {player_sheet.wisdom}  CHA {player_sheet.charisma}")
    print(f"  Skills: {', '.join(player_sheet.skills)}\n")

    teammate_name  = random.choice(TEAMMATE_NAMES)
    teammate_class = random.choice(VALID_CLASSES)
    teammate_sheet = create_character(teammate_name, teammate_class)
    teammate.set_character(teammate_sheet)

    print(f"  {teammate_name} the {teammate_class.capitalize()} joins you.")
    print(f"  HP {teammate_sheet.hp}  |  MP {teammate_sheet.mp}")
    print(f"  Skills: {', '.join(teammate_sheet.skills)}\n")

    return player_sheet, teammate_sheet


# ---------------------------------------------------------------------------
# Delegation tag parser
# ---------------------------------------------------------------------------

_ENEMY_TAG = re.compile(r'\[ENEMY_TURN:\s*(.*?)\]', re.DOTALL)
_NPC_TAG   = re.compile(r'\[NPC_TURN:\s*(.*?)\]',   re.DOTALL)

async def resolve_delegation_tags(
    dm_text: str,
    tools: list,
    llm_sub: ChatOllama,
) -> str:
    enemy_match = _ENEMY_TAG.search(dm_text)
    if enemy_match:
        situation = enemy_match.group(1).strip()
        enemy     = game_state.get("active_enemy") or {}
        context   = (
            f"Your character: {json.dumps(enemy)}\n"
            f"Situation: {situation}\n"
            "Take your action now."
        )
        reply    = await run_sub_agent("enemy", context, tools, llm_sub)
        name     = enemy.get("name", "The enemy")
        dm_text  = _ENEMY_TAG.sub(f"\n{name}: {reply}\n", dm_text, count=1)

    npc_match = _NPC_TAG.search(dm_text)
    if npc_match:
        situation = npc_match.group(1).strip()
        npc       = game_state.get("active_npc") or {}
        context   = (
            f"Your character: {json.dumps(npc)}\n"
            f"Situation: {situation}\n"
            "Respond as your character."
        )
        reply    = await run_sub_agent("npc", context, tools, llm_sub)
        name     = npc.get("name", "The NPC")
        dm_text  = _NPC_TAG.sub(f'\n{name}: "{reply}"\n', dm_text, count=1)

    return dm_text


# ---------------------------------------------------------------------------
# Teammate-address detector
# ---------------------------------------------------------------------------

def player_is_addressing_teammate(text: str) -> bool:
    lower         = text.lower()
    teammate_name = (game_state["teammate"] or {}).get("name", "").lower()
    signals = [
        teammate_name and teammate_name in lower,
        lower.startswith("hey "),
        "what do you think" in lower,
        "should we" in lower,
        "do you want" in lower,
        "do you think" in lower,
        "what if we" in lower,
        "you agree" in lower,
    ]
    return any(signals)


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

async def run_agent(collection: chromadb.Collection):
    print("=" * 60)
    print("  Dungeons and Dragons -- AI Dungeon Master")
    print("  /exit         quit the game")
    print("  /room <theme>   generate a new room")
    print("  /skills       list your skills")
    print("=" * 60)

    mcp_server_code = """
import random
from typing import Optional
from fastmcp import FastMCP

mcp = FastMCP("dnd-tools-server")

@mcp.tool()
def roll_dice(n_dice: int, sides: int, modifier: int = 0) -> str:
    \"\"\"Roll n_dice dice with the given number of sides and add a modifier.
    ONLY accepts: n_dice (int), sides (int), modifier (int, default 0).
    Do NOT pass entity, skill, or any other arguments — they are invalid.\"\"\"
    rolls = [random.randint(1, sides) for _ in range(n_dice)]
    total = sum(rolls) + modifier
    return f"Rolled {n_dice}d{sides}+{modifier}: {rolls} = {total}"

@mcp.tool()
def calculate_damage(base_damage: Optional[int], armor_class: Optional[int], attack_roll: Optional[int]) -> str:
    \"\"\"Determine whether an attack hits and how much damage it deals.
    You MUST call roll_dice twice first to obtain base_damage and attack_roll
    before calling this tool. Never pass None for any argument.\"\"\"
    if base_damage is None or armor_class is None or attack_roll is None:
        return (
            "Error: one or more required arguments were None. "
            "You must call roll_dice to obtain base_damage and attack_roll "
            "BEFORE calling calculate_damage. Do not pass None."
        )
    if attack_roll >= armor_class:
        return f"Hit! {base_damage} damage dealt."
    return "Miss! No damage dealt."

@mcp.tool()
def get_character_stat(entity: str, stat: str) -> str:
    \"\"\"Look up a stat. entity is player, teammate, active_enemy, or active_npc.\"\"\"
    return f"Check the game state context for {entity}.{stat}."

@mcp.tool()
def update_stats(entity: str, field: str, new_value: int) -> str:
    \"\"\"Update a numeric stat on player, teammate, or active_enemy.
    entity : player | teammate | active_enemy
    field  : hp | mp | strength | dexterity | constitution | intelligence | wisdom | charisma
    new_value : the new integer value to set
    Returns a confirmation string. Always call this after damage is dealt or HP changes.\"\"\"
    import json, pathlib, os
    state_file = pathlib.Path(os.environ.get("DND_STATE_FILE", "/tmp/dnd_state.json"))
    if not state_file.exists():
        return f"Error: state file not found at {state_file}."
    state = json.loads(state_file.read_text())
    target = state.get(entity)
    if target is None:
        return f"Error: entity '{entity}' not found in game state."
    if field not in target:
        return f"Error: field '{field}' not found on '{entity}'."
    old = target[field]
    target[field] = new_value
    state_file.write_text(json.dumps(state))
    return f"[STAT UPDATE] {entity}.{field}: {old} -> {new_value}"

mcp.run(transport="stdio")
"""

    import tempfile, pathlib, os

    state_file = pathlib.Path(tempfile.mktemp(suffix="_dnd_state.json"))
    state_file.write_text(json.dumps(game_state))
    env = {**os.environ, "DND_STATE_FILE": str(state_file)}

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-c", mcp_server_code],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            llm_dm       = ChatOllama(model=MODEL, temperature=0.75)
            llm_sub      = ChatOllama(model=MODEL, temperature=0.85)
            llm_teammate = ChatOllama(model=MODEL, temperature=0.9)

            teammate = TeammateAgent(llm_teammate, tools)  # temporary, name not yet known

            player_sheet, teammate_sheet = character_creation_flow(teammate)

            # Now that we have the player name, reinitialise the teammate's
            # system prompt so it refers to the player by name correctly.
            player_name = game_state["player"]["name"]
            teammate.history[0] = {"role": "system", "content": make_teammate_prompt(player_name)}

            dm_history: list[dict] = [
                {"role": "system", "content": DM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Character creation is complete. Here are the sheets:\n\n"
                        f"PLAYER: {json.dumps(game_state['player'], indent=2)}\n\n"
                        f"TEAMMATE: {json.dumps(game_state['teammate'], indent=2)}\n\n"
                        "Begin the adventure now. Describe a vivid opening scene. "
                        "Do NOT ask for a name or class. Do NOT output any JSON. "
                        "End by presenting the situation and waiting for the players to decide what to do."
                    ),
                },
            ]

            dm_agent = create_react_agent(llm_dm, tools)

            def maybe_summarize_dm():
                nonlocal dm_history
                if len(dm_history) >= SUMMARY_THRESHOLD:
                    dm_history = summarize_history(dm_history, "DM")

            while True:
                maybe_summarize_dm()

                state_file.write_text(json.dumps(game_state))

                dm_raw  = await dm_agent.ainvoke({"messages": dm_history})
                dm_text = sanitize(dm_raw["messages"][-1].content)

                dm_text = await resolve_delegation_tags(dm_text, tools, llm_sub)
                dm_text = sanitize(dm_text)

                print(f"\nDM: {dm_text}\n")
                dm_history.append({"role": "assistant", "content": dm_text})

                try:
                    updated = json.loads(state_file.read_text())
                    for k in ("player", "teammate", "active_enemy", "active_npc"):
                        if updated.get(k) is not None:
                            game_state[k] = updated[k]
                except Exception:
                    pass

                tname    = game_state["teammate"]["name"]
                reaction = await teammate.react(dm_text)
                reaction = reaction.strip()

                side_exchange: list[str] = []
                user_input = None

                if reaction:
                    print(f"{tname}: {reaction}\n")
                    side_exchange.append(f"{tname}: {reaction}")

                    while teammate.is_asking_player(reaction):
                        sub_input = input(f"{player_name} (you): ").strip()
                        if not sub_input:
                            break

                        if sub_input == "/exit":
                            print("Farewell, adventurer!")
                            user_input = "/exit"
                            side_exchange = []
                            break

                        elif sub_input == "/skills":
                            print(f"\n  {player_name}'s skills : {', '.join(game_state['player_skills'])}")
                            print(f"  {tname}'s skills : {', '.join(game_state['teammate_skills'])}\n")
                            continue

                        side_exchange.append(f"Player: {sub_input}")

                        if not player_is_addressing_teammate(sub_input):
                            user_input = sub_input
                            reaction = ""
                            break

                        reaction = await teammate.converse(sub_input)
                        reaction = sanitize(reaction).strip()
                        if reaction:
                            print(f"\n{tname}: {reaction}\n")
                            side_exchange.append(f"{tname}: {reaction}")

                if user_input == "/exit":
                    print("Farewell, adventurer!")
                    break

                if user_input is None:
                    user_input = input(f"{player_name} (you): ").strip()

                if not user_input:
                    if side_exchange:
                        exchange_text = "\n".join(side_exchange)
                        dm_history.append({
                            "role": "user",
                            "content": (
                                f"[Player and teammate had a side conversation]\n"
                                f"{exchange_text}\n"
                                "[No world action taken yet. Wait for the players to act.]"
                            ),
                        })
                    continue

                if user_input == "/exit":
                    print("Farewell, adventurer!")
                    break

                elif user_input == "/skills":
                    print(f"\n  {player_name}'s skills : {', '.join(game_state['player_skills'])}")
                    print(f"  {tname}'s skills : {', '.join(game_state['teammate_skills'])}\n")
                    if side_exchange:
                        exchange_text = "\n".join(side_exchange)
                        dm_history.append({
                            "role": "user",
                            "content": (
                                f"[Player and teammate had a side conversation]\n"
                                f"{exchange_text}\n"
                                "[No world action taken yet.]"
                            ),
                        })
                    continue

                elif user_input.startswith("/room "):
                    theme = user_input[6:].strip()
                    print("\n[Generating room...]")
                    room = generate_room(theme)
                    print(f"\n  {room.room_name} [{room.danger_level.upper()}]")
                    print(f"  {room.description}")
                    print(f"  Enemies : {', '.join(room.enemies) or 'None'}")
                    print(f"  Items   : {', '.join(room.items) or 'None'}")
                    print(f"  Exits   : {', '.join(room.exits)}\n")
                    dm_history.append({
                        "role": "user",
                        "content": (
                            f"The players entered a new room: {room.model_dump()}. "
                            "Narrate their entrance dramatically. Do NOT output JSON."
                        ),
                    })
                    continue

                exchange_summary = ""
                if side_exchange:
                    exchange_summary = (
                        f"[Player and {tname} discussed]\n"
                        + "\n".join(side_exchange)
                        + "\n\n"
                    )

                if player_is_addressing_teammate(user_input):
                    teammate_reply = await teammate.converse(user_input)
                    teammate_reply = sanitize(teammate_reply)
                    print(f"\n{tname}: {teammate_reply}\n")
                    side_exchange.append(f"Player: {user_input}")
                    side_exchange.append(f"{tname}: {teammate_reply}")
                    exchange_text = "\n".join(side_exchange)
                    dm_history.append({
                        "role": "user",
                        "content": (
                            f"[The two players are talking to each other. Do not interject.]\n"
                            f"{exchange_text}\n"
                            "[Wait for them to address the world or take an action.]"
                        ),
                    })
                    continue

                else:
                    rag   = retrieve_context(collection, user_input)
                    p_hp  = game_state["player"].get("hp", "?")
                    t_hp  = game_state["teammate"].get("hp", "?")
                    e_hp  = game_state["active_enemy"]["hp"] if game_state["active_enemy"] else "N/A"
                    dm_history.append({
                        "role": "user",
                        "content": (
                            f"{exchange_summary}"
                            f"[DnD rules context]\n{rag}\n\n"
                            f"[HP snapshot] Player: {p_hp} | Teammate: {t_hp} | Enemy: {e_hp}\n\n"
                            f"[Player action]: {user_input}\n\n"
                            "Resolve this action and narrate only what happens in the world as a result. "
                            "Do NOT narrate what the player decides to do next or put thoughts/intentions "
                            "in the player's head. End by presenting the situation and waiting for their "
                            "next choice. Do NOT output raw JSON or tool syntax."
                        ),
                    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building knowledge base...")
    collection = build_vector_store()
    asyncio.run(run_agent(collection))