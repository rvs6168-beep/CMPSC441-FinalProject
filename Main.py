import asyncio
import chromadb
from ollama import embeddings, chat as ollama_chat
from pydantic import BaseModel, Field
from typing import Literal
import random
from fastmcp import FastMCP
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import threading
import time
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

# Pydantic Models

class CombatAction(BaseModel):
    action_type: Literal["Attack", "Spell", "Flee", "Defend", "Help"]
    target: str
    dice_roll: int = Field(ge=1, le=20)
    damage: int = Field(ge=0)
    description: str
    hit: bool

class DungeonRoom(BaseModel):
    room_name: str
    description: str
    enemies: list[str]
    items: list[str]
    exits: list[Literal["north", "south", "east", "west", "up", "down"]]
    danger_level: Literal["safe", "low", "medium", "high", "deadly"]

# RAG Knowledge Base

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

CHARACTERS = {
    "fighter": {"strength": 16, "dexterity": 14, "constitution": 15, "intelligence": 10, "wisdom": 12, "charisma": 8},
    "wizard":  {"strength": 8,  "dexterity": 14, "constitution": 12, "intelligence": 18, "wisdom": 15, "charisma": 10},
    "rogue":   {"strength": 10, "dexterity": 18, "constitution": 12, "intelligence": 14, "wisdom": 10, "charisma": 14},
}

# RAG Setup

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

# MCP Server (runs in background thread)

def start_mcp_server():
    mcp = FastMCP("dnd-tools-server")

    @mcp.tool()
    def roll_dice(n_dice: int, sides: int, modifier: int = 0) -> str:
        """Roll n_dice dice with the given number of sides, plus a modifier."""
        rolls = [random.randint(1, sides) for _ in range(n_dice)]
        total = sum(rolls) + modifier
        return f"Rolled {n_dice}d{sides}+{modifier}: {rolls} + {modifier} = {total}"

    @mcp.tool()
    def get_character_stat(character: str, stat: str) -> str:
        """Look up a character's stat."""
        character, stat = character.lower(), stat.lower()
        if character not in CHARACTERS:
            return f"Error: character '{character}' does not exist."
        if stat not in CHARACTERS[character]:
            return f"Error: stat '{stat}' does not exist for '{character}'."
        return f"{character.capitalize()}'s {stat} is {CHARACTERS[character][stat]}."

    @mcp.tool()
    def calculate_damage(base_damage: int, armor_class: int, attack_roll: int) -> str:
        """Calculate damage dealt based on attack roll vs armor class."""
        if attack_roll >= armor_class:
            return f"Attack hit! {base_damage} damage dealt!"
        return "Attack missed! No damage dealt!"

    mcp.run(transport="stdio")

# Structured Output Helpers

def generate_room(theme: str) -> DungeonRoom:
    response = ollama_chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a DnD dungeon designer. Generate a dungeon room as valid JSON only."},
            {"role": "user",   "content": f"Generate a dungeon room with theme: {theme}"}
        ],
        format=DungeonRoom.model_json_schema(),
        options={"temperature": 0.9}
    )
    return DungeonRoom.model_validate_json(response.message.content)

def resolve_combat(situation: str) -> CombatAction:
    response = ollama_chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a DnD combat resolver. Return valid JSON only."},
            {"role": "user",   "content": f"Resolve this combat situation: {situation}"}
        ],
        format=CombatAction.model_json_schema(),
        options={"temperature": 0.8}
    )
    return CombatAction.model_validate_json(response.message.content)

# Main Agent

SYSTEM_PROMPT = """You are a DnD Dungeon Master with access to tools and deep knowledge of DnD rules.

Your behavior:
- Describe situations vividly in 2-3 sentences.
- Use roll_dice for ANY dice roll needed in the game. This includes for enemy attacks.
- Use get_character_stat when the player asks about their character's abilities.
- Use calculate_damage to resolve whether attacks hit and how much damage is dealt.
- When the player is in combat with an enemy, once the player rolls for an action, let the enemy take an action next.
- When it is the enemy's turn, show their action and their roll. Reduce the player health accordingly before progressing.
- Always display the player and the enemy's health during combat. But hide these during regular actions such as talking or exploring.
- Always set the health points of the player and the enemy at the start of the combat.
- Store and carry over the player's health to the next combat. If they heal or use a potion, make sure to reflect it.
- Never repeat the same situation twice.
- Always end your response with a specific question for the player unless they are combat and it is the enemy's turn.

CRITICAL - Tool calling rules:
- You must NEVER output any text or narration before or during a tool call.
- You must NEVER output raw tool call JSON visibly in your response.
- You must NEVER nest tool calls inside each other.
- Each tool must be called separately and independently.
- You must ONLY narrate AFTER you have received the tool result.
- Your response must contain EITHER a tool call OR narration, never both at the same time.
- You must NEVER output the raw text in brackets such as
    {"name": "roll_dice", "parameters": {"modifier":0,"n_dice":1,"sides":20}}
    [Silent tool call: roll_dice(n_dice=1, sides=20, modifier=0)]

Correct tool call order for an attack:
Step 1: Call roll_dice silently with no text before or after.
Step 2: Receive the result.
Step 3: Call calculate_damage using the number from Step 2.
Step 4: Receive the result.
Step 5: Write your narration using both results.

Example of CORRECT behavior:
The brackets here indicate that it's an action that only you should know about. Do not display this to the player.
[Silent tool call: roll_dice(n_dice=1, sides=20, modifier=0)]
[Receive result: 15]
[Silent tool call: calculate_damage(base_damage=6, armor_class=12, attack_roll=15)]
[Receive result: Attack hit! 6 damage dealt!]
[Now narrate]: "Your blade finds a gap in the goblin's armor, dealing 6 damage! What do you do next?"

Example of INCORRECT behavior (NEVER do this):
"To see if you can catch up, we need to roll Athletics."
{"name": "roll_dice", "parameters": {...}}

You also have access to retrieved DnD rules context which will be prepended to relevant queries.
"""

async def run_agent(collection: chromadb.Collection):
    print("=" * 60)
    print("  Dungeons and Dragons AI Dungeon Master")
    print("  Type /exit to quit at any time")
    print("=" * 60)

    server_params = StdioServerParameters(command=sys.executable, args=["-c", """
import random
from fastmcp import FastMCP

CHARACTERS = {
    "fighter": {"strength": 16, "dexterity": 14, "constitution": 15, "intelligence": 10, "wisdom": 12, "charisma": 8},
    "wizard":  {"strength": 8,  "dexterity": 14, "constitution": 12, "intelligence": 18, "wisdom": 15, "charisma": 10},
    "rogue":   {"strength": 10, "dexterity": 18, "constitution": 12, "intelligence": 14, "wisdom": 10, "charisma": 14},
}

mcp = FastMCP("dnd-tools-server")

@mcp.tool()
def roll_dice(n_dice: int, sides: int, modifier: int = 0) -> str:
    rolls = [random.randint(1, sides) for _ in range(n_dice)]
    total = sum(rolls) + modifier
    return f"Rolled {n_dice}d{sides}+{modifier}: {rolls} + {modifier} = {total}"

@mcp.tool()
def get_character_stat(character: str, stat: str) -> str:
    character, stat = character.lower(), stat.lower()
    if character not in CHARACTERS:
        return f"Error: character '{character}' does not exist."
    if stat not in CHARACTERS[character]:
        return f"Error: stat '{stat}' does not exist for '{character}'."
    return f"{character.capitalize()}'s {stat} is {CHARACTERS[character][stat]}."

@mcp.tool()
def calculate_damage(base_damage: int, armor_class: int, attack_roll: int) -> str:
    if attack_roll >= armor_class:
        return f"Attack hit! {base_damage} damage dealt!"
    return "Attack missed! No damage dealt!"

mcp.run(transport="stdio")
"""])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(f"MCP Tools loaded: {[t.name for t in tools]}\n")

            llm = ChatOllama(model=MODEL, temperature=0.8)
            agent = create_react_agent(llm, tools)

            chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
            chat_history.append({"role": "user", "content": "Start the adventure. Tell me what character I am playing and describe the opening scene."})

            while True:
                response = await agent.ainvoke({"messages": chat_history})
                dm_response = response["messages"][-1].content
                print(f"\nDM: {dm_response}\n")
                chat_history.append({"role": "assistant", "content": dm_response})

                user_input = input("You: ").strip()
                if not user_input:
                    continue

                if user_input == "/exit":
                    print("Farewell, adventurer!")
                    break

                elif user_input.startswith("/room "):
                    theme = user_input[6:]
                    print("\n[Generating room...]")
                    room = generate_room(theme)
                    print(f"\n  Room     : {room.room_name} [{room.danger_level.upper()}]")
                    print(f"  Desc     : {room.description}")
                    print(f"  Enemies  : {', '.join(room.enemies) if room.enemies else 'None'}")
                    print(f"  Items    : {', '.join(room.items) if room.items else 'None'}")
                    print(f"  Exits    : {', '.join(room.exits)}\n")
                    augmented = f"The player entered a new room. Room details: {room.model_dump()}. Narrate their entrance dramatically."
                    chat_history.append({"role": "user", "content": augmented})

                elif user_input.startswith("/combat "):
                    situation = user_input[8:]
                    print("\n[Resolving combat...]")
                    action = resolve_combat(situation)
                    print(f"\n  Action   : {action.action_type} on {action.target}")
                    print(f"  Roll     : {action.dice_roll}  |  Hit: {action.hit}  |  Damage: {action.damage}")
                    print(f"  Result   : {action.description}\n")
                    augmented = f"Combat was resolved: {action.model_dump()}. Narrate this dramatically."
                    chat_history.append({"role": "user", "content": augmented})

                else:
                    context = retrieve_context(collection, user_input)
                    augmented = f"[Relevant DnD rules context]\n{context}\n\n[Player says]: {user_input}"
                    chat_history.append({"role": "user", "content": augmented})


if __name__ == "__main__":
    print("Building DnD knowledge base...")
    collection = build_vector_store()
    asyncio.run(run_agent(collection))