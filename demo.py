import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import arxiv

from colors import BOLD, DIM, RESET, BLUE, GREEN, YELLOW, CYAN, RED


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """
    Suche nach wissenschaftlichen Papern auf arXiv.
    Gibt Titel, Autoren, Jahr und arXiv-ID zurück.
    Nutze die IDs mit fetch_abstract für den vollständigen Abstract.
    """
    client = arxiv.Client(page_size=max_results, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        authors = ", ".join(a.name for a in paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        results.append(
            f"ID: {paper.get_short_id()}\n"
            f"Titel: {paper.title}\n"
            f"Autoren: {authors}\n"
            f"Jahr: {paper.published.year}\n"
            f"Kategorien: {', '.join(paper.categories[:3])}"
        )
    return "\n\n---\n\n".join(results) if results else "Keine Paper gefunden."


@tool
def fetch_abstract(arxiv_id: str) -> str:
    """
    Lädt den vollständigen Abstract eines arXiv-Papers per ID.
    Nutze IDs aus arxiv_search (Format: '2602.03128' oder '2602.03128v1').
    """
    client = arxiv.Client(page_size=1, delay_seconds=3)
    search = arxiv.Search(id_list=[arxiv_id])
    for paper in client.results(search):
        authors = ", ".join(a.name for a in paper.authors)
        return (
            f"Titel: {paper.title}\n"
            f"Autoren: {authors}\n"
            f"Veröffentlicht: {paper.published.strftime('%d.%m.%Y')}\n"
            f"URL: {paper.entry_id}\n\n"
            f"Abstract:\n{paper.summary}"
        )
    return f"Paper '{arxiv_id}' nicht gefunden."


@tool
def summarize(text: str) -> str:
    """
    Verdichtet einen langen Text zu 4-6 prägnanten Kernaussagen auf Deutsch.
    Ideal nach dem Laden mehrerer Abstracts zur Synthese.
    """
    llm = ChatOpenAI(model="gpt-5.5", temperature=0)
    return llm.invoke(
        "Fasse den folgenden Text in 4-6 prägnante Kernaussagen auf Deutsch zusammen. "
        "Jede Aussage als Bullet Point (•), maximal 2 Sätze:\n\n" + text
    ).content


# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Du bist ein akademischer Forschungsassistent.
Deine Aufgabe: wissenschaftliche Literatur zu einem Thema recherchieren und strukturiert aufbereiten.

Wichtig: Bevor du ein Tool aufrufst, schreibe immer zuerst einen kurzen Gedanken im Format:
Gedanke: <deine Überlegung, warum du dieses Tool mit diesen Argumenten aufrufst>

Vorgehensweise:
1. Suche mit arxiv_search nach relevanten Papern
2. Lade die Abstracts der 2-3 relevantesten Paper mit fetch_abstract
3. Verdichte die Erkenntnisse mit summarize
4. Präsentiere ein strukturiertes Ergebnis mit Quellenangaben (Autor, Jahr, arXiv-ID)

Antworte immer auf Deutsch. Zitiere alle Quellen am Ende.

Die finale Antwort darf absolut kein Markdown enthalten. Keine ##, keine **, keine *, keine --, keine Tabellen, keine ---. Nur reiner Fließtext mit normalen Absätzen."""


# ── Pretty-Printing des ReAct-Loops ─────────────────────────────────────────

def print_message(msg: object) -> None:
    """Gibt eine Nachricht im ReAct-Loop formatiert aus."""
    if isinstance(msg, HumanMessage):
        print(f"\n{BOLD}{BLUE}┌─ USER QUERY {'─' * 42}┐{RESET}")
        print(f"{BLUE}  {msg.content}{RESET}")
        print(f"{BOLD}{BLUE}└{'─' * 56}┘{RESET}\n")

    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            if msg.content:
                print(f"{BOLD}💭 THOUGHT{RESET}  {DIM}{msg.content}{RESET}")
            for call in msg.tool_calls:
                print(f"{BOLD}{YELLOW}▶ ACTION   {RESET}{YELLOW}{call['name']}{RESET}")
                for k, v in call["args"].items():
                    preview = repr(v)
                    if len(preview) > 80:
                        preview = preview[:77] + "..."
                    print(f"  {DIM}{k} = {preview}{RESET}")
        else:
            print(f"\n{BOLD}{GREEN}✓ FINAL ANSWER {'─' * 40}{RESET}")
            print(f"{GREEN}{msg.content}{RESET}")
            print(f"{BOLD}{GREEN}{'─' * 56}{RESET}\n")

    elif isinstance(msg, ToolMessage):
        preview = msg.content.replace("\n", " ")[:280]
        if len(msg.content) > 280:
            preview += " ..."
        print(f"  {CYAN}↳ OBSERVATION  {DIM}{preview}{RESET}\n")


# ── Agent Setup & Run ────────────────────────────────────────────────────────

def build_agent():
    llm = ChatOpenAI(model="gpt-5.5", temperature=0)
    tools = [arxiv_search, fetch_abstract, summarize]
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def run(query: str) -> None:
    agent = build_agent()

    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  Academic Research Agent  ·  LangGraph ReAct{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}")

    seen_ids: set[str] = set()
    for chunk in agent.stream(
        {"messages": [("user", query)]},
        stream_mode="values",
    ):
        msg = chunk["messages"][-1]
        msg_id = getattr(msg, "id", None)
        if msg_id and msg_id in seen_ids:
            continue
        if msg_id:
            seen_ids.add(msg_id)
        print_message(msg)


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{RED}Fehler: OPENAI_API_KEY ist nicht gesetzt.{RESET}")
        sys.exit(1)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print(f"{BOLD}Academic Research Agent{RESET} — Thema eingeben oder Enter für Beispiel:")
        raw = input("  Thema: ").strip()
        query = raw or "Multi-Agent LLM frameworks benchmark comparison 2024 2025"

    run(query)