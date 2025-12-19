"""
The Boardroom AI - Strategic Decision Arena POC
================================================
Une arène de décision stratégique automatisée avec 5 personas IA.

Modèle: Gemini 2.0 Flash-Lite (Google AI)
"""

import os
import asyncio
from typing import TypedDict, List, Dict, Any, Literal
from dotenv import load_dotenv

import chainlit as cl
from langgraph.graph import StateGraph, END
import litellm

# ============================================================================
# CONFIGURATION & VALIDATION
# ============================================================================

load_dotenv()

# Configuration LiteLLM pour Gemini
MODEL_NAME = "gemini/gemini-2.0-flash-lite"

def validate_environment() -> bool:
    """Valide la présence de la clé API Google."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
    os.environ["GEMINI_API_KEY"] = api_key
    return True

# ============================================================================
# ÉTAT DU GRAPHE
# ============================================================================

class AgentState(TypedDict):
    """État partagé entre tous les agents du débat."""
    topic: str
    current_step: int
    messages: List[Dict[str, str]]

# ============================================================================
# DÉFINITION DES PERSONAS
# ============================================================================

PERSONAS = {
    "visionnaire": {
        "name": "Le Visionnaire",
        "avatar": "https://api.dicebear.com/7.x/adventurer/svg?seed=adv-1",
        "color": "#4CAF50",
    },
    "analyste_risques": {
        "name": "L'Analyste Risques",
        "avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=bot-2",
        "color": "#FF9800",
    },
    "executeur": {
        "name": "L'Exécuteur",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=av-2",
        "color": "#2196F3",
    },
    "avocat_diable": {
        "name": "L'Avocat du Diable",
        "avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=bot-3",
        "color": "#F44336",
    },
    "maestro": {
        "name": "Le Maestro",
        "avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=bot-1",
        "color": "#9C27B0",
    },
}

# ============================================================================
# PROMPTS SYSTÈME PAR TOUR
# ============================================================================

ROUND_CONFIGS = {
    1: {
        "persona": "visionnaire",
        "role": "Cadrage des opportunités",
        "system_prompt": """Tu es Le Visionnaire dans une arène de décision stratégique.

STYLE OBLIGATOIRE:
- Ton direct, incisif, zéro politesse corporate
- Si une donnée manque, formule une "Hypothèse Critique" explicite
- N'invente JAMAIS de faits ou de chiffres

TON RÔLE (Tour 1 - Cadrage Opportunités):
Analyse la question posée et identifie:
1. Les opportunités stratégiques majeures (max 3)
2. Le potentiel de création de valeur
3. Les tendances du marché favorables
4. Les avantages compétitifs possibles

Sois audacieux mais ancré dans la réalité. Structure ta réponse clairement.
Termine par une question provocante pour le prochain intervenant.""",
    },
    2: {
        "persona": "analyste_risques",
        "role": "Cadrage des menaces",
        "system_prompt": """Tu es L'Analyste Risques dans une arène de décision stratégique.

STYLE OBLIGATOIRE:
- Ton direct, incisif, zéro politesse corporate
- Cite explicitement un point du message précédent: "Tu dis X..."
- Si une donnée manque, formule une "Hypothèse Critique" explicite
- N'invente JAMAIS de faits ou de chiffres

TON RÔLE (Tour 2 - Cadrage Menaces):
Réponds au Visionnaire et identifie:
1. Les risques business majeurs (max 3)
2. Les menaces réglementaires/légales
3. Les risques de marché et concurrence
4. Les vulnérabilités opérationnelles

Challenge les opportunités identifiées avec des contre-arguments factuels.
Termine par une question sur la faisabilité concrète.""",
    },
    3: {
        "persona": "executeur",
        "role": "Faisabilité opérationnelle",
        "system_prompt": """Tu es L'Exécuteur dans une arène de décision stratégique.

STYLE OBLIGATOIRE:
- Ton direct, incisif, zéro politesse corporate
- Cite explicitement un point du message précédent: "Tu dis X..."
- Si une donnée manque, formule une "Hypothèse Critique" explicite
- N'invente JAMAIS de faits ou de chiffres

TON RÔLE (Tour 3 - Faisabilité):
Évalue concrètement:
1. Faisabilité technique et organisationnelle
2. Estimation des coûts (ordres de grandeur)
3. Délais réalistes de mise en œuvre
4. Ressources nécessaires (équipe, compétences, outils)

Sois pragmatique et terre-à-terre. Pas de promesses vagues.
Propose un premier plan d'action concret.""",
    },
    4: {
        "persona": "avocat_diable",
        "role": "Contradiction frontale",
        "system_prompt": """Tu es L'Avocat du Diable dans une arène de décision stratégique.

STYLE OBLIGATOIRE:
- Ton direct, incisif, AGRESSIVEMENT contradictoire
- Cite explicitement un point du message précédent: "Tu dis X..."
- Si une donnée manque, formule une "Hypothèse Critique" explicite
- N'invente JAMAIS de faits mais pousse les scénarios pessimistes

TON RÔLE (Tour 4 - Contradiction Frontale):
Attaque sans pitié:
1. Les angles morts des analyses précédentes
2. Les hypothèses non validées
3. Les risques sous-estimés
4. Les biais d'optimisme évidents

Joue le rôle du pire scénario réaliste.
Pose LA question qui dérange le plus.""",
    },
    5: {
        "persona": "executeur",
        "role": "Réponse aux critiques",
        "system_prompt": """Tu es L'Exécuteur dans une arène de décision stratégique.

STYLE OBLIGATOIRE:
- Ton direct, défensif mais factuel
- Cite explicitement les critiques de l'Avocat du Diable: "Tu dis X..."
- Si une donnée manque, formule une "Hypothèse Critique" explicite
- N'invente JAMAIS de faits ou de chiffres

TON RÔLE (Tour 5 - Réponse aux Critiques):
Réponds point par point:
1. Accepte les critiques valides et ajuste le plan
2. Réfute les critiques exagérées avec des arguments
3. Propose des mitigations concrètes pour les vrais risques
4. Présente un plan révisé plus robuste

Montre que tu as écouté et intégré le feedback.
Le plan doit être plus solide qu'avant.""",
    },
    6: {
        "persona": "avocat_diable",
        "role": "Contre-attaque finale",
        "system_prompt": """Tu es L'Avocat du Diable dans une arène de décision stratégique.

STYLE OBLIGATOIRE:
- Ton direct, incisif, dernier assaut
- Cite explicitement les réponses de l'Exécuteur: "Tu dis X..."
- Si une donnée manque, formule une "Hypothèse Critique" explicite
- N'invente JAMAIS de faits

TON RÔLE (Tour 6 - Contre-Attaque Finale):
Dernière salve:
1. Les scénarios d'échec les plus probables
2. Ce qui pourrait tout faire échouer
3. Les points de non-retour dangereux
4. Le coût réel de l'échec

C'est ta dernière chance de stopper un mauvais projet.
Mais sois fair-play si le plan tient la route.""",
    },
    7: {
        "persona": "maestro",
        "role": "Synthèse du débat",
        "system_prompt": """Tu es Le Maestro dans une arène de décision stratégique.

STYLE OBLIGATOIRE:
- Ton direct, neutre, synthétique
- Tu as accès à TOUT l'historique du débat
- Sois exhaustif mais concis

TON RÔLE (Tour 7 - Synthèse):
Résume objectivement:
1. ACCORDS: Les points où tous convergent
2. DÉSACCORDS: Les points de friction non résolus
3. POINTS MANQUANTS: Ce qui n'a pas été abordé
4. HYPOTHÈSES CRITIQUES: Les assumptions non validées

Ne prends pas parti. Prépare le terrain pour l'arbitrage.""",
    },
    8: {
        "persona": "maestro",
        "role": "Présentation des options",
        "system_prompt": """Tu es Le Maestro dans une arène de décision stratégique.

STYLE OBLIGATOIRE:
- Ton direct, analytique
- Tu as accès à TOUT l'historique du débat
- Sois structuré et actionnable

TON RÔLE (Tour 8 - Arbitrage):
Présente les options:
1. OPTION A: Go complet - conditions et implications
2. OPTION B: Go conditionnel - avec quelles conditions préalables
3. OPTION C: No-Go - pourquoi et alternatives

Pour chaque option:
- Probabilité de succès estimée
- Risques résiduels
- Recommandation provisoire

Prépare la décision finale.""",
    },
    9: {
        "persona": "maestro",
        "role": "Décision finale",
        "system_prompt": """Tu es Le Maestro dans une arène de décision stratégique.

STYLE OBLIGATOIRE:
- Ton direct, décisif
- Tu dois trancher
- Justifie clairement

TON RÔLE (Tour 9 - Décision):
DÉCISION FINALE:
1. GO / GO CONDITIONNEL / NO-GO
2. Justification en 5 points maximum
3. Conditions sine qua non (si Go conditionnel)
4. Risques acceptés explicitement
5. Prochaine étape immédiate

Assume ta décision. Pas de langue de bois.""",
    },
    10: {
        "persona": "maestro",
        "role": "Rapport final",
        "system_prompt": """Tu es Le Maestro dans une arène de décision stratégique.

TON RÔLE (Tour 10 - Rapport Final):
Génère le RAPPORT FINAL au format Markdown EXACT suivant:

---

# 📋 RAPPORT DE DÉCISION

## 🎯 Décision
**[GO / GO CONDITIONNEL / NO-GO]**

## 📝 Justification
[5 lignes maximum expliquant la décision]

## ⚠️ Top 3 Risques

| Risque | Impact | Mitigation |
|--------|--------|------------|
| [Risque 1] | [Impact 1] | [Mitigation 1] |
| [Risque 2] | [Impact 2] | [Mitigation 2] |
| [Risque 3] | [Impact 3] | [Mitigation 3] |

## 🔬 Hypothèses Critiques

| Hypothèse | Méthode de Validation |
|-----------|----------------------|
| [Hypothèse 1] | [Validation rapide en < 7 jours] |
| [Hypothèse 2] | [Validation rapide en < 7 jours] |
| [Hypothèse 3] | [Validation rapide en < 7 jours] |

## 📅 Plan d'Action (7 jours)

| Jour | Qui | Action |
|------|-----|--------|
| J+1 | [Responsable] | [Action concrète] |
| J+2 | [Responsable] | [Action concrète] |
| J+3 | [Responsable] | [Action concrète] |
| J+5 | [Responsable] | [Action concrète] |
| J+7 | [Responsable] | [Action concrète] |

---

Remplis ce template avec les informations du débat. Sois concis et actionnable.""",
    },
}

# ============================================================================
# FONCTIONS LLM
# ============================================================================

async def call_llm_streaming(
    system_prompt: str,
    messages: List[Dict[str, str]],
    topic: str
) -> str:
    """Appelle le LLM avec streaming et retourne la réponse complète."""

    # Construire l'historique pour le LLM
    llm_messages = [{"role": "system", "content": system_prompt}]

    # Ajouter le topic initial
    llm_messages.append({
        "role": "user",
        "content": f"SUJET DU DÉBAT:\n{topic}"
    })

    # Ajouter l'historique des messages
    for msg in messages:
        llm_messages.append({
            "role": "assistant" if msg["role"] != "user" else "user",
            "content": f"[{msg['role']}]: {msg['content']}"
        })

    # Appel LLM avec streaming (Gemini 2.0 Flash-Lite)
    response = await litellm.acompletion(
        model=MODEL_NAME,
        messages=llm_messages,
        temperature=0.8,
        max_tokens=1500,
        stream=True
    )

    full_response = ""
    async for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            await cl.context.current_step.stream_token(content)

    return full_response

# ============================================================================
# NŒUDS DU GRAPHE
# ============================================================================

async def create_round_node(state: AgentState, round_num: int) -> AgentState:
    """Crée un nœud pour un tour spécifique du débat."""
    config = ROUND_CONFIGS[round_num]
    persona_key = config["persona"]
    persona = PERSONAS[persona_key]

    # Créer le message Chainlit avec avatar
    async with cl.Step(
        name=f"{persona['name']} - {config['role']}",
        type="llm"
    ) as step:
        step.input = f"Tour {round_num}/10"

        # Appeler le LLM avec streaming
        response = await call_llm_streaming(
            config["system_prompt"],
            state["messages"],
            state["topic"]
        )

        step.output = response

    # Envoyer le message avec avatar
    await cl.Message(
        content=response,
        author=persona["name"],
        avatar=persona["avatar"],
    ).send()

    # Mettre à jour l'état
    new_messages = state["messages"] + [{
        "role": persona["name"],
        "content": response
    }]

    return {
        "topic": state["topic"],
        "current_step": round_num + 1,
        "messages": new_messages
    }

# Créer les fonctions de nœud pour chaque tour
async def round_1(state: AgentState) -> AgentState:
    return await create_round_node(state, 1)

async def round_2(state: AgentState) -> AgentState:
    return await create_round_node(state, 2)

async def round_3(state: AgentState) -> AgentState:
    return await create_round_node(state, 3)

async def round_4(state: AgentState) -> AgentState:
    return await create_round_node(state, 4)

async def round_5(state: AgentState) -> AgentState:
    return await create_round_node(state, 5)

async def round_6(state: AgentState) -> AgentState:
    return await create_round_node(state, 6)

async def round_7(state: AgentState) -> AgentState:
    return await create_round_node(state, 7)

async def round_8(state: AgentState) -> AgentState:
    return await create_round_node(state, 8)

async def round_9(state: AgentState) -> AgentState:
    return await create_round_node(state, 9)

async def round_10(state: AgentState) -> AgentState:
    return await create_round_node(state, 10)

# ============================================================================
# CONSTRUCTION DU GRAPHE
# ============================================================================

def build_debate_graph() -> StateGraph:
    """Construit le graphe LangGraph pour le débat."""

    # Créer le graphe
    workflow = StateGraph(AgentState)

    # Ajouter les nœuds
    workflow.add_node("round_1", round_1)
    workflow.add_node("round_2", round_2)
    workflow.add_node("round_3", round_3)
    workflow.add_node("round_4", round_4)
    workflow.add_node("round_5", round_5)
    workflow.add_node("round_6", round_6)
    workflow.add_node("round_7", round_7)
    workflow.add_node("round_8", round_8)
    workflow.add_node("round_9", round_9)
    workflow.add_node("round_10", round_10)

    # Définir le point d'entrée
    workflow.set_entry_point("round_1")

    # Ajouter les transitions linéaires
    workflow.add_edge("round_1", "round_2")
    workflow.add_edge("round_2", "round_3")
    workflow.add_edge("round_3", "round_4")
    workflow.add_edge("round_4", "round_5")
    workflow.add_edge("round_5", "round_6")
    workflow.add_edge("round_6", "round_7")
    workflow.add_edge("round_7", "round_8")
    workflow.add_edge("round_8", "round_9")
    workflow.add_edge("round_9", "round_10")
    workflow.add_edge("round_10", END)

    return workflow.compile()

# ============================================================================
# INTERFACE CHAINLIT
# ============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialisation de la session Chainlit."""

    # Valider l'environnement
    if not validate_environment():
        await cl.Message(
            content="❌ **ERREUR DE CONFIGURATION**\n\n"
                    "Variable `GEMINI_API_KEY` manquante dans le fichier `.env`\n\n"
                    "Créez un fichier `.env` à la racine du projet avec:\n"
                    "```\nGEMINI_API_KEY=votre-clé-google-ai-ici\n```\n\n"
                    "Obtenez votre clé sur: https://aistudio.google.com/apikey",
            author="System"
        ).send()
        return

    # Compiler le graphe
    graph = build_debate_graph()
    cl.user_session.set("graph", graph)

    # Message d'accueil
    await cl.Message(
        content="# 🏛️ The Boardroom AI\n\n"
                "Bienvenue dans l'arène de décision stratégique.\n\n"
                "**5 experts IA** vont débattre votre question en **10 tours**:\n\n"
                "1. 🌟 **Visionnaire** - Opportunités\n"
                "2. ⚠️ **Analyste Risques** - Menaces\n"
                "3. 🔧 **Exécuteur** - Faisabilité\n"
                "4. 😈 **Avocat du Diable** - Contradiction\n"
                "5. 🔧 **Exécuteur** - Réponse aux critiques\n"
                "6. 😈 **Avocat du Diable** - Contre-attaque\n"
                "7. 👑 **Maestro** - Synthèse\n"
                "8. 👑 **Maestro** - Options\n"
                "9. 👑 **Maestro** - Décision\n"
                "10. 👑 **Maestro** - Rapport Final\n\n"
                "---\n\n"
                "**Posez votre question stratégique** pour lancer le débat.",
        author="The Boardroom"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Traitement du message utilisateur et lancement du débat."""

    graph = cl.user_session.get("graph")

    if not graph:
        await cl.Message(
            content="❌ Session non initialisée. Veuillez rafraîchir la page.",
            author="System"
        ).send()
        return

    topic = message.content

    # Confirmation du lancement
    await cl.Message(
        content=f"## 🚀 Débat lancé!\n\n"
                f"**Sujet:** {topic}\n\n"
                f"---\n\n"
                f"*Le débat va commencer. Vous êtes spectateur.*",
        author="The Boardroom"
    ).send()

    # État initial
    initial_state: AgentState = {
        "topic": topic,
        "current_step": 1,
        "messages": []
    }

    # Exécuter le graphe
    try:
        final_state = None
        async for state in graph.astream(initial_state):
            final_state = state

        # Message de fin
        await cl.Message(
            content="## ✅ Débat terminé!\n\n"
                    "Le rapport final est affiché ci-dessus.\n\n"
                    "---\n\n"
                    "*Posez une nouvelle question pour relancer un débat.*",
            author="The Boardroom"
        ).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ **Erreur durant le débat:**\n\n```\n{str(e)}\n```",
            author="System"
        ).send()

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    print("Lancez l'application avec: chainlit run app.py")
