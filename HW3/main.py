from __future__ import annotations
from autogen import ConversableAgent, register_function
import os, sys, re, ast
from typing import Dict, List, get_type_hints
import json

SCORE_KEYWORDS: dict[int, list[str]] = {
    1: ["awful", "horrible", "disgusting"],
    2: ["bad", "offensive", "disinterested"],
    3: ["average", "uninspiring", "middling", "okay", "unpleasant"],
    4: ["good", "enjoyable", "satisfying", "forgettable", "friendly", "peak", "great", "efficient"],
    5: ["awesome", "incredible", "amazing"]
}

# ────────────────────────────────────────────────────────────────
# 0. OpenAI API key setup ── *Do **not** modify this block.*
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ────────────────────────────────────────────────────────────────
# 1. Utility data structures & helper functions
# ────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def fetch_restaurant_data(restaurant_name: str) -> dict[str, list[str]]:
    data = {}
    target = normalize(restaurant_name)
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            name, review = line.split('.', 1)
            if normalize(name) == target:
                data.setdefault(name.strip(), []).append(review.strip())
    data = json.dumps(data, ensure_ascii=False)
    # save into json
    with open(f"{target}_restaurant_data.json", "w", encoding="utf-8") as f:
        f.write(data)
    return data


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> dict[str, str]:
    """Geometric-mean rating rounded to 3 dp."""
    n = len(food_scores)
    print(f"Calculating score for {restaurant_name} with {n} reviews.")
    if n == 0 or n != len(customer_service_scores):
        raise ValueError("food_scores and customer_service_scores must be non-empty and same length")
    total = sum(((f**2 * s)**0.5) * (1 / (n * (125**0.5))) * 10 for f, s in zip(food_scores, customer_service_scores))
    return {restaurant_name: f"{total:.3f}"}

# register functions
fetch_restaurant_data.__annotations__ = get_type_hints(fetch_restaurant_data)
calculate_overall_score.__annotations__ = get_type_hints(calculate_overall_score)

# ──────────────────────────────────────────────
# 2. Agent setup
# ──────────────────────────────────────────────

def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

DATA_FETCH = build_agent(
    "fetch_agent",
    'Return JSON {"call":"fetch_restaurant_data","args":{"restaurant_name":"<name>"}}'
)
ANALYZER = build_agent(
    "review_analyzer_agent",
    f"""
Input is {{"Restaurant Name": ["Review1", "Review2", ... ]}}. 
Treat each array element as one review, we will give you the amount of reviews, the returned array size needs to be exactly the same as the input size.
For each review (i.e. each string in the array), identify exactly two adjectives:
  - one describing the FOOD 
  - one describing the SERVICE 
FOOD and SERVICE adjectives are independent of each others.
Map each adjective to its numeric score using the predefined Table, if there are multiple adjectives, use the score of the avg.
f{SCORE_KEYWORDS}.
There might be along with some common synonyms and typical misspellings.

Reply only:\nfood_scores=[...]\ncustomer_service_scores=[...]
"""
)
SCORER = build_agent(
    "scoring_agent",
    "Given name + two lists. Reply only: \n{restaurant_name} = <score>\n"
)
ENTRY = build_agent("entry", "Coordinator")

# register functions
register_function(
    fetch_restaurant_data,
    caller=DATA_FETCH,
    executor=ENTRY,
    name="fetch_restaurant_data",
    description="Fetch reviews from specified data file by name.",
)
register_function(
    calculate_overall_score,
    caller=SCORER,
    executor=ENTRY,
    name="calculate_overall_score",
    description="Compute final rating via geometric mean.",
)


# ────────────────────────────────────────────────────────────────
# 3. Conversation helpers
# ────────────────────────────────────────────────────────────────

def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
    for step in sequence:
        msg = step["message"].format(**ctx)
        chat = entry.initiate_chat(
            step["recipient"], message=msg,
            summary_method=step.get("summary_method", "last_msg"),
            max_turns=step.get("max_turns", 2),
        )
        out = chat.summary
        # Data fetch output
        if step["recipient"] is DATA_FETCH:
            for past in reversed(chat.chat_history):
                try:
                    data = ast.literal_eval(past["content"])
                    if isinstance(data, dict) and data and not ("call" in data):
                        ctx.update({"reviews_dict": data, "restaurant_name": next(iter(data))})
                        break
                except:
                    continue
        # Analyzer output passed directly
        elif step["recipient"] is ANALYZER:
            ctx["analyzer_output"] = out
    return out

ConversableAgent.initiate_chats = lambda self, seq: run_chat_sequence(self, seq)

# ──────────────────────────────────────────────
# 4. Main entry
# ──────────────────────────────────────────────

def main(user_query: str, data_path: str = "restaurant-data.txt"):
    global DATA_PATH
    DATA_PATH = data_path
    score_result = None

    while score_result is None:
        # 1. ENTRY calls DATA_FETCH
        ENTRY._initiate_chats_ctx = {"user_query": user_query}
        fetch_chat = ENTRY.initiate_chat(
            DATA_FETCH,
            message=f"Find reviews for this query: {user_query}",
            summary_method="last_msg",
            max_turns=2
        )


        reviews_dict = None
        for msg in reversed(fetch_chat.chat_history):
            if msg.get("role") == "tool":
                try:
                    reviews_dict = json.loads(msg["content"])
                except json.JSONDecodeError:
                    continue
                break

        if not reviews_dict:
            raise RuntimeError("No reviews returned by DATA_FETCH")

        restaurant_name = next(iter(reviews_dict))

        # 3. ENTRY calls ANALYZER, feeding it the received data
        reviews_list = reviews_dict[restaurant_name]
        n = len(reviews_list)
        reviews_json = json.dumps(reviews_dict, ensure_ascii=False)

        analyzer_prompt = f"""
    Here is a valid JSON object mapping one restaurant to its reviews:

    {reviews_json}

    There are exactly {n} reviews in the array under "{restaurant_name}".  
    Reply only with two Python lists of length exactly {n}:

        food_scores = [...]
        customer_service_scores = [...]
    """
        print(analyzer_prompt)
        analyzer_chat = ENTRY.initiate_chat(
            ANALYZER,
            message=(
                analyzer_prompt
            ),
            summary_method="last_msg",
            max_turns=1
        )

        analyzer_output = analyzer_chat.summary  # e.g. "food_scores=[...]\ncustomer_service_scores=[...]"
        analyzer_output += f"\n{restaurant_name}"
        print("ANALYZER output:", analyzer_output)
        

        # 4. ENTRY calls SCORER, feeding it the analyzer’s output
        scorer_chat = ENTRY.initiate_chat(
            SCORER,
            message=analyzer_output,
            summary_method="last_msg",
            max_turns=2
        )

        score_result = scorer_chat.summary  # e.g. '{"Subway": "4.236"}'
        # if the result does'n contain any numbers, retry
        if not re.search(r"\d", score_result):
            print("No numbers found in SCORER output, retrying...")
            score_result = None
            continue
        

    print("Final score:", score_result)
    return score_result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py path/to/data.txt "How good is Subway?" ')
        sys.exit(1)

    path = sys.argv[1]
    query = sys.argv[2]
    main(query, path)
