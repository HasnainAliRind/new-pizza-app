from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from model import RecipeSession, get_db
from pydantic import BaseModel
import openai
import json

router = APIRouter()

# --- Pydantic schema ---
class RecipeRequest(BaseModel):
    session_id: str
    input_text: str
    language: str = "en"

# --- Helper: call the LLM ---
def call_llm(conversation: list) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        temperature=0.7,
        max_tokens=600,
    )
    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)  # parse into structured JSON
    except json.JSONDecodeError:
        # fallback if model slips extra text
        return {"raw_output": content}

# --- Questions map for one-by-one mode ---
QUESTIONS_MAP = {
    "experience": "What is your cooking experience? (beginner, intermediate, expert)",
    "dish_type": "What type of dish are you making? (starter, main, dessert, snack)",
    "cuisine": "What cuisine style would you like? (Italian, Indian, Mexican…)",
    "include_ingredients": "Any ingredients you'd like me to include?",
    "avoid_ingredients": "Any ingredients you'd like me to avoid?",
    "equipment": "What cooking equipment do you have? (oven, stove, grill…)",
    "time_available": "How much time do you have for cooking? (30 min, 1h, 2h+)",
    "servings": "How many servings do you need?",
    "dietary": "Any dietary preferences? (vegetarian, vegan, keto, gluten-free)",
    "special_goal": "Do you have any special goals? (healthy, gourmet, quick, special occasion)",
    "format": "How would you like the recipe format? (step-by-step, compact, mixed)",
}

# --- Endpoint: /recipes ---
@router.post("/recipes")
async def recipes(request: RecipeRequest, db: Session = Depends(get_db)):
    # Get or create session
    session = (
        db.query(RecipeSession)
        .filter(RecipeSession.session_id == request.session_id)
        .first()
    )
    if not session:
        session = RecipeSession(session_id=request.session_id, answers={})
        db.add(session)
        db.commit()
        db.refresh(session)

    if not isinstance(session.answers, dict):
        session.answers = {}

    user_message = request.input_text.strip()

    # --- Check for existing mode first ---
    current_mode = session.answers.get("mode")

    # --- Step 1: Mode selection (only if mode is not set) ---
    if not current_mode:
        if user_message.lower() in ["one-by-one", "one by one", "one"]:
            updated_answers = dict(session.answers)
            updated_answers["mode"] = "one-by-one"
            session.answers = updated_answers
            db.commit()
            db.refresh(session)

            # Return the first question
            next_field = list(QUESTIONS_MAP.keys())[0]
            updated_answers = dict(session.answers)
            updated_answers["last_field"] = next_field
            session.answers = updated_answers
            db.commit()
            db.refresh(session)
            return {
                "status": "question",
                "question": QUESTIONS_MAP[next_field],
                "field": next_field,
            }
        elif user_message.lower() in ["all-at-once", "all at once", "all"]:
            updated_answers = dict(session.answers)
            updated_answers["mode"] = "all-at-once"
            session.answers = updated_answers
            db.commit()
            db.refresh(session)
            return {
                "status": "success",
                "response": "Perfect 👍 Please tell me everything at once: your dish, cuisine style, servings, dietary needs, ingredients to include/avoid, equipment, and cooking time."
            }
        else:
            return {
                "status": "question",
                "question": "Would you like me to ask questions one-by-one, or all-at-once?",
                "options": ["one-by-one", "all-at-once"]
            }

    # --- Step 2: One-by-one mode logic ---
    if current_mode == "one-by-one":
        last_field = session.answers.get("last_field")

        # Save the answer for the last asked field
        if last_field and last_field in QUESTIONS_MAP:
            updated_answers = dict(session.answers)
            updated_answers[last_field] = user_message
            session.answers = updated_answers
            db.commit()
            db.refresh(session)

        # Find the next missing field
        required_fields = list(QUESTIONS_MAP.keys())
        missing_fields = [f for f in required_fields if f not in session.answers]

        if missing_fields:
            next_field = missing_fields[0]
            updated_answers = dict(session.answers)
            updated_answers["last_field"] = next_field
            session.answers = updated_answers
            db.commit()
            db.refresh(session)
            return {
                "status": "question",
                "question": QUESTIONS_MAP[next_field],
                "field": next_field,
            }

        # All fields answered → generate recipe
        conversation = [
            {"role": "system", "content": (
                "You are a helpful recipe assistant. "
                "Your ONLY output must be valid JSON, with no explanations or extra text. "
                "Use exactly this structure:\n\n"
                "{\n"
                '  \"dish\": \"\",\n'
                '  \"ingredients\": [ {\"name\": \"\", \"quantity\": \"\"} ],\n'
                '  \"steps\": [ \"\" ],\n'
                '  \"variations\": \"\",\n'
                '  \"plating_tips\": \"\",\n'
                '  \"storage_tips\": \"\"\n'
                "}\n\n"
                f"User answers: {session.answers}"
            )},
            {"role": "user", "content": "Please generate the recipe now."}
        ]
        recipe_json = call_llm(conversation)
        return {"status": "success", "recipe": recipe_json}

    # --- Step 3: All-at-once mode logic ---
    if current_mode == "all-at-once":
        updated_answers = dict(session.answers)
        updated_answers["last_input"] = user_message
        session.answers = updated_answers
        db.commit()
        db.refresh(session)

        conversation = [
            {"role": "system", "content": (
                "You are a helpful recipe assistant. "
                "Your ONLY output must be valid JSON, with no explanations or extra text. "
                "Use exactly this structure:\n\n"
                "{\n"
                '  \"dish\": \"\",\n'
                '  \"ingredients\": [ {\"name\": \"\", \"quantity\": \"\"} ],\n'
                '  \"steps\": [ \"\" ],\n'
                '  \"variations\": \"\",\n'
                '  \"plating_tips\": \"\",\n'
                '  \"storage_tips\": \"\"\n'
                "}\n\n"
                "Do not include Markdown, headings, or extra formatting — only valid JSON."
            )},
            {"role": "user", "content": user_message},
        ]

        recipe_json = call_llm(conversation)
        return {"status": "success", "recipe": recipe_json}