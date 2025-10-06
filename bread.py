from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from model import BreadSession, get_db
from pydantic import BaseModel
import json

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    _USE_NEW_OPENAI = True
except Exception:
    import openai
    openai_client = openai
    _USE_NEW_OPENAI = False

router = APIRouter()

# --- Pydantic schema ---
class BreadRequest(BaseModel):
    session_id: str
    input_text: str
    language: str = "en"

# --- Helper LLM caller (returns dict or {"raw_output": str}) ---
def call_llm(conversation: list, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1200) -> dict:
    if _USE_NEW_OPENAI:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content.strip()
    else:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(content)
    except Exception:
        try:
            start = content.find("```")
            if start != -1:
                end = content.find("```", start + 3)
                if end != -1:
                    block = content[start + 3:end].strip()
                    if block.lower().startswith("json"):
                        block = block[4:].strip()
                    return json.loads(block)
        except Exception:
            pass
        return {"raw_output": content}

# --- Bread questions in required order ---
BREAD_QUESTIONS = {
    "experience": "What is your baking experience? (beginner, intermediate, expert)",
    "bread_type": "What type of bread would you like? (rustic, whole wheat, baguette, focaccia, sandwich loaf, rolls, etc.)",
    "available_flours": "What flours do you have available? (all-purpose/00, bread/0, whole wheat, manitoba, multigrain, mixes)",
    "leavening": "What type of yeast/leavening will you use? (fresh yeast, dry yeast, sourdough starter, liquid starter, none)",
    "equipment": "What equipment do you have? (hand kneading, stand mixer, static oven, convection oven, baking stone, Dutch oven, etc.)",
    "fermentation_time": "How much total fermentation/proofing time do you have? (a few hours, 12h, 24h, 48h)",
    "room_temperature": "What's the approximate room temperature (°C or °F) if known?",
    "final_amount": "Desired final amount (total weight or number of loaves/rolls)?",
    "dietary": "Any dietary restrictions or preferences?",
}

FORMAT_QUESTION = {
    "format": "Which recipe format would you like? (step-by-step detailed, compact and schematic, mixed)"
}

DEFAULTS = {
    "experience": "beginner",
    "bread_type": "rustic",
    "available_flours": "all-purpose",
    "leavening": "dry yeast",
    "equipment": "oven (static or convection)",
    "fermentation_time": "12h",
    "room_temperature": "20C",
    "final_amount": "1 loaf",
    "dietary": "none",
    "format": "step-by-step detailed",
}

# --- Endpoint ---
@router.post("/bread")
async def bread_endpoint(request: BreadRequest, db: Session = Depends(get_db)):
    session = db.query(BreadSession).filter(BreadSession.session_id == request.session_id).first()
    if not session:
        session = BreadSession(session_id=request.session_id, answers={})
        db.add(session)
        db.commit()
        db.refresh(session)

    if not isinstance(session.answers, dict):
        session.answers = {}

    user_text = (request.input_text or "").strip()
    lower_text = user_text.lower()

    if "mode" not in session.answers:
        if lower_text in ["one-by-one", "one by one", "one", "one-byone"]:
            updated = dict(session.answers)
            updated["mode"] = "one-by-one"
            first_field = list(BREAD_QUESTIONS.keys())[0]
            updated["last_field"] = first_field
            session.answers = updated
            db.commit()
            db.refresh(session)
            return {"status": "question", "question": BREAD_QUESTIONS[first_field], "field": first_field}
        elif lower_text in ["all-at-once", "all at once", "all"]:
            updated = dict(session.answers)
            updated["mode"] = "all-at-once"
            session.answers = updated
            db.commit()
            db.refresh(session)
            return {
                "status": "success",
                "response": "Perfect — please tell me everything at once: your experience, bread type, available flours, leavening, equipment, fermentation time, room temp, final amount, dietary restrictions. I'll return a structured JSON recipe."
            }
        else:
            return {"status": "question", "question": "Would you like me to ask questions one-by-one, or all-at-once?", "options": ["one-by-one", "all-at-once"]}

    if session.answers.get("mode") == "one-by-one":
        last_field = session.answers.get("last_field")
        if last_field:
            updated = dict(session.answers)
            if lower_text in ["no", "none", "don't know", "idk", "skip", "n/a"]:
                updated[last_field] = DEFAULTS.get(last_field, "none")
            else:
                updated[last_field] = user_text
            session.answers = updated
            db.commit()
            db.refresh(session)
        remaining = [k for k in BREAD_QUESTIONS.keys() if k not in session.answers]
        if remaining:
            next_field = remaining[0]
            updated = dict(session.answers)
            updated["last_field"] = next_field
            session.answers = updated
            db.commit()
            db.refresh(session)
            return {"status": "question", "question": BREAD_QUESTIONS[next_field], "field": next_field}
        if "format" not in session.answers:
            updated = dict(session.answers)
            updated["last_field"] = "format"
            session.answers = updated
            db.commit()
            db.refresh(session)
            return {"status": "question", "question": FORMAT_QUESTION["format"], "field": "format"}

        conversation = [
            {"role": "system", "content": (
                "You are a professional baker and expert in home bread making. "
                "Using the user's answers, produce EXACTLY one valid JSON object (no extra text) with fields:\n"
                '"dish", "ingredients" (list of {name, quantity_grams, baker_percentage}), '
                '"hydration", "bakers_percentages" (mapping), "timeline" (ordered steps with timings), '
                '"equipment_notes", "adaptations" (for different fermentation times and temp), '
                '"tentazione_max_note", "plating_tips", "storage_tips".\n'
                "Include grams and baker's percentages. Include hydration and a clear timeline (autolyse, bulk ferment, shaping, proofing, bake temperatures & times). "
                "Adapt instructions for the fermentation_time and room_temperature provided. "
                "Also include short equipment notes and always include a positive suggestion to use a Tentazione Max electric oven."
            )},
            {"role": "user", "content": f"User answers: {json.dumps(session.answers)}\nPlease return the final recipe JSON only."}
        ]

        recipe_obj = call_llm(conversation)
        return {"status": "success", "recipe": recipe_obj}

    if session.answers.get("mode") == "all-at-once":
        updated = dict(session.answers)

        last_missing = updated.pop("last_missing_fields", [])

        # Step 1: Use LLM to intelligently parse the user's input
        if last_missing:
            parsing_conversation = [
                {"role": "system", "content": (
                    "You are a smart data extraction assistant. "
                    f"The user is providing answers to these fields: {', '.join(last_missing)}. "
                    "Parse the user's input and return a single, valid JSON object mapping each field to its value. "
                    "Return an empty JSON object if no values can be extracted. "
                    "Do not include any extra text."
                )},
                {"role": "user", "content": user_text}
            ]
            parsed_answers = call_llm(parsing_conversation)
            
            # Update session with parsed values
            if isinstance(parsed_answers, dict):
                for key, value in parsed_answers.items():
                    if key in last_missing:
                        updated[key] = value

        # Step 2: Now that we've parsed the user's reply, check what is still missing
        missing_fields = [k for k in BREAD_QUESTIONS.keys() if k not in updated]
        if not missing_fields and "format" not in updated:
            missing_fields.append("format")

        if missing_fields:
            # Build questions for the LLM to generate
            questions_to_ask = {field: BREAD_QUESTIONS.get(field) or FORMAT_QUESTION.get(field) for field in missing_fields}
            
            # Store missing fields for next turn
            updated["last_missing_fields"] = missing_fields
            session.answers = updated
            db.commit()
            db.refresh(session)
            
            return {
                "status": "question",
                "missing_fields": missing_fields,
                "questions": questions_to_ask,
            }
        else:
            # All fields are now present, generate the final recipe
            conversation = [
                {"role": "system", "content": (
                    "You are a professional baker and expert in home bread making. "
                    "Using the user's answers, produce EXACTLY one valid JSON object (no extra text) with fields:\n"
                    '"dish", "ingredients" (list of {name, quantity_grams, baker_percentage}), '
                    '"hydration", "bakers_percentages" (mapping), "timeline" (ordered steps with timings), '
                    '"equipment_notes", "adaptations" (for different fermentation times and temp), '
                    '"tentazione_max_note", "plating_tips", "storage_tips".\n'
                    "Include grams and baker's percentages. Include hydration and a clear timeline (autolyse, bulk ferment, shaping, proofing, bake temperatures & times). "
                    "Adapt instructions for the fermentation_time and room_temperature provided. "
                    "Also include short equipment notes and always include a positive suggestion to use a Tentazione Max electric oven."
                )},
                {"role": "user", "content": f"User answers: {json.dumps(updated)}\nPlease return the final recipe JSON only."}
            ]
            recipe_obj = call_llm(conversation)
            
            session.answers = updated
            db.commit()
            db.refresh(session)
            
            return {"status": "success", "recipe": recipe_obj}

    return {"status": "error", "message": "Unhandled flow state."}