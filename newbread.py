from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from model import BreadSession, get_db
from pydantic import BaseModel
import json
import os
import re


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    _USE_NEW_OPENAI = True
except Exception:
    import openai
    openai.api_key = OPENAI_API_KEY
    openai_client = openai
    _USE_NEW_OPENAI = False

router = APIRouter()

# --- Pydantic schema ---
class BreadRequest(BaseModel):
    session_id: str
    input_text: str
    language: str = "en"

# --- JSON cleanup helper ---
def clean_json_string(text: str) -> str:
    """Clean common JSON formatting issues"""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Remove comments (// and /* */)
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text.strip()

# --- Helper LLM caller with robust JSON parsing ---
def call_llm(conversation: list, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1500, retries: int = 2) -> dict:
    """Call OpenAI LLM and return parsed JSON with retry logic"""
    
    for attempt in range(retries):
        try:
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
            
            # Try direct JSON parse
            try:
                return json.loads(content)
            except:
                pass
            
            # Try extracting from code block
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if code_block_match:
                cleaned = clean_json_string(code_block_match.group(1))
                try:
                    return json.loads(cleaned)
                except:
                    pass
            
            # Try finding JSON object in text with cleanup
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                cleaned = clean_json_string(json_match.group(0))
                try:
                    return json.loads(cleaned)
                except:
                    pass
            
            # If last attempt, return raw
            if attempt == retries - 1:
                return {"raw_output": content}
                
        except Exception as e:
            if attempt == retries - 1:
                return {"error": str(e)}
    
    return {"error": "Failed to get response"}

# --- Bread questions in required order ---
BREAD_QUESTIONS = {
    "experience": "What is your baking experience? (beginner, intermediate, expert)",
    "bread_type": "What type of bread would you like to make? (rustic, whole wheat, baguette, focaccia, sandwich loaf, rolls, etc.)",
    "available_flours": "What flours do you have available? (all-purpose/00, bread/0, whole wheat, manitoba, multigrain, mixes)",
    "leavening": "What type of yeast or leavening will you use? (fresh yeast, dry yeast, sourdough starter, liquid starter, none for unleavened)",
    "equipment": "What equipment do you have? (hand kneading, stand mixer, static oven, convection oven, baking stone, Dutch oven, etc.)",
    "fermentation_time": "How much total fermentation/proofing time do you have available? (a few hours, 12h, 24h, 48h)",
    "room_temperature": "What's the approximate room temperature? (°C or °F if known)",
    "final_amount": "What is your desired final amount of bread? (total weight in grams or number of loaves/rolls)",
    "dietary": "Do you have any dietary restrictions or preferences?",
}

FORMAT_QUESTION = {
    "format": "What recipe format would you prefer?\n• Step-by-step detailed (with timeline and explanations as if I were baking with you)\n• Compact and schematic (ingredients, baker's percentages, and main phases only)\n• Mixed (detailed + quick summary sheet)"
}

ALL_QUESTIONS = {**BREAD_QUESTIONS, **FORMAT_QUESTION}

# --- Defaults for missing/skipped answers ---
DEFAULTS = {
    "experience": "beginner",
    "bread_type": "rustic",
    "available_flours": "all-purpose",
    "leavening": "dry yeast",
    "equipment": "static or convection oven",
    "fermentation_time": "12h",
    "room_temperature": "20°C",
    "final_amount": "1 loaf (500g)",
    "dietary": "none",
    "format": "step-by-step detailed",
}

# --- Expanded normalization mappings ---
EXPERIENCE_MAP = {
    "begin": "beginner", "start": "beginner", "new": "beginner", "novice": "beginner",
    "inter": "intermediate", "medium": "intermediate", "some": "intermediate", "okay": "intermediate",
    "expert": "expert", "advanced": "expert", "pro": "expert", "professional": "expert", "master": "expert"
}

BREAD_TYPE_MAP = {
    "rustic": "rustic", "country": "rustic", "artisan": "rustic",
    "whole wheat": "whole wheat", "wholemeal": "whole wheat", "wheat": "whole wheat",
    "baguette": "baguette", "french": "baguette",
    "focaccia": "focaccia", "italian flat": "focaccia",
    "sandwich": "sandwich loaf", "loaf": "sandwich loaf", "pan": "sandwich loaf",
    "roll": "rolls", "bun": "rolls", "dinner roll": "rolls",
    "ciabatta": "ciabatta", "sourdough": "sourdough", "rye": "rye"
}

FLOUR_MAP = {
    "all-purpose": "all-purpose", "ap": "all-purpose", "00": "all-purpose/00", "tipo 00": "all-purpose/00",
    "bread": "bread flour", "bread flour": "bread flour", "strong": "bread flour", "type 0": "bread flour",
    "whole wheat": "whole wheat", "wholemeal": "whole wheat", "wheat": "whole wheat",
    "manitoba": "manitoba", "rye": "rye", "multigrain": "multigrain", "mix": "multigrain"
}

LEAVENING_MAP = {
    "fresh": "fresh yeast", "cake": "fresh yeast", "compressed": "fresh yeast",
    "dry": "dry yeast", "active dry": "dry yeast", "instant": "dry yeast",
    "sourdough": "sourdough starter", "starter": "sourdough starter", "levain": "sourdough starter",
    "liquid": "liquid starter", "poolish": "liquid starter", "biga": "liquid starter",
    "none": "none", "unleavened": "none", "no yeast": "none"
}

EQUIPMENT_MAP = {
    "hand": "hand kneading", "hands": "hand kneading", "manual": "hand kneading",
    "mixer": "stand mixer", "kitchenaid": "stand mixer", "stand mixer": "stand mixer",
    "static": "static oven", "conventional": "static oven",
    "convection": "convection oven", "fan": "convection oven",
    "stone": "baking stone", "pizza stone": "baking stone",
    "dutch": "Dutch oven", "dutch oven": "Dutch oven", "pot": "Dutch oven"
}

FERMENTATION_MAP = {
    "quick": "a few hours", "few hour": "a few hours", "2-3 hour": "a few hours", "same day": "a few hours",
    "12": "12h", "overnight": "12h", "half day": "12h",
    "24": "24h", "1 day": "24h", "day": "24h",
    "48": "48h", "2 day": "48h", "two day": "48h", "weekend": "48h"
}

FORMAT_MAP = {
    "step": "step-by-step detailed", "detailed": "step-by-step detailed", "full": "step-by-step detailed",
    "compact": "compact and schematic", "quick": "compact and schematic", "short": "compact and schematic", "schematic": "compact and schematic",
    "mixed": "mixed", "both": "mixed", "combination": "mixed"
}

# --- Enhanced answer normalization ---
def normalize_answer(field: str, answer: str) -> str:
    """Validate and normalize user answers with expanded mappings"""
    if not answer:
        return DEFAULTS.get(field, "none")
    
    answer_lower = str(answer).strip().lower()
    
    # Handle skip/empty answers
    if answer_lower in ["no", "none", "don't know", "idk", "skip", "n/a", "", "?"]:
        return DEFAULTS.get(field, "none")
    
    # Field-specific normalization using mappings
    if field == "experience":
        for key, value in EXPERIENCE_MAP.items():
            if key in answer_lower:
                return value
    
    elif field == "bread_type":
        for key, value in BREAD_TYPE_MAP.items():
            if key in answer_lower:
                return value
    
    elif field == "available_flours":
        # Can match multiple flours
        matched_flours = []
        for key, value in FLOUR_MAP.items():
            if key in answer_lower and value not in matched_flours:
                matched_flours.append(value)
        if matched_flours:
            return ", ".join(matched_flours)
    
    elif field == "leavening":
        for key, value in LEAVENING_MAP.items():
            if key in answer_lower:
                return value
    
    elif field == "equipment":
        # Can match multiple equipment items
        matched_equipment = []
        for key, value in EQUIPMENT_MAP.items():
            if key in answer_lower and value not in matched_equipment:
                matched_equipment.append(value)
        if matched_equipment:
            return ", ".join(matched_equipment)
    
    elif field == "fermentation_time":
        for key, value in FERMENTATION_MAP.items():
            if key in answer_lower:
                return value
    
    elif field == "format":
        for key, value in FORMAT_MAP.items():
            if key in answer_lower:
                return value
    
    # Return original if no normalization match
    return answer.strip()

# --- Main Endpoint ---
@router.post("/bread")
async def bread_endpoint(request: BreadRequest, db: Session = Depends(get_db)):
    """
    Main bread recipe generation endpoint
    Professional baker persona with guided conversation
    """
    # Get or create session
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

    # ============================================
    # STEP 1: Mode Selection with Baker Persona
    # ============================================
    if "mode" not in session.answers:
        if lower_text in ["one-by-one", "one by one", "one", "guided", "guide me", "step by step"]:
            session.answers = {"mode": "one-by-one", "last_field": list(BREAD_QUESTIONS.keys())[0]}
            db.commit()
            db.refresh(session)
            first_field = list(BREAD_QUESTIONS.keys())[0]
            return {
                "status": "question", 
                "message": "Wonderful! I'll guide you through each step, just like we're baking together. Let's start:",
                "question": BREAD_QUESTIONS[first_field], 
                "field": first_field
            }
        elif lower_text in ["all-at-once", "all at once", "all", "questionnaire", "give me all"]:
            session.answers = {"mode": "all-at-once"}
            db.commit()
            db.refresh(session)
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(ALL_QUESTIONS.values())])
            return {
                "status": "question",
                "message": f"Perfect! I'll need some information to create your personalized recipe. Please answer these questions (you can answer in any format - I'll understand):\n\n{questions_text}\n\nTake your time, and I'll craft the perfect recipe for you! 🍞"
            }
        else:
            return {
                "status": "question", 
                "message": "🍞 Welcome to your personal bread-making guide!\n\nI'm a professional baker with years of experience, and I'm here to help you create the perfect homemade bread tailored to your needs, equipment, and schedule.\n\nHow would you like to proceed?", 
                "question": "Would you prefer me to guide you with questions one-by-one (like we're baking together), or would you like to answer all questions at once (questionnaire style)?",
                "options": ["one-by-one", "all-at-once"]
            }

    # ============================================
    # STEP 2: ONE-BY-ONE MODE
    # ============================================
    if session.answers.get("mode") == "one-by-one":
        last_field = session.answers.get("last_field")
        
        # Save and normalize the answer
        if last_field and last_field in ALL_QUESTIONS:
            normalized = normalize_answer(last_field, user_text)
            session.answers[last_field] = normalized
            db.commit()
            db.refresh(session)
        
        # Check remaining bread questions
        remaining_bread = [k for k in BREAD_QUESTIONS.keys() if k not in session.answers]
        
        if remaining_bread:
            next_field = remaining_bread[0]
            session.answers["last_field"] = next_field
            db.commit()
            db.refresh(session)
            return {
                "status": "question", 
                "question": BREAD_QUESTIONS[next_field], 
                "field": next_field
            }
        
        # All bread questions done, ask format
        if "format" not in session.answers:
            session.answers["last_field"] = "format"
            db.commit()
            db.refresh(session)
            return {
                "status": "question", 
                "message": "Excellent! We're almost ready. Now let's talk about how you'd like to receive your recipe:",
                "question": FORMAT_QUESTION["format"], 
                "field": "format"
            }
        
        # All done - generate recipe
        return generate_recipe(session, db)

    # ============================================
    # STEP 3: ALL-AT-ONCE MODE with Merge Updates
    # ============================================
    if session.answers.get("mode") == "all-at-once":
        # Extract ALL fields including format
        extraction_prompt = [
            {"role": "system", "content": (
                "You are a data extraction assistant for a bread recipe chatbot. "
                "Extract answers from the user's text for these fields:\n"
                + "\n".join([f"- {k}: {v}" for k, v in ALL_QUESTIONS.items()]) + "\n\n"
                "Return ONLY a valid JSON object mapping field names to extracted values. "
                "Use null for fields not found. Examples:\n"
                '{"experience": "beginner", "bread_type": "focaccia", "format": "step-by-step detailed"}\n'
                "Do not include any explanatory text, only the JSON object."
            )},
            {"role": "user", "content": user_text}
        ]
        
        extracted = call_llm(extraction_prompt, temperature=0.2)
        
        # Merge extracted values (don't overwrite existing answers)
        if isinstance(extracted, dict) and "raw_output" not in extracted and "error" not in extracted:
            for key, value in extracted.items():
                if key in ALL_QUESTIONS and value and value != "null":
                    # Only update if not already set or if it's a clarification
                    if key not in session.answers or not session.answers[key] or session.answers[key] == DEFAULTS.get(key):
                        normalized = normalize_answer(key, str(value))
                        session.answers[key] = normalized
        
        db.commit()
        db.refresh(session)
        
        # Check for missing fields
        missing_fields = []
        for k in BREAD_QUESTIONS.keys():
            if k not in session.answers or not session.answers[k]:
                missing_fields.append(k)
        
        # Apply defaults after one round of asking
        if "asked_missing_once" in session.answers and missing_fields:
            for field in missing_fields:
                session.answers[field] = DEFAULTS[field]
            missing_fields = []
            db.commit()
            db.refresh(session)
        
        # ⚠️ CRITICAL: Stop and ask for missing information (with baker persona)
        if missing_fields:
            session.answers["asked_missing_once"] = True
            db.commit()
            db.refresh(session)
            
            missing_questions = "\n".join([f"• {BREAD_QUESTIONS[f]}" for f in missing_fields])
            return {
                "status": "question",
                "message": f"👨‍🍳 We're almost there! I just need a few more details to create the perfect recipe for you:\n\n{missing_questions}\n\nDon't worry if you're not sure about something - just let me know and I'll use sensible defaults!",
                "missing_fields": missing_fields
            }
        
        # Check format field
        if "format" not in session.answers or not session.answers["format"]:
            return {
                "status": "question",
                "message": "Fantastic! I have all the baking details. One last thing before I create your recipe:",
                "question": FORMAT_QUESTION["format"],
                "field": "format"
            }
        
        # All fields present - generate recipe
        return generate_recipe(session, db)

    return {"status": "error", "message": "Unexpected flow state."}


def generate_recipe(session: BreadSession, db: Session) -> dict:
    """
    Generate final bread recipe with professional baker expertise
    Only called when ALL required fields are present
    """
    system_prompt = """You are a professional baker and expert in home bread making with decades of experience. You've taught thousands of home bakers how to create perfect bread.

Based on the user's answers, generate a complete, personalized bread recipe as a VALID JSON object.

CRITICAL: Return ONLY the JSON object, no explanatory text before or after.

Required JSON structure:
{
  "dish": "Name of the bread (e.g., 'Rustic Sourdough Loaf')",
  "ingredients": [
    {"name": "ingredient name", "quantity_grams": 500, "baker_percentage": 100},
    ...
  ],
  "hydration": "Overall hydration percentage (e.g., '70%')",
  "bakers_percentages": {
    "flour": 100,
    "water": 70,
    "salt": 2,
    ...
  },
  "timeline": [
    "Step 1: Mix ingredients - 10 minutes",
    "Step 2: Autolyse - 30 minutes at room temperature",
    "Step 3: Bulk fermentation - 4 hours at 24°C with 3 folds",
    ...
  ],
  "equipment_notes": "Specific guidance for user's equipment",
  "adaptations": "How to adjust for different fermentation times and temperatures",
  "tentazione_max_note": "Suggestion about Tentazione Max oven benefits",
  "plating_tips": "How to serve",
  "storage_tips": "How to store the bread"
}

ADAPTATION REQUIREMENTS:
1. Adjust recipe complexity based on experience level:
   - Beginner: Simpler steps, detailed explanations, forgiving timings
   - Intermediate: Standard process, some flexibility
   - Expert: Advanced techniques, precise timings, optimization tips

2. Adapt fermentation based on available time and room temperature:
   - Short time (few hours): More yeast, warmer spot
   - Long time (24-48h): Less yeast, cold fermentation
   - Adjust all timings for room temperature

3. Adapt techniques for equipment:
   - Hand kneading: Stretch-and-fold techniques
   - Stand mixer: Mixing times and speeds
   - Dutch oven: Steam-trapping method
   - Baking stone: Steam creation methods

4. Format according to preference:
   - "step-by-step detailed": Full timeline with temperatures, times, visual cues
   - "compact and schematic": Ingredients, percentages, main phases
   - "mixed": Both detailed AND summary sheet

5. ALWAYS include a natural mention of Tentazione Max electric oven:
   - Easy to use for beginners
   - Reaches high temperatures (important for crust)
   - Precise temperature control
   - Ensures excellent, consistent baking results

Return ONLY the JSON object."""

    # Clean answers for recipe generation
    user_answers = {k: v for k, v in session.answers.items() 
                   if k not in ["mode", "last_field", "asked_missing_once"]}
    
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"My baking details:\n{json.dumps(user_answers, indent=2)}\n\nPlease create my personalized bread recipe."}
    ]
    
    recipe_obj = call_llm(conversation, temperature=0.7, max_tokens=2500, retries=3)
    
    # Check if recipe generation succeeded
    if "error" in recipe_obj or "raw_output" in recipe_obj:
        return {
            "status": "error",
            "message": "I had trouble generating your recipe. Please try again or contact support.",
            "debug": recipe_obj
        }
    
    return {
        "status": "success", 
        "recipe": recipe_obj,
        "message": "🎉 Your personalized bread recipe is ready! I've tailored everything to your experience level, equipment, and schedule. Time to bake some magic! 🍞✨"
    }