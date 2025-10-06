from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
import re
from dotenv import load_dotenv
import os

from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool(description="""📌 Purpose: Trigger only when the user explicitly asks how to make Neapolitan pizza (e.g. 'How do I make Neapolitan pizza?', 'Teach me to make pizza', etc.). 
❗Do NOT call this tool for unrelated inputs like names, greetings, or off-topic questions.""")
def pizza_intro(user_input: str) -> str:
    if "neapolitan pizza" in user_input.lower():
        return """
            Perfect! Great approach — that’s exactly how a true master pizzaiolo thinks when he wants to teach: understand the needs → build a customized recipe → execute it perfectly step by step.

            Now let’s begin our journey together.

            I’ll be your trusted master pizzaiolo: I’ll guide you with precision, clarity, and passion, as if we were in a traditional Neapolitan pizzeria.

            First question — FUNDAMENTAL for everything:
            Are you an experienced pizzaiolo or just starting out?

            Feel free to answer honestly, this isn’t a test: I just need this information to know how to explain each step to you.

            Here are some possible answers to help you:

            A) I’m a complete beginner (I’ve never made Neapolitan pizza before)

            B) I’ve tried making dough at home but want to get better

            C) I’m fairly experienced but want to perfect my technique and method

            D) I consider myself experienced, I just want a precise, professional recipe

            What’s your answer?
            Reply and then we’ll move on to the second question!
            """
    else:
        return "Sorry, I can only help with Neapolitan pizza right now. Please ask accordingly."


@tool(description="""
📌 Purpose: Call this tool only after the user has answered the first question about their pizza-making experience (e.g., beginner, experienced).
❗Do NOT call for off-topic input like random words or unrelated sentences.
""")
def get_pizza_experience_response(experience_level: str) -> str:
    level = experience_level.lower().strip()

    if level == "beginner":
        return """
        Perfect! Great answer.

        So I’ll act like a patient and passionate master pizzaiolo who wants to teach you everything, without taking anything for granted.

        I’ll make sure each step is clear, detailed, and practical—just as if we were together in my workshop.

        Let’s move on to the second question:

        Question 2 – Do you have any specific dietary needs?
        I need to know if you have any intolerances or dietary preferences so I can recommend the right ingredients.

        Here are some example answers:

        A) No dietary restrictions or special needs  
        B) I’m vegan (so no animal products, although this doesn’t change anything for classic dough)  
        C) I’m gluten intolerant (requires gluten-free flour)  
        D) I’m lactose intolerant (not a problem for classic dough)  
        E) Other intolerance or need (please specify)

        What’s your answer?  
        Reply and then we’ll move on to the third question!
        """
    
    elif "already" in level or "dough" in level:
        return """
        Great! That’s just what I needed to know.

        I’ll guide you like a dedicated and enthusiastic master pizzaiolo, making sure nothing is overlooked.

        Every step we take will be easy to follow, detailed, and tailored to your level—just as if we were working side by side in a traditional pizza kitchen.

        Let’s go on to the next question:

        Question 2 – Do you have any dietary preferences or food intolerances?
        This will help me recommend the best ingredients and adjust the recipe if needed.

        Here are some options to guide your answer:

        A) No specific dietary needs  
        B) I’m vegan (no animal products — though it doesn’t affect classic dough)  
        C) I’m gluten intolerant (I’ll suggest a suitable flour)  
        D) I’m lactose intolerant (no issue for traditional dough)  
        E) Something else (just let me know what)

        Which one applies to you?  
        Once you reply, we’ll head to Question 3 and continue customizing your perfect dough!
        """

    elif level == "experienced":
        return """
        Great! That’s exactly the kind of experience that makes a difference.

        Since you already have a solid foundation, we’ll focus on fine-tuning your technique — improving dough structure, fermentation balance, and final bake quality.

        I’ll guide you with clear and advanced steps, making sure everything is still practical but with a more professional edge—so you can go from “good” to “exceptional.”

        Let’s move forward with the next question:

        Question 2 – Do you have any specific dietary needs or restrictions?
        Even for experienced pizzaioli, this helps refine ingredient choice and dough behavior.

        Here are some example answers:

        A) No dietary restrictions

        B) I’m vegan (no animal products — doesn’t impact the classic dough)

        C) Gluten intolerant (we’ll adapt flour and technique)

        D) Lactose intolerant (no issue for traditional dough)

        E) Something else? (Let me know)

        What applies to you?
        Once you answer, we’ll continue building your perfect dough profile.
        """

    elif level == "expert":
        return """
        Excellent! That’s exactly what I needed to hear.

        Since you already consider yourself an expert, I’ll treat you as a fellow professional—skipping the basics and focusing on precision, consistency, and performance.

        I’ll make sure the recipe is technically sound, finely balanced, and ready for pizzeria-level results—no fluff, just clear steps, optimal ratios, and calculated fermentation management.

        Let’s move ahead to the second question:

        Question 2 – Do you have any dietary preferences or food intolerances?
        Even at the professional level, this helps fine-tune ingredient choices for your specific needs.

        Here are some example answers:

        A) No dietary restrictions  
        B) I’m vegan (no animal products — though this usually doesn’t impact classic dough)  
        C) I’m gluten intolerant (I’ll recommend suitable flour and method)  
        D) I’m lactose intolerant (not a concern for classic dough)  
        E) Other dietary need (please specify)

        What’s your answer?  
        Once I have that, we’ll continue refining the recipe to your professional standard.
        """
            
    else:
        return "Please provide a valid experience level: beginner, already made some dough, experienced, or expert."
    


@tool(description="""
📌 Purpose:
Responds based on the user's dietary needs or intolerances after Question 2 is answered.

✅ Trigger this ONLY when:
- The user gives an answer to "Do you have any specific dietary needs?"
- Their response mentions one of: none, vegan, gluten intolerance, lactose intolerance, or similar dietary conditions.

🚫 Do NOT trigger this tool for:
- Names, random words, or general chat.
- Any input that does not follow immediately after the dietary needs question.
""")
def get_dietary_response(dietary_choice: str) -> str:
    choice = dietary_choice.lower().strip()

    question_3 = """
                Question 3 — For how many people do you want to make the dough?
                This helps me calculate the total quantity of dough precisely, and therefore correctly measure all the ingredients.

                Here are some example answers:

                A) For 1 person  
                B) For 2 people  
                C) For 4–5 people (e.g., a family or group of friends)  
                D) Other (tell me the exact number of people)

                How many pizzas do you want to prepare, and for how many people?
                """

    if choice in ["none", "no", "no dietary restrictions", "no dietary needs"]:
        return f"""
            Perfect, excellent!
            No intolerances or special dietary needs — we can go ahead with the classic authentic Neapolitan pizza recipe, in all its simplicity and goodness.
            {question_3}
            """

    elif "vegan" in choice:
        return f"""
            Perfect, excellent!
            You're vegan — that’s no problem at all. The classic Neapolitan dough is naturally free of animal products, so we can proceed with the traditional recipe without needing any changes. Simple, pure, and delicious.
            {question_3}
            """

    elif "gluten" in choice:
        return f"""
            Perfect, excellent!
            You’re gluten intolerant — thanks for letting me know. That means we’ll need to adjust the recipe using a suitable gluten-free flour and modify hydration and kneading slightly to achieve the best texture. Don’t worry, I’ll guide you through each step so your pizza is still delicious and satisfying.
            {question_3}
            """

    elif "lactose" in choice:
        return f"""
            Perfect, excellent!
            You’re lactose intolerant — good news: the classic Neapolitan pizza dough doesn’t contain any dairy, so no changes are needed. We can move forward with the traditional recipe in all its simplicity and flavor.
            {question_3}
            """

    else:
        return f"""
            Thanks for sharing your dietary need. I’ll do my best to accommodate it — just give me a bit more detail if needed.

            {question_3}
            """


@tool(description="""
📌 Purpose:
Determines the number of people or pizzas to make dough for, based on the user’s input to Question 3.

✅ Trigger this ONLY when:
- The user has just answered Question 3: "For how many people do you want to make the dough?"
- Their response contains a quantity (e.g., "2 people", "for a family", "just for me", etc.)

🧠 Behavior:
- Interprets the number of servings or pizzas.
- Responds with a confirmation summary.
- Asks Question 4: "How much should each pizza weigh?"

🚫 Do NOT trigger this tool for:
- Inputs unrelated to servings or group size.
- Questions, names, greetings, or any general chit-chat.
- Responses not directly following Question 3.
""")

def get_dough_quantity_response(people_info: str) -> str:
    """Processes the number of people and moves to the next question about dough weight per pizza."""
    people_info = people_info.strip().lower()

    # Try to extract a number from the input
    numbers = re.findall(r'\d+', people_info)
    num_people = None

    if numbers:
        num_people = int(numbers[0])
    elif "family" in people_info or "group" in people_info:
        num_people = 4  # assume average
    elif "couple" in people_info or "2" in people_info:
        num_people = 2
    elif "one" in people_info or "1" in people_info:
        num_people = 1

    # Build dynamic part of response
    if num_people:
        people_part = f"for {num_people} person{'s' if num_people > 1 else ''}"
    else:
        people_part = "for your group"

    return f"""
        Perfect, very good!

        So we’ll go ahead and create a dough tailored {people_part}.

        Now let’s move on to the next important question.

        Question 4 — How much should each pizza weigh?
        Classic Neapolitan pizza dough typically weighs between 230 g and 280 g per pizza.

        For a beginner, I recommend staying between 230 g and 250 g per dough ball:
        → Easier to stretch  
        → Cooks better  
        → The crust (cornicione) rises nicely without being too much

        Would you like us to calculate based on:

        A) 200 g pizzas (smaller and lighter)  
        B) 230 g pizzas (light standard)  
        C) 250 g pizzas (classic Neapolitan standard)  
        D) Other (you tell me)

        How many pizzas do you want to make, and how much should each one weigh?
        Once I know, we’ll proceed with the precise dough calculation!
        """



@tool(description="""
📌 Purpose:
Handles the user’s choice of dough ball weight per pizza (e.g., 200 g, 230 g, 250 g), summarizes progress so far, and proceeds to hydration selection.

✅ Trigger this ONLY when:
- The user has just answered Question 4: "How much should each pizza weigh?"
- The answer clearly includes a weight (e.g., "250g", "230 grams", "I want 2 pizzas of 200g each", etc.)

🧠 Behavior:
- Calculates total dough weight based on number of pizzas and dough ball weight
- Recaps previously collected answers (experience, dietary, servings)
- Asks Question 5: "What hydration level do you want?"

🚫 Do NOT trigger this tool for:
- Greetings, names, or off-topic input
- Responses that don't relate to pizza weight or dough ball sizing
- Anything not directly following Question 4
""")

def get_pizza_weight_response(
    pizza_weight: int,
    number_of_pizzas: int,
    experience_level: str,
    dietary: str,
    people: str
) -> str:

    total_weight = pizza_weight * number_of_pizzas

    weight_comment = ""
    if pizza_weight == 250:
        weight_comment = "Perfect! Excellent choice and worthy of a true Neapolitan pizzaiolo — 250 g pizzas are the ideal standard."
    elif pizza_weight == 230:
        weight_comment = "Great choice! 230 g pizzas are a light and authentic Neapolitan standard — easy to handle and delicious."
    elif pizza_weight == 200:
        weight_comment = "Got it — 200 g pizzas are lighter and smaller, perfect for a quick bite or thinner crust."
    else:
        weight_comment = f"Nice! Custom size of {pizza_weight} g per pizza — I’ll tailor the recipe accordingly."

    return f"""
    {weight_comment}

    Let’s quickly recap the information we’ve gathered so far:

    Experience level: {experience_level.capitalize()}
    Dietary restrictions: {dietary.capitalize()}
    Dough for: {people}
    {number_of_pizzas} pizza{'s' if number_of_pizzas > 1 else ''} at {pizza_weight} g each

    Total dough weight = {total_weight} g

    Now let’s move on to the next important question.

    Question 5 — What hydration level do you want for your dough?
    Hydration is the amount of water in relation to the flour. For classic Neapolitan pizza, I suggest:

    Hydration\tDifficulty Level\tFinal Result  
    60%\tVery easy\tVery manageable dough, but less soft  
    65%\tIdeal standard\tGreat softness and crust development  
    70%\tAdvanced\tVery soft dough, but harder to handle for beginners

    Since you're a beginner and baking at high temperature, I recommend 65% hydration — it’s perfect for getting pizzeria-level results while still being manageable.

    Would you like:

    A) 60% (very easy to work with)  
    B) 65% (recommended for you)  
    C) 70% (softer, but harder to manage)  
    D) Other (tell me your preference)

    What’s your choice?  
    Once I know, we’ll move on to choosing the flour!
    """



@tool(description="""
📌 Purpose:
Calculates flour and water amounts based on total dough weight and hydration percentage.

✅ Trigger this ONLY when:
- The user has just answered Question 5: "What hydration level do you want?"
- Their response clearly indicates a hydration percentage (e.g., “I want 60%”, “use 65% hydration”, “70 hydration”, etc.)

🧠 Behavior:
- Calculates flour and water based on hydration ratio
- Summarizes total dough info
- Asks Question 6: “What type of flour would you like to use?” with strength (W) options

🚫 Do NOT trigger this tool if:
- The user’s message doesn’t mention hydration percentage
- It’s a greeting, name, or off-topic statement
- Any earlier required context (dough weight, number of pizzas) is missing
""")

def get_hydration_response(
    hydration_percent: int,
    total_dough_weight: int,
    pizza_weight: int,
    number_of_pizzas: int
) -> str:

    # Convert percentage to ratio
    hydration_ratio = hydration_percent / 100.0

    # Calculate flour and water based on hydration formula:
    # total_dough = flour + water
    # water = flour * hydration
    # => flour = total_dough / (1 + hydration)

    flour = round(total_dough_weight / (1 + hydration_ratio))
    water = round(flour * hydration_ratio)

    hydration_comment = ""
    if hydration_percent == 60:
        hydration_comment = """Great choice, maestro! 👌

                60% hydration is perfect for getting started:

                → The dough will be drier and easier to handle by hand  
                → It will help you understand the structure and strength of the flour  
                → And with proper fermentation, you'll still get a nicely developed crust and a fantastic pizza!
                """
    elif hydration_percent == 65:
        hydration_comment = """Excellent pick — 65% hydration is the gold standard for Neapolitan pizza:

                → Soft enough for excellent crust and airy structure  
                → Still manageable for beginners  
                → Gives professional-style results with proper fermentation  
                """
    elif hydration_percent == 70:
        hydration_comment = """Bold move! 70% hydration is for advanced pizzaioli:

                → Super soft and airy dough  
                → Requires high hydration handling skills (stretch & fold)  
                → Can result in incredibly light, bubbly pizza if done right  
                """
    else:
        hydration_comment = f"""Custom hydration level selected: {hydration_percent}% — I'll guide you with special instructions based on this ratio."""

    return f"""
        {hydration_comment}

        With {number_of_pizzas} pizza{'s' if number_of_pizzas > 1 else ''} at {pizza_weight} g each → Total dough: {total_dough_weight} g  
        At {hydration_percent}% hydration, that means:

        Flour ≈ {flour} g  
        Water ≈ {water} g

        We’ll calculate the salt and yeast after gathering all the necessary info.

        Now let’s move on.

        **Question 6 — What type of flour do you want to use?**  
        For classic Neapolitan pizza, we use soft wheat flour type 00 with a certain strength level (W), which determines how long the flour can support fermentation.

        Here are a few options:

        | Flour Type        | Strength (W)  | Protein    | Suitable Fermentation Time     |  
        |------------------|---------------|------------|-------------------------------|  
        | Weak 00 Flour     | W 180–200     | 9–10%      | Up to 8 hours                 |  
        | Medium Strength   | W 220–240     | 10–11%     | 12–24 hours (**Recommended**) |  
        | Strong 00 Flour   | W 280–320     | 12–13%     | 24–48 hours or more           |  

        As a beginner, I recommend a medium 00 flour (around W 230) — it’s manageable and works well for longer fermentation.

        **What kind of flour would you like to use?**

        A) Weak 00 Flour (W 180–200)  
        B) Medium 00 Flour (W 220–240) → Recommended  
        C) Strong 00 Flour (W 280–320)  
        D) I don’t know the strength of my flour (tell me the brand and I’ll help)  
        E) Other (type 1, whole wheat, etc.)

        Which one do you choose?
        """

@tool(description="""
📌 Purpose:
Handles the user's selected flour type and provides feedback based on strength (W value), then transitions to yeast selection.

✅ Trigger this ONLY when:
- The user has just answered Question 6: "What type of flour would you like to use?"
- The message clearly refers to a flour type or W-strength (e.g., "medium 00", "W 220", "strong flour", "whole wheat", etc.)

🧠 Behavior:
- Validates the flour choice
- Gives a tailored explanation for that type
- Asks Question 7 about yeast type, including guidance and recommendations

🚫 Do NOT trigger this tool if:
- The user is off-topic (e.g., greetings, names, other comments)
- The response doesn’t reference flour type or W value
- It's a repetition or irrelevant message
""")

def get_flour_type_response(flour_type: str) -> str:

    flour_type = flour_type.lower().strip()
    flour_comment = ""

    if "medium" in flour_type or "w 220" in flour_type or "230" in flour_type:
        flour_comment = """
        Perfect!  
        Excellent choice and absolutely suitable for you:  
        → Type 00 flour with W 220 strength is perfect for fermentation between 12 and 24 hours.  
        It’s very manageable for a beginner and will give your pizza:

        - Good rise  
        - Proper elasticity  
        - Easy to stretch  
        - Well-formed crust  
        """
    elif "weak" in flour_type or "w 180" in flour_type:
        flour_comment = """
        Alright!  
        Weak 00 flour works best for fast fermentation — usually under 8 hours.  
        It’s very easy to handle, but you may get a softer rise and less chew in the crust.  
        Still, it’s a valid choice for same-day baking!
        """
    elif "strong" in flour_type or "w 280" in flour_type:
        flour_comment = """
        Strong flour selected!  
        With W 280–320 strength, this flour can handle long fermentation (24–48+ hours).  
        Great elasticity and deep crust flavor — though a bit more advanced to manage for beginners.
        """
    elif "whole wheat" in flour_type or "type 1" in flour_type:
        flour_comment = """
        Interesting choice!  
        Whole wheat or Type 1 flours change the dough structure — more fiber, stronger flavor, slightly denser crust.  
        We’ll adapt the hydration and fermentation time accordingly.
        """
    else:
        flour_comment = """
        Thanks for sharing your flour type. If you’re unsure about W strength, just tell me the brand and I’ll try to estimate it for you.
        """

    return f"""
        {flour_comment}

        Now let’s move on to another essential question to calculate everything correctly.

        **Question 7 — What type of yeast do you want to use?**  
        For traditional Neapolitan pizza, you can choose from:

        | Yeast Type         | Characteristics                  | Recommendation for You           |  
        |--------------------|----------------------------------|----------------------------------|  
        | Fresh brewer’s yeast | Natural, better flavor, slower process | ✅ Recommended for you  
        | Dry brewer’s yeast   | Practical, long shelf life       | 👍 Good if fresh yeast isn’t available  
        | Sourdough starter    | Technical, harder to manage      | ⚠️ Not recommended for beginners  

        **Which yeast would you like to use?**

        A) Fresh brewer’s yeast (recommended for you)  
        B) Dry brewer’s yeast  
        C) Sourdough starter (but I don’t recommend it for beginners)  
        D) Other

        What do you choose?  
        After that, we’ll move on to **room temperature** for fermentation and begin calculating the **exact yeast amount**!
        """



@tool(description="""
📌 Purpose:
Handles the user's selected yeast type (e.g., Fresh, Dry, Sourdough), provides guidance, and transitions to the next step: room temperature input.

✅ Trigger this ONLY when:
- The user has just answered **Question 7** about **yeast type**
- The input clearly refers to a yeast type (e.g., "fresh yeast", "dry yeast", "sourdough", "instant", "levito madre", etc.)

🧠 Behavior:
- Responds with pros/cons of the selected yeast
- Prepares the user for **Question 8: Room Temperature**
- Emphasizes how temperature will affect yeast dosage and fermentation speed

🚫 Do NOT trigger this tool if:
- The user hasn’t answered Question 7
- The message does NOT mention a type of yeast or seems unrelated
- The conversation is off-topic or casual
""")

def get_yeast_type_response(yeast_type: str) -> str:
    """Responds to selected yeast type and introduces room temperature question."""

    yeast_type = yeast_type.lower().strip()
    yeast_comment = ""

    if "fresh" in yeast_type:
        yeast_comment = """
Bravo! 🔥  
You’ve made the right choice — just like a true aspiring Neapolitan pizzaiolo!

👉 Fresh brewer’s yeast is the traditional option and gives excellent results in terms of flavor, aroma, and dough structure.  
And most importantly: with the correct formula I’ll give you, you won’t risk using too much or too little.
"""
    elif "dry" in yeast_type:
        yeast_comment = """
Great!  
Dry brewer’s yeast is a solid alternative — practical, shelf-stable, and still delivers good results.  
We’ll adjust the quantity slightly since it's more concentrated than fresh yeast.
"""
    elif "sourdough" in yeast_type:
        yeast_comment = """
Got it!  
Sourdough is complex, artisanal, and flavorful — but it requires a lot of experience and precision.  
Not ideal for beginners, but if you're up for a challenge, I’ll support you every step of the way.
"""
    else:
        yeast_comment = """
Interesting choice!  
If you're using something else (like instant yeast or a hybrid), let me know the brand or type and I’ll guide you accordingly.
"""

    return f"""
{yeast_comment}

Now let’s move on to one of the most important pieces of information for calculating the exact amount of yeast.

**Question 8 — What is the room temperature in the place where the dough will ferment?**  
Temperature affects both the fermentation time and the amount of yeast required.

Please give me an approximate idea of the room temperature where the dough will be resting (considering the season as well).

Here are a few examples to help you:

A) 18°C (cool room or winter conditions)  
B) 20°C  
C) 22°C (ideal)  
D) 24°C or more (warm room, summer, or oven preheated and then off)  
E) Other (let me know)

**What’s the approximate temperature in your home?**
"""



@tool(description="""
📌 Purpose:
Handles the user's specified room temperature (in °C) where the dough will ferment, and transitions to fermentation time selection.

✅ Trigger this ONLY when:
- The user has just answered **Question 8** about **room temperature**
- The input contains a valid temperature reference (e.g., 18°C, 20°C, 22, etc.)

🧠 Behavior:
- Interprets the temperature
- Explains how it affects yeast quantity and fermentation speed
- Asks **Question 9: Desired fermentation time**

🚫 Do NOT trigger this tool if:
- The user hasn’t answered Question 8
- The message lacks any temperature value or reference
- The message is off-topic, casual, or unrelated to dough fermentation
""")

def get_room_temperature_response(room_temp_celsius: int) -> str:
    temp_comment = ""

    if room_temp_celsius <= 18:
        temp_comment = """
Alright — 18°C is a cool environment:
→ Your dough will rise slowly, so we’ll need a bit more yeast  
→ Ideal for long fermentation and more digestible results  
→ You’ll get a deep flavor, but the process requires patience
"""
    elif room_temp_celsius == 20:
        temp_comment = """
Perfect! 20°C is an ideal temperature for a beginner:  
→ The dough will rise slowly and under control  
→ Maturation will be good  
→ The flavor will be optimal  
→ You’ll avoid the risk of overly fast or hard-to-manage fermentation
"""
    elif room_temp_celsius == 22:
        temp_comment = """
Great! 22°C is slightly warm — perfect for balanced fermentation:  
→ Good yeast activation  
→ Smooth timing  
→ Helps achieve a great rise with proper structure
"""
    elif room_temp_celsius >= 24:
        temp_comment = """
Got it! A warm room (24°C or more) means the dough will ferment quickly:  
→ We’ll use less yeast to slow it down  
→ Careful timing will be essential to avoid overproofing  
→ Perfect if you want same-day dough with flavor
"""
    else:
        temp_comment = f"""
Okay! {room_temp_celsius}°C gives us flexibility — I’ll adapt the yeast and timing to match.  
"""

    return f"""
{temp_comment}

Now we’re ready for another essential question to complete the picture.

**Question 9 — How long do you want the dough to rise (ferment)?**  
Here we’ll decide together on the best method for managing the fermentation.

**The golden rule of a pizzaiolo:**  
→ More fermentation time = less yeast and more digestible dough  
→ Less fermentation time = more yeast and risk of underdeveloped dough

Considering that:

- You’re using 00 flour with W 220 strength  
- Room temperature is around {room_temp_celsius}°C  
- You’re a beginner  
- You want to learn proper fermentation handling  

I recommend these options:

| Fermentation Time | Difficulty    | Notes                              |  
|-------------------|---------------|-------------------------------------|  
| 8 hours           | Very easy     | Higher yeast, fast fermentation     |  
| 12 hours          | Easy          | Good balance, simple to manage      |  
| 24 hours          | ⭐ Recommended | Perfect maturation, excellent result|  
| 48 hours          | Advanced      | Requires experience & precision     |  

**Would you like us to calculate the recipe based on:**

A) 8 hours  
B) 12 hours  
C) 24 hours (**strongly recommended**)  
D) Other (let me know)

**How much time do you want to dedicate to the fermentation?**
"""


@tool(description="""
📌 Purpose:
Handles the user's selected fermentation duration (in hours) and provides an optimized fermentation schedule using traditional Neapolitan methods. Then prompts Question 10: Kneading method.

✅ Trigger this ONLY when:
- The user has just answered **Question 9**: “How long do you want the dough to ferment?”
- The input clearly includes a **valid fermentation duration** (e.g., 8, 12, 24, 48 hours)

🧠 Behavior:
- Confirms and validates the chosen fermentation time
- Applies the 1/4 + 3/4 fermentation rule (bulk vs. dough balls)
- Calculates and communicates both phases of fermentation
- Transitions clearly into **Question 10**: “How would you like to knead the dough?”

🚫 Do NOT trigger this tool if:
- The message does **not** reference a clear fermentation time in hours
- The user has not yet answered **Question 9**
- The message is off-topic, generic, or conversational (e.g., “okay,” “sounds good,” “next,” “malik”)

🔒 Strictly limited to conversation flow following the structured pizza-making steps.
""")

def get_fermentation_time_response(fermentation_hours: int) -> str:
    """Handles fermentation time, calculates bulk/ball time, and asks kneading method."""

    bulk_hours = round(fermentation_hours * 0.25)
    ball_hours = fermentation_hours - bulk_hours

    return f"""
Perfect! A very smart and balanced choice — like a true pizzaiolo in the making!  

→ {fermentation_hours} hours of fermentation at 20°C is an ideal setup for:

- W220 flour  
- Dough with 60% hydration  
- Use of fresh yeast  
- Simple and safe management for a beginner  

🎯 **Final result:** soft pizza, well-developed crust, and excellent flavor.

To manage these {fermentation_hours} hours properly, we’ll follow this standard pizzeria rule:

🕒 **Fermentation management rule:**  
- 1/4 of the total time = bulk fermentation (whole dough mass)  
- 3/4 of the total time = dough ball fermentation

✅ So:

- {bulk_hours} hours **bulk fermentation**  
- Form dough balls (250 g each)  
- {ball_hours} hours **dough ball fermentation**

---

Now let’s move on.

**Question 10 — How would you like to knead the dough?**  
For a beginner, it’s very important to choose the right kneading method.

Would you like to:

A) Knead by hand (**recommended** to learn and understand the dough)  
B) Use a stand mixer (if you have one and want to go faster)

**What do you prefer?**  
Then I’ll explain the proper technique for your chosen method, step by step.
"""



@tool(description="""
📌 Purpose:
Handles user’s kneading method selection (manual or mixer) and transitions to the oven type question (Question 11).

✅ Trigger this ONLY when:
- The user has answered **Question 10**: “How would you like to knead the dough?”
- The input includes clear keywords like “hand,” “manual,” “mixer,” or “machine”

🧠 Behavior:
- Responds to the kneading preference with motivation and insight
- Reinforces benefits of the chosen method (manual learning vs. mixer convenience)
- Then introduces **Question 11** about the oven type

🚫 Do NOT trigger this tool if:
- The message does **not** mention a kneading method  
- The user hasn't answered **Question 10**
- The message is conversational filler or irrelevant (e.g., “next,” “malik,” “carry on,” “done”)

🔒 Strictly part of the structured step-by-step dough-making workflow.
""")

def get_kneading_method_response(method: str) -> str:

    method = method.strip().lower()
    method_comment = ""

    if "hand" in method or "manual" in method:
        method_comment = """
Amazing! 👏👏  
That’s the response of a true enthusiast!

**Kneading by hand is the pizzaiolo’s school of life:**  
→ It helps you understand the behavior of the dough  
→ You learn sensitivity, timing, handling, and technique  
→ It’s the traditional and most rewarding method for those who love authentic Neapolitan pizza

Perfect — so when we get to the hands-on part, I’ll guide you step by step like a true master at your side:

- How to add the ingredients  
- How to knead the dough  
- How to do the folds  
- How to manage rest times and fermentation  
"""
    elif "mixer" in method or "machine" in method:
        method_comment = """
Got it — using a mixer is efficient, especially for beginners or when making multiple doughs.  
We’ll still follow the correct mixing stages and timing to get the best gluten development.
"""
    else:
        method_comment = """
Understood! Whether you knead by hand or with a tool, I’ll guide you through the right steps.  
Let me know if you switch methods — both have their advantages.
"""

    return f"""
{method_comment}

Now let’s move on.

**Question 11 — What type of oven do you use to bake your pizza?**  
I want to know:

- What kind of oven do you have?  
- Or do you need a recommendation for buying or choosing one?

Here are the options:

A) Wood-fired oven  
B) Gas oven  
C) Standard household electric oven  
D) I don’t have a specific pizza oven but I’d like a recommendation

💡 *For a beginner who wants to make real Neapolitan pizza at home,*  
**I recommend the Tentazione Max electric oven**, which has separate top and bottom heat control and reaches up to **500°C** — specifically designed for Neapolitan pizza.

**What kind of oven do you have, or which one would you like to use?**
"""

@tool(description="""
📌 Purpose:
Handles the selected oven type from the user (electric, gas, wood-fired, Tentazione Max) and transitions to **Question 12**: What is the maximum oven temperature?

✅ Trigger this ONLY when:
- The user has **answered Question 11**
- The input contains specific oven-related keywords such as:
  “electric,” “wood-fired,” “gas,” “household oven,” “Tentazione,” or a known brand/model of pizza oven

🧠 Behavior:
- Confirms the user’s oven type with encouragement or tailored feedback
- Prepares the user for baking instructions suited to that oven
- Introduces the next and final required info: **maximum temperature**

🚫 Do NOT call this tool if:
- The user didn’t mention an oven type
- The message is vague (e.g., “ok,” “continue,” “next step”)
- The oven was already handled and this is a repeat message not asking for more help

🔒 Use strictly within the structured flow — only after **Question 11** is complete and clear.
""")

def get_oven_type_response(oven_type: str) -> str:
    """Responds to selected oven type and asks for maximum temperature."""

    oven_type = oven_type.lower().strip()
    oven_comment = ""

    if "electric" in oven_type or "household" in oven_type:
        oven_comment = """
Excellent! 💪  
You’ve made the perfect choice for a motivated beginner — the electric oven is ideal for learning and making great Neapolitan pizzas at home.
"""
    elif "wood" in oven_type:
        oven_comment = """
Beautiful! A wood-fired oven gives unbeatable authenticity and flavor.  
It requires a bit more control, but the results are incredible — just like in a pizzeria.
"""
    elif "gas" in oven_type:
        oven_comment = """
Nice — gas ovens give great heat distribution and are more powerful than standard electric ones.  
We’ll adapt the technique slightly to match the way it heats.
"""
    elif "tentazione" in oven_type or "500" in oven_type:
        oven_comment = """
Wow! You’ve got the Tentazione Max — that’s the gold standard for home Neapolitan pizza.  
It’s made exactly for this purpose, and with top/bottom heat control and 500°C, you’ll get restaurant-level results.
"""
    else:
        oven_comment = """
Got it! No problem — I’ll adapt the instructions to suit your oven.  
If you’re unsure about its specs, just let me know the model.
"""

    return f"""
{oven_comment}

Now I need to know one last essential thing before we move on to the complete recipe with all the detailed steps.

**Question 12 — What is the maximum temperature your oven can reach?**  
This information is crucial for:

- Understanding preheating times  
- Deciding whether to use a pizza stone or an upside-down baking tray  
- Optimizing the bake to get that classic Neapolitan crust

Here are some options:

A) My oven goes up to 250°C (standard household oven)  
B) My oven goes up to 300°C (more powerful)  
C) I have a dedicated pizza oven that reaches 400°C or more  
D) I have the Tentazione Max electric oven (500°C — highly recommended)  
E) Other (tell me the model or the max temperature)

**What is the maximum temperature of your oven?**
"""

@tool(description="""
📌 Purpose:
Finalizes the full user input session and confirms all choices made during the 12-question flow. It prepares to deliver the complete custom Neapolitan pizza recipe.

✅ Trigger this ONLY when:
- The user has completed all 12 questions
- You have gathered confirmed values for:
  → Experience level  
  → Dietary needs  
  → Number of servings  
  → Pizza weight  
  → Hydration %  
  → Flour type  
  → Yeast type  
  → Room temperature  
  → Fermentation hours  
  → Kneading method  
  → Oven type  
  → Oven max temperature

🧠 Behavior:
- Displays a full structured summary of the user's selections
- Reassures the user with clear confirmation and encouragement
- Asks how they'd prefer to receive the final recipe:
  🟢 One complete message or  
  🟦 Divided into step-by-step blocks (dough → fermentation → baking)

🚫 Do NOT trigger this tool if:
- Any of the 12 question answers are missing
- The user message is vague, not about confirmation, or repeats prior input
- The flow hasn’t reached the summary/delivery decision stage

⚠️ Use ONLY as the final step before recipe generation.
""")

def get_final_confirmation_response(
    level: str,
    dietary: str,
    servings: int,
    pizza_weight: int,
    hydration_percent: int,
    flour_type: str,
    yeast: str,
    room_temperature: int,
    fermentation_hours: int,
    kneading: str,
    oven_type: str,
    oven_temp: int
) -> str:
    """Returns the final confirmation message and recipe delivery options after all questions are answered."""

    total_dough = servings * pizza_weight

    return f"""
    Then you’re absolutely ready to enter the **true school of Neapolitan pizza**! 🔥👨‍🍳🍕

    The **{oven_type.capitalize()}**, with a temperature of up to **{oven_temp}°C**, is top-of-the-line for baking authentic Neapolitan pizza at home:

    ✅ Perfect cooking in 60–90 seconds  
    ✅ Puffy, well-developed crust  
    ✅ Well-cooked, light base  
    ✅ Total heat control — just like in a real pizzeria!

    We now have all the information we need.  
    🎯 **Summary of your preferences:**

    | Item               | Choice                        |  
    |--------------------|-------------------------------|  
    | Level              | {level.capitalize()}          |  
    | Dietary needs      | {dietary.capitalize()}        |  
    | Servings           | {servings}                    |  
    | Pizza weight       | {pizza_weight} g              |  
    | Total dough        | {total_dough} g               |  
    | Hydration          | {hydration_percent}%          |  
    | Flour              | {flour_type}                  |  
    | Yeast              | {yeast.capitalize()}          |  
    | Room temperature   | {room_temperature}°C          |  
    | Fermentation time  | {fermentation_hours} hours    |  
    | Kneading           | {kneading.capitalize()}       |  
    | Oven               | {oven_type} ({oven_temp}°C)   |

    🎉 Now I’ll prepare your **COMPLETE CUSTOM RECIPE**

    In the next message, I’ll give you:

    - Detailed dough measurements using the correct formula  
    - Step-by-step instructions for kneading by hand  
    - Precise fermentation management  
    - When to re-knead and form dough balls  
    - How to manage baking in your {oven_type} oven

    **Are you ready?**  
    Would you like me to send everything in:

    🟢 A) One single message (complete recipe)  
    🟦 B) Divided blocks (Dough → Fermentation → Baking) so you can follow step by step?
    """

@tool(description="""
📌 Purpose:
Delivers Block 1 of the final recipe: Ingredient calculations based on user's full input.

✅ Trigger this ONLY when:
- The user selected “Step by step” delivery option in the final confirmation step.
- All prior 12 questions have been completed and values confirmed.

🧠 What it does:
- Calculates and displays exact ingredient amounts:
  → Flour, Water, Salt, Yeast
- Uses correct Neapolitan pizzeria yeast math formula
- Displays a clear ingredient table with explanations
- Ends by asking if the user is ready to continue to the next step (Block 2: kneading)

🚫 Do NOT trigger this tool if:
- The user hasn't chosen the “step-by-step” format
- Any critical input (servings, hydration, temp, yeast type, etc.) is missing
- The user message is off-topic or unrelated to baking preparation

🧑‍🍳 Why it matters:
This tool provides precise, beginner-friendly instructions with correct proportions for perfect Neapolitan dough and prepares the user for hands-on steps.
""")

def get_step_by_step_start_response(
    servings: int,
    pizza_weight: int,
    hydration_percent: int,
    flour_type: str,
    room_temperature: int,
    fermentation_hours: int,
    yeast_type: str
) -> str:
    """Begins step-by-step instruction: Block 1 - Dough Calculation"""

    # Step 1: Calculate total dough
    total_dough = servings * pizza_weight

    # Step 2: Calculate flour and water
    hydration_ratio = hydration_percent / 100.0
    flour = round(total_dough / (1 + hydration_ratio))
    water = round(flour * hydration_ratio)

    # Step 3: Estimate salt and yeast
    salt = round(flour * 0.032)  # 3.2% gives flexibility: 2–3%
    yeast = round((flour * 23) / (fermentation_hours * hydration_percent * room_temperature), 1)

    return f"""
    Perfect! 🧑‍🍳  
    We’ll do everything step by step, just like a true master pizzaiolo would with a student by their side.

    **Let’s start with Block 1:**  
    🧮 1. **DOUGH INGREDIENT CALCULATION**  
    🎯 Goal: {servings} pizza{'s' if servings > 1 else ''} at {pizza_weight} g each → Total dough: {total_dough} g  
    💧 Hydration: {hydration_percent}%  
    🌾 Flour: {flour_type}  
    🌡️ Room temperature: {room_temperature}°C  
    ⏱️ Fermentation: {fermentation_hours} hours  
    🍃 Yeast: {yeast_type.title()}

    📦 **Ingredients**

    | Ingredient     | Quantity | Notes                                              |  
    |----------------|----------|-----------------------------------------------------|  
    | Type 00 Flour  | {flour} g   | Calculated to reach {total_dough} g total with {hydration_percent}% hydration |  
    | Water          | {water} g   | {hydration_percent}% of {flour} g flour                        |  
    | Salt           | {salt} g    | About 2–3% of the flour weight                              |  
    | {yeast_type.title()} Yeast | {yeast} g    | Formula: (Flour × 23) ÷ (Time × Hydration% × Temp)  
    → ({flour} × 23) ÷ ({fermentation_hours} × {hydration_percent} × {room_temperature}) = {yeast} g

    ✅ This amount of yeast is **PERFECT** for a {fermentation_hours}-hour fermentation at {room_temperature}°C using your {flour_type}.

    ---

    **Do you need to confirm or gather the ingredients before we move on to the next step?**  
    Let me know when you’re ready for **Block 2: How to knead by hand, step by step 💪🍞**
    """


