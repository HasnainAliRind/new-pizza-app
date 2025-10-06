# Pizza App Backend API Documentation

## Overview
This document provides comprehensive API documentation for the Pizza App backend endpoints, specifically for the `/bread` and `/recipes` endpoints. These endpoints are designed to help users generate personalized bread and recipe recommendations through an interactive conversation flow.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, no authentication is required for these endpoints.

---

## 1. Bread Recipe Generator API

### Endpoint: `POST /bread`

Generates personalized bread recipes based on user preferences and requirements through an interactive conversation flow.

#### Request

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "session_id": "string",
  "input_text": "string", 
  "language": "string" // optional, defaults to "en"
}
```

**Request Parameters:**
- `session_id` (string, required): Unique identifier for the user session
- `input_text` (string, required): User's input text containing preferences or answers
- `language` (string, optional): Language code (defaults to "en")

#### Response

The API returns different response formats based on the conversation state:

**1. Incomplete Response (Missing Information):**
```json
{
  "status": "incomplete",
  "questions": ["string"] // Array of questions to ask the user
}
```

**2. Success Response (Recipe Generated):**
```json
{
  "status": "success", 
  "recipe": "string" // Generated bread recipe
}
```

#### Required Information Fields

The API collects the following information before generating a recipe:

1. **mode** - `"all-at-once"` or `"one-by-one"`
2. **experience** - `"beginner"`, `"intermediate"`, or `"expert"`
3. **bread_type** - Type of bread (e.g., "rustic", "whole wheat", "baguette", "focaccia")
4. **flours** - Available flours (e.g., "all-purpose", "bread flour", "whole wheat")
5. **leavening** - Leavening method (e.g., "fresh yeast", "dry yeast", "sourdough starter")
6. **equipment** - Available equipment (e.g., "hand kneading", "stand mixer", "Dutch oven")
7. **fermentation_time** - Time available for fermentation (e.g., "few hours", "12h", "24h")
8. **room_temp** - Room temperature in °C or °F
9. **final_amount** - Desired final amount (weight in g/kg or number of loaves)
10. **dietary** - Dietary restrictions (e.g., "vegan", "gluten-free", "low-salt")
11. **format** - Recipe format preference (`"step-by-step"`, `"compact"`, or `"mixed"`)

#### Example Usage

**Initial Request:**
```json
POST /bread
{
  "session_id": "user123",
  "input_text": "I want to make bread",
  "language": "en"
}
```

**Response:**
```json
{
  "status": "incomplete",
  "questions": ["Do you prefer all questions at once, or one-by-one? (reply: all-at-once or one-by-one)"]
}
```

**Follow-up Request:**
```json
POST /bread
{
  "session_id": "user123", 
  "input_text": "one-by-one",
  "language": "en"
}
```

**Final Success Response:**
```json
{
  "status": "success",
  "recipe": "Here's your personalized bread recipe:\n\nIngredients:\n- 500g bread flour\n- 350ml water\n- 10g salt\n- 5g dry yeast\n\nInstructions:\n1. Mix all ingredients...\n2. Knead for 10 minutes...\n..."
}
```

---

## 2. Generic Recipes API

### Endpoint: `POST /recipes`

Generates personalized recipes for various dishes based on user preferences and requirements through an interactive conversation flow.

#### Request

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "session_id": "string",
  "input_text": "string",
  "language": "string" // optional, defaults to "en"
}
```

**Request Parameters:**
- `session_id` (string, required): Unique identifier for the user session
- `input_text` (string, required): User's input text containing preferences or answers
- `language` (string, optional): Language code (defaults to "en")

#### Response

The API returns different response formats based on the conversation state:

**1. Incomplete Response (Missing Information):**
```json
{
  "status": "incomplete",
  "questions": ["string"] // Array of questions to ask the user
}
```

**2. Success Response (Recipe Generated):**
```json
{
  "status": "success",
  "recipe": "string" // Generated recipe
}
```

#### Required Information Fields

The API collects the following information before generating a recipe:

1. **mode** - `"all-at-once"` or `"one-by-one"`
2. **experience** - `"beginner"`, `"intermediate"`, or `"expert"`
3. **dish_type** - Type of dish (e.g., "starter", "main", "dessert")
4. **cuisine** - Cuisine preference (e.g., "Italian", "French", "Asian", "Mediterranean")
5. **include_ingredients** - Main ingredients to include
6. **avoid_ingredients** - Allergies or ingredients to avoid
7. **equipment** - Available equipment (e.g., "stove", "oven", "grill", "blender")
8. **time_available** - Available cooking time (e.g., "30 min", "1h", "2h+")
9. **servings** - Number of servings
10. **dietary** - Dietary style (e.g., "vegetarian", "vegan", "gluten-free")
11. **special_goal** - Special goal (e.g., "healthy", "gourmet", "quick meal")
12. **format** - Recipe format preference (`"step-by-step"`, `"compact"`, or `"mixed"`)

#### Example Usage

**Initial Request:**
```json
POST /recipes
{
  "session_id": "user456",
  "input_text": "I want to cook something for dinner",
  "language": "en"
}
```

**Response:**
```json
{
  "status": "incomplete",
  "questions": ["Would you like all questions at once, or one-by-one? (reply: all-at-once or one-by-one)"]
}
```

**Follow-up Request:**
```json
POST /recipes
{
  "session_id": "user456",
  "input_text": "all-at-once",
  "language": "en"
}
```

**Final Success Response:**
```json
{
  "status": "success",
  "recipe": "Here's your personalized recipe:\n\nIngredients:\n- 2 chicken breasts\n- 1 cup rice\n- 2 tbsp olive oil\n- Salt and pepper to taste\n\nInstructions:\n1. Season chicken with salt and pepper...\n2. Heat oil in a pan...\n..."
}
```

---

## Implementation Notes for Frontend

### Session Management
- Use a consistent `session_id` for each user session
- The `session_id` can be a UUID or any unique string
- Sessions persist across multiple API calls

### Conversation Flow
1. **Start**: Send initial request with user's intent
2. **Collect Information**: Handle "incomplete" responses by asking questions to users
3. **Complete**: When all required information is collected, receive the final recipe

### Error Handling
- Always check the `status` field in responses
- Handle "incomplete" status by displaying questions to users
- Handle "success" status by displaying the generated recipe

### Mode Selection
- **"all-at-once"**: Returns all remaining questions in a single response
- **"one-by-one"**: Returns one question at a time for a guided experience

### Language Support
- Currently supports English ("en") by default
- Language parameter can be extended for internationalization

### Response Status Codes
- The API uses HTTP 200 for all responses
- Check the `status` field in the JSON response to determine the conversation state

---

## Frontend Integration Example

```javascript
// Example frontend implementation
class RecipeAPI {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.sessionId = this.generateSessionId();
  }

  generateSessionId() {
    return 'session_' + Math.random().toString(36).substr(2, 9);
  }

  async sendMessage(endpoint, inputText, language = 'en') {
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: this.sessionId,
        input_text: inputText,
        language: language
      })
    });

    return await response.json();
  }

  async getBreadRecipe(inputText) {
    return await this.sendMessage('/bread', inputText);
  }

  async getGenericRecipe(inputText) {
    return await this.sendMessage('/recipes', inputText);
  }
}

// Usage example
const api = new RecipeAPI();

// Start bread recipe conversation
const response = await api.getBreadRecipe("I want to make bread");
if (response.status === 'incomplete') {
  // Display questions to user
  console.log('Questions:', response.questions);
} else if (response.status === 'success') {
  // Display recipe
  console.log('Recipe:', response.recipe);
}
```

---

## Testing

You can test these endpoints using tools like:
- **Postman**
- **curl**
- **Thunder Client** (VS Code extension)
- **Insomnia**

### Example curl commands:

```bash
# Test bread endpoint
curl -X POST http://localhost:8000/bread \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test123", "input_text": "I want to make bread", "language": "en"}'

# Test recipes endpoint  
curl -X POST http://localhost:8000/recipes \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test456", "input_text": "I want to cook dinner", "language": "en"}'
```
