const API_URL = "http://127.0.0.1:8000/bread";
const START_URL = "http://127.0.0.1:8000/start/";

const chatBox = document.getElementById("chat-box");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const startBtn = document.getElementById("start-btn");
const recipeContainer = document.getElementById("recipe-container");

// --- Reset state on reload ---
localStorage.removeItem("bread_session");
let sessionId = null;
chatBox.innerHTML = "";
recipeContainer.innerHTML = "";

// --- Show intro --- 
addMessage("👋 Welcome to your Bread Baking Assistant! Click 'Start' to begin.", "bot");

// Show Start button, hide chat form
chatForm.style.display = "none";
startBtn.style.display = "block";

startBtn.onclick = async () => {
  // Hide Start button, show chat form
  startBtn.style.display = "none";
  chatForm.style.display = "flex";
  recipeContainer.innerHTML = "";

  // Request session_id from backend /start
  addMessage("Starting chat session...", "bot");
  try {
    const res = await fetch(START_URL, { method: "POST" });
    const data = await res.json();
    if (!data.conversation_id) throw new Error("No session_id returned from /start");
    sessionId = data.conversation_id;
    localStorage.setItem("bread_session", sessionId);
    addMessage("How would you like to proceed — one-by-one or all-at-once?", "bot");
  } catch (err) {
    addMessage("⚠️ Could not start session: " + err.message, "bot");
    chatForm.style.display = "none";
    startBtn.style.display = "block";
  }
};

// --- Display a message in the chat ---
function addMessage(text, sender = "bot") {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  msg.textContent = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// --- Render recipe beautifully ---
function renderRecipe(recipe) {
  recipeContainer.innerHTML = ""; // clear previous
  if (!recipe) return;

  // Title
  if (recipe.dish) {
    const title = document.createElement("h2");
    title.textContent = recipe.dish;
    recipeContainer.appendChild(title);
  }

  // Ingredients
  if (Array.isArray(recipe.ingredients)) {
    const h = document.createElement("h3");
    h.textContent = "Ingredients";
    recipeContainer.appendChild(h);
    const ul = document.createElement("ul");
    recipe.ingredients.forEach(ing => {
      const li = document.createElement("li");
      li.textContent =
        ing.name +
        (ing.quantity_grams ? `: ${ing.quantity_grams}g` : "") +
        (ing.baker_percentage ? ` (${ing.baker_percentage}%)` : "");
      ul.appendChild(li);
    });
    recipeContainer.appendChild(ul);
  }

  // Baker's Percentages
  if (recipe.bakers_percentages) {
    const h = document.createElement("h3");
    h.textContent = "Baker's Percentages";
    recipeContainer.appendChild(h);
    const ul = document.createElement("ul");
    for (const [k, v] of Object.entries(recipe.bakers_percentages)) {
      const li = document.createElement("li");
      li.textContent = `${k}: ${v}%`;
      ul.appendChild(li);
    }
    recipeContainer.appendChild(ul);
  }

  // Hydration
  if (recipe.hydration) {
    const p = document.createElement("p");
    p.innerHTML = `<strong>Hydration:</strong> ${recipe.hydration}`;
    recipeContainer.appendChild(p);
  }

  // Timeline
  if (Array.isArray(recipe.timeline)) {
    const h = document.createElement("h3");
    h.textContent = "Timeline";
    recipeContainer.appendChild(h);
    const ol = document.createElement("ol");
    recipe.timeline.forEach(step => {
      const li = document.createElement("li");
      li.textContent = step;
      ol.appendChild(li);
    });
    recipeContainer.appendChild(ol);
  }

  // Notes
  const notes = [
    ["Equipment Notes", recipe.equipment_notes],
    ["Adaptations", recipe.adaptations],
    ["Tentazione Max Note", recipe.tentazione_max_note],
    ["Plating Tips", recipe.plating_tips],
    ["Storage Tips", recipe.storage_tips]
  ];
  notes.forEach(([header, text]) => {
    if (text) {
      const h = document.createElement("h4");
      h.textContent = header;
      recipeContainer.appendChild(h);
      const p = document.createElement("p");
      p.textContent = text;
      recipeContainer.appendChild(p);
    }
  });
}

// --- Handle user submission ---
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text || !sessionId) return;

  addMessage(text, "user");
  userInput.value = "";
  addMessage("Thinking... 🍞", "bot");

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        input_text: text,
        language: "en"
      }),
    });

    const data = await res.json();
    chatBox.lastChild.remove(); // remove "Thinking..." message

    if (data.status === "question") {
      if (data.message) addMessage(data.message, "bot");
      if (data.question) addMessage(data.question, "bot");
    } 
    else if (data.status === "success") {
      addMessage(data.message, "bot");
      renderRecipe(data.recipe);
    } 
    else {
      addMessage("⚠️ " + (data.message || "Error occurred."), "bot");
    }
  } catch (err) {
    chatBox.lastChild.remove();
    addMessage("⚠️ Connection error: " + err.message, "bot");
  }
});