from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import re
from uuid import uuid4
import shutil
from typing import List
import PyPDF2
import docx
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pydantic import EmailStr
import uvicorn
import asyncio
from PIL import Image
from io import BytesIO
import aiohttp
from typing import Dict, Tuple
import io
import base64
from openai import OpenAI
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel
from model import (
    UserDB, UserCreate, User, Token, ContactDB, ContactCreate, ContactResponse,PromptCreate, PromptResponse,PromptDB,
    get_db, create_tables, BreadSession, RecipeSession
)
from tool_calling import (pizza_intro, get_pizza_experience_response, get_dietary_response, get_dough_quantity_response, get_pizza_weight_response, get_hydration_response,
                          get_flour_type_response, get_yeast_type_response, get_room_temperature_response, get_kneading_method_response, get_oven_type_response,
                          get_final_confirmation_response, get_step_by_step_start_response, get_fermentation_time_response)


llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([
    pizza_intro,
    get_pizza_experience_response,
    get_dietary_response,
    get_dough_quantity_response,
    get_pizza_weight_response,
    get_hydration_response,
    get_flour_type_response,
    get_yeast_type_response,
    get_room_temperature_response,
    get_final_confirmation_response,
    get_kneading_method_response,
    get_oven_type_response,
    get_step_by_step_start_response,
    get_fermentation_time_response
])

# Tool registry
tool_registry = {
    "pizza_intro": pizza_intro,
    "get_pizza_experience_response": get_pizza_experience_response,
    "get_dietary_response": get_dietary_response,
    "get_dough_quantity_response": get_dough_quantity_response,
    "get_pizza_weight_response": get_pizza_weight_response,
    "get_hydration_response": get_hydration_response,
    "get_flour_type_response": get_flour_type_response,
    "get_yeast_type_response": get_yeast_type_response,
    "get_room_temperature_response": get_room_temperature_response,
    "get_final_confirmation_response": get_final_confirmation_response,
    "get_kneading_method_response": get_kneading_method_response,
    "get_oven_type_response": get_oven_type_response,
    "get_step_by_step_start_response": get_step_by_step_start_response,
    "get_fermentation_time_response": get_fermentation_time_response
}

CASUAL_RESPONSES = {
    "en": "I am a pizza assistant. I can help you with pizza recipes and cooking instructions. Please ask me about making different types of pizzas!",
    "fr": "Je suis un assistant pizza. Je peux vous aider avec des recettes de pizza et des instructions de cuisine. S'il vous plaît, posez-moi des questions sur la préparation de différents types de pizzas !",
    "de": "Ich bin ein Pizza-Assistent. Ich kann Ihnen mit Pizza-Rezepten und Kochanweisungen helfen. Bitte fragen Sie mich nach der Zubereitung verschiedener Pizzasorten!",
    "it": "Sono un assistente pizza. Posso aiutarti con ricette per la pizza e istruzioni di cottura. Chiedimi pure come preparare diversi tipi di pizza!",
    "nl": "Ik ben een pizza-assistent. Ik kan je helpen met pizzarecepten en kookinstructies. Vraag me gerust naar het maken van verschillende soorten pizza's!"
}

# Add pizza-related keywords for each language
PIZZA_KEYWORDS = {
    "en": ["pizza", "dough", "recipe", "neapolitan", "margherita", "topping", "flour", "yeast", "sauce", "cheese", "bake", "oven", "fermentation", "knead", "stretch", "tomato", "mozzarella", "basil", "crust", "hydration"],
    "de": ["pizza", "teig", "rezept", "napolitanisch", "margherita", "belag", "mehl", "hefe", "sauce", "käse", "backen", "ofen", "fermentation", "kneten", "dehnen", "tomate", "mozzarella", "basilikum", "kruste", "hydration"],
    "fr": ["pizza", "pâte", "recette", "napolitaine", "margherita", "garniture", "farine", "levure", "sauce", "fromage", "cuire", "four", "fermentation", "pétrir", "étirer", "tomate", "mozzarella", "basilic", "croûte", "hydratation"],
    "it": ["pizza", "impasto", "ricetta", "napoletana", "margherita", "condimento", "farina", "lievito", "salsa", "formaggio", "cuocere", "forno", "fermentazione", "impastare", "stendere", "pomodoro", "mozzarella", "basilico", "crosta", "idratazione"]
}


create_tables()
app = FastAPI(title="Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

os.makedirs("uploads", exist_ok=True)

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_W5Yz5_nkXMKQGgzAeVfD85CXpqzivXzctr2tfju4e4AmRX4EuYEGSDJYhotiMVxpb7MV")
INDEX_NAME = "image-qa-index"
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI embeddings dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

SECRET_KEY = "your-secret-key-here"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if user is None:
        raise credentials_exception
    return user


document_store = {} 
conversation_store = {}  
 

def extract_pizza_type(question: str) -> str:
    """Extracts the pizza type dynamically from the question."""
    pizza_types = ["Neapolitan", "Chicago Deep Dish", "New York Style", "Sicilian", "Margherita", "Pepperoni"]
    for pizza in pizza_types:
        if pizza.lower() in question.lower():
            return pizza
    return "a generic pizza" 

def process_pdf(file_path: str) -> List[str]:
    """Extract text from PDF and split into chunks."""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle NoneType if page text is empty
    return text_splitter.split_text(text)

def process_docx(file_path: str) -> List[str]:
    """Extract text from DOCX and split into chunks."""
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text_splitter.split_text(text)




# Language mappings
LANGUAGE_MAP = {
    'fr': 'French',
    'french': 'French',
    'français': 'French',
    'de': 'German',
    'german': 'German',
    'deutsch': 'German',
    'it': 'Italian',
    'italian': 'Italian',
    'italiano': 'Italian',
    'en': 'English',
    'english': 'English'
}

# Language-specific templates
ANALYSIS_TEMPLATES = {
    'French': "Analysez cette pizza en détail. Décrivez les garnitures, le style et l'apparence en 3-4 phrases. Limitez votre réponse à environ 100 mots.",
    'German': "Analysieren Sie diese Pizza im Detail. Beschreiben Sie die Beläge, den Stil und das Aussehen in 3-4 Sätzen. Beschränken Sie Ihre Antwort auf etwa 100 Wörter.",
    'Italian': "Analizza questa pizza in dettaglio. Descrivi condimenti, stile e aspetto in 3-4 frasi. Limita la tua risposta a circa 100 parole.",
    'English': "Analyze this pizza in detail. Describe the toppings, style, and appearance in 3-4 sentences. Limit your response to about 100 words."
}

RECIPE_TEMPLATES = {
    'French': "Créez une recette rapide de pizza basée sur cette description : {analysis}. Incluez les ingrédients et les étapes principales.",
    'German': "Erstellen Sie ein schnelles Pizzarezept basierend auf dieser Beschreibung: {analysis}. Fügen Sie die Hauptzutaten und Schritte hinzu.",
    'Italian': "Crea una ricetta veloce della pizza basata su questa descrizione: {analysis}. Includi gli ingredienti principali e i passaggi.",
    'English': "Create a quick pizza recipe based on this description: {analysis}. Include main ingredients and steps."
}


BASE_DOUGH_RECIPES_EN = {
    'napoletana_imperatore': {
        'title': 'Pizza Napoletana (Guido Imperatore Method)',
        'ingredients': [
            '1000g high-protein flour (12-13% protein)',
            '600g water (60% hydration)',
            '3g fresh yeast (0.3%) for long fermentation in summer',
            '30g salt (3%)'
        ],
        'instructions': [
            'Calculate water temperature using this formula: (Desired dough temp × 3) - (room temp + flour temp) - mixer heating factor (spiral mixer ~10°, plunging mixer ~6°, fork/hand ~2°)',
            'Mix flour, water, and yeast (dry yeast for reliability)',
            'Add salt halfway through the mixing process',
            'Mix for about 10 minutes or until a good gluten network forms (dough becomes resistant)',
            'Let the dough bulk ferment for about 30 minutes',
            'Let the dough rest for at least 6-8 hours',
            'Form dough balls of about 280g each',
            'Let them proof in plastic containers at room temperature',
            'Alternatively, you can leave the dough in bulk overnight and form balls the next morning, letting them rise until evening. It\'s important to form the balls when the dough is cold',
            'In winter, if house temperatures are too low, consider increasing the yeast to about 1% if you want to ferment at controlled temperature'
        ]
    },
    'napoletana_curro': {
        'title': 'Pizza Napoletana Verace (Massimo Currò Method)',
        'ingredients': [
            '1000g type 00 flour (12-13% protein)',
            '600-620g water (60-62% hydration)',
            '30g salt (3%)',
            'Fresh yeast (calculated with formula: flour weight × 23 ÷ room temp ÷ hours ÷ hydration)'
        ],
        'instructions': [
            'Calculate yeast amount using the formula: flour weight in grams × 23 ÷ room temperature ÷ hours needed ÷ hydration percentage',
            'For dry yeast: up to 3g use 1:1 ratio with fresh yeast, 4-9g use 2:3 ratio, 10g+ use 1:3 ratio',
            'Dissolve salt in water in a bowl',
            'Begin adding flour, when it reaches a creamy consistency dissolve the yeast',
            'Continue adding remaining flour',
            'Transfer dough to work surface and knead for 7-8 minutes',
            'Once proper consistency is reached, let it rise covered for about 30 minutes',
            'Divide into 250g balls to make 5-6 pizzas',
            'Let proof for the time calculated in your yeast formula at the specified temperature'
        ]
    },
    'gluten_free': {
        'title': 'Gluten-Free Dough Mix (for bread or pizza)',
        'ingredients': [
            '100g corn starch',
            '100g rice flour',
            '50g potato starch',
            '30g buckwheat flour',
            '5g xanthan flour (or equal amount of guar gum)',
            '5g powdered milk',
            '6.5g salt',
            '8-10g extra virgin olive oil',
            '155g water',
            '3g fresh yeast'
        ],
        'instructions': [
            'Mix all dry ingredients except salt',
            'Add salt, oil, and water in sequence as with regular dough',
            'Hydration is 55%, ferment at room temperature for about 12 hours',
            'Fold every 3-4 hours',
            'Note that yeast amount is higher than standard due to the flours used, but still less than pre-packaged mixes recommend',
            'Bake on parchment paper in a pan, as the dough is difficult to handle and tends to break apart',
            'Bake at 180°C for the first 20 minutes, then at 150-160°C for about 15 more minutes or until the surface is golden'
        ]
    }
}

# Italian version
BASE_DOUGH_RECIPES_IT = {
    'napoletana_imperatore': {
        'title': 'Pizza Napoletana (ricetta del pizzaiolo Guido Imperatore)',
        'ingredients': [
            '100% farina 12%/13% proteine',
            '60% acqua',
            '0.3% di lievito per lievitazioni lunghe in estate',
            '3% di sale'
        ],
        'instructions': [
            'Calcolo della temperatura dell\'acqua da inserire: (Temperatura desiderata impasto x 3) - (temperatura ambiente + temperatura farina) - gradi riscaldamento impastatrice',
            'Impasto versando farina, acqua, lievito (secco per maggiore tranquillità)',
            'A metà impastamento aggiungo il sale',
            'Impasto per circa 10 minuti o fino a formazione di una buona maglia glutinica',
            'Lascio lievitare in massa per circa 30 minuti',
            'Staglio per almeno 6/8 ore',
            'Formo panetti da 280 g circa',
            'Lievitazione in contenitori di plastica a temperatura ambiente',
            'Lo stesso impasto può essere lasciato in massa per una notte e formato in panetti la mattina successiva lasciando lievitare fino a sera. Importante è formare i panetti da freddo',
            'In inverno, in caso di temperature troppo basse in casa, valutare di aumentare il lievito fino almeno ad 1% circa se si vuole fare lievitazione a temperatura controllata'
        ]
    },
    'napoletana_curro': {
        'title': 'Pizza Napoletana Verace (ricetta di Massimo Currò)',
        'ingredients': [
            '100% farina tipo 00 12%/13% proteine',
            '60/62% acqua',
            '3% di sale',
            'Lievito: calcolato con formula (quantità di farina × 23 ÷ temperatura ambiente ÷ ore ÷ idratazione)'
        ],
        'instructions': [
            'Calcola il lievito con questa formula: farina in grammi × 23 ÷ temperatura ambiente ÷ ore di lievitazione ÷ percentuale di idratazione',
            'Per il lievito secco: fino a 3g rapporto 1:1 con lievito fresco, 4-9g rapporto 2:3, 10g+ rapporto 1:3',
            'Prendere ciotola con 600 di acqua, sciogliere al suo interno il sale',
            'Inizia ad aggiungere la farina, appena l\'impasto ha consistenza di crema sciogli il lievito',
            'Continuare fino a finire la farina',
            'Portare l\'impasto fuori dalla ciotola e spostarlo su un piano di lavoro, lavorarlo per circa 7/8 minuti',
            'Una volta raggiunta consistenza, farlo lievitare coperto per una mezz\'oretta',
            'Passata la mezz\'oretta fai lo staglio separandolo in panetti da 250 grammi l\'uno (5/6 pizze)',
            'Fallo lievitare per il tempo che hai inserito nella formula per il calcolo del lievito alla temperatura indicata'
        ]
    },
    'gluten_free': {
        'title': 'Ricetta Mix senza glutine per pane (utilizzabile anche per pizza)',
        'ingredients': [
            '100gr amido di mais',
            '100gr farina di riso',
            '50gr fecola di patate',
            '30gr farina di grano saraceno',
            '5gr di farina di Xantano (oppure pari quantità di gomma "guar")',
            '5gr latte in polvere',
            '6,5gr sale',
            '8/10 gr olio E.V.O.',
            '155gr acqua',
            '3gr lievito di birra fresco'
        ],
        'instructions': [
            'Unite e mescolate tutti gli ingredienti tranne sale, olio e acqua',
            'Aggiungete sale, olio e acqua nella sequenza di ogni altro impasto',
            'Idratazione 55%, circa 12 ore a Temperatura Ambiente',
            'Serie di pieghe ogni 3-4 ore',
            'Il lievito è maggiore rispetto allo standard a causa del tipo di farine usate, ma comunque molto sotto alle dosi consigliate dei mix preconfezionati',
            'Cottura in teglia su carta forno, il panetto è di difficile gestione, tende a sfaldarsi',
            'Cuocere a 180°C i primi 20 minuti, i successivi 15 minuti a 150-160°C fino a dorare la superficie'
        ]
    }
}

BASE_DOUGH_RECIPES_DE = {
    'napoletana_imperatore': {
        'title': 'Pizza Napoletana (Methode von Guido Imperatore)',
        'ingredients': [
            '1000g hochproteinhaltiges Mehl (12-13% Protein)',
            '600g Wasser (60% Hydratation)',
            '3g frische Hefe (0,3%) für lange Gärung im Sommer',
            '30g Salz (3%)'
        ],
        'instructions': [
            'Berechnen Sie die Wassertemperatur mit dieser Formel: (Gewünschte Teigtemperatur × 3) - (Raumtemperatur + Mehltemperatur) - Erwärmungsfaktor des Mixers (Spiral ~10°, Tauch ~6°, Gabel/Hand ~2°)',
            'Mehl, Wasser und Hefe mischen (Trockenhefe für mehr Stabilität verwenden)',
            'Salz in der Mitte des Mischvorgangs hinzufügen',
            'Etwa 10 Minuten kneten, bis sich ein gutes Klebergerüst bildet (Teig wird elastisch)',
            'Den Teig ca. 30 Minuten ruhen lassen',
            'Mindestens 6-8 Stunden ruhen lassen',
            'Teiglinge von ca. 280g formen',
            'In Plastikbehältern bei Raumtemperatur gehen lassen',
            'Alternativ kann der Teig über Nacht als Ganzes ruhen und am nächsten Morgen zu Teiglingen geformt werden, um bis zum Abend aufzugehen. Wichtig ist, dass die Teiglinge im kalten Zustand geformt werden',
            'Im Winter, wenn die Raumtemperaturen zu niedrig sind, kann die Hefemenge auf 1% erhöht werden, um eine kontrollierte Gärung zu ermöglichen'
        ]
    },
    'napoletana_curro': {
        'title': 'Pizza Napoletana Verace (Methode von Massimo Currò)',
        'ingredients': [
            '1000g Tipo-00-Mehl (12-13% Protein)',
            '600-620g Wasser (60-62% Hydratation)',
            '30g Salz (3%)',
            'Frische Hefe (berechnet nach der Formel: Mehlgewicht × 23 ÷ Raumtemperatur ÷ Stunden ÷ Hydratation)'
        ],
        'instructions': [
            'Berechnen Sie die Hefe mit der Formel: Mehlgewicht × 23 ÷ Raumtemperatur ÷ Stunden ÷ Hydratationsprozentsatz',
            'Für Trockenhefe: bis zu 3g im Verhältnis 1:1 zu frischer Hefe, 4-9g im Verhältnis 2:3, über 10g im Verhältnis 1:3',
            'Salz im Wasser auflösen',
            'Nach und nach das Mehl hinzufügen, wenn eine cremige Konsistenz erreicht ist, die Hefe auflösen',
            'Restliches Mehl hinzufügen und mischen',
            'Auf eine Arbeitsfläche übertragen und 7-8 Minuten kneten',
            '30 Minuten abgedeckt ruhen lassen',
            'In 250g Teiglinge teilen (5-6 Pizzen)',
            'Für die berechnete Gärzeit bei der angegebenen Temperatur gehen lassen'
        ]
    },
    'gluten_free': {
        'title': 'Glutenfreie Teigmischung (für Brot oder Pizza)',
        'ingredients': [
            '100g Maisstärke',
            '100g Reismehl',
            '50g Kartoffelstärke',
            '30g Buchweizenmehl',
            '5g Xanthanmehl (oder gleiche Menge Guarkernmehl)',
            '5g Milchpulver',
            '6,5g Salz',
            '8-10g Olivenöl extra vergine',
            '155g Wasser',
            '3g frische Hefe'
        ],
        'instructions': [
            'Alle trockenen Zutaten außer Salz mischen',
            'Salz, Öl und Wasser nacheinander hinzufügen',
            'Hydratation beträgt 55%, etwa 12 Stunden bei Raumtemperatur fermentieren lassen',
            'Alle 3-4 Stunden falten',
            'Die Hefemenge ist höher als bei Standardrezepten aufgrund der verwendeten Mehle, jedoch geringer als bei Fertigmischungen',
            'Auf Backpapier in einer Form backen, da der Teig schwer zu handhaben ist und leicht bricht',
            'Bei 180°C für 20 Minuten backen, dann bei 150-160°C für weitere 15 Minuten oder bis die Oberfläche goldbraun ist'
        ]
    }
}

BASE_DOUGH_RECIPES_FR = {
    'napoletana_imperatore': {
        'title': 'Pizza Napoletana (Méthode Guido Imperatore)',
        'ingredients': [
            '1000g de farine à haute teneur en protéines (12-13% protéines)',
            '600g d’eau (60% hydratation)',
            '3g de levure fraîche (0,3%) pour une fermentation longue en été',
            '30g de sel (3%)'
        ],
        'instructions': [
            'Calculer la température de l’eau avec cette formule : (Température souhaitée de la pâte × 3) - (température ambiante + température de la farine) - facteur de chauffage du pétrin (spirale ~10°, plongeant ~6°, fourchette/main ~2°)',
            'Mélanger la farine, l’eau et la levure (levure sèche pour plus de stabilité)',
            'Ajouter le sel à mi-pétrissage',
            'Pétrir pendant environ 10 minutes jusqu’à obtenir un bon réseau de gluten (la pâte devient résistante)',
            'Laisser fermenter en masse pendant environ 30 minutes',
            'Laisser reposer au minimum 6-8 heures',
            'Former des pâtons d’environ 280g',
            'Les laisser lever dans des récipients en plastique à température ambiante',
            'Alternativement, la pâte peut être laissée en masse toute la nuit et les pâtons formés le matin suivant pour une levée jusqu’au soir. Il est important que les pâtons soient formés à froid',
            'En hiver, si la température ambiante est trop basse, augmenter la levure jusqu’à environ 1% pour assurer une fermentation à température contrôlée'
        ]
    },
    'napoletana_curro': {
        'title': 'Pizza Napoletana Verace (Méthode Massimo Currò)',
        'ingredients': [
            '1000g de farine type 00 (12-13% protéines)',
            '600-620g d’eau (60-62% hydratation)',
            '30g de sel (3%)',
            'Levure fraîche (calculée avec la formule : poids de la farine × 23 ÷ température ambiante ÷ heures ÷ hydratation)'
        ],
        'instructions': [
            'Calculer la quantité de levure avec la formule : poids de la farine × 23 ÷ température ambiante ÷ heures nécessaires ÷ pourcentage d’hydratation',
            'Pour la levure sèche : jusqu’à 3g ratio 1:1 avec la levure fraîche, 4-9g ratio 2:3, 10g+ ratio 1:3',
            'Dissoudre le sel dans l’eau',
            'Ajouter progressivement la farine, une fois une consistance crémeuse atteinte, dissoudre la levure',
            'Ajouter le reste de la farine et mélanger',
            'Transférer sur un plan de travail et pétrir pendant 7-8 minutes',
            'Laisser reposer couvert pendant environ 30 minutes',
            'Diviser en pâtons de 250g (5-6 pizzas)',
            'Laisser lever pour le temps calculé selon la formule à la température indiquée'
        ]
    },
    'gluten_free': {
        'title': 'Mélange de pâte sans gluten (pour pain ou pizza)',
        'ingredients': [
            '100g de fécule de maïs',
            '100g de farine de riz',
            '50g de fécule de pomme de terre',
            '30g de farine de sarrasin',
            '5g de farine de xanthane (ou même quantité de gomme de guar)',
            '5g de lait en poudre',
            '6,5g de sel',
            '8-10g d’huile d’olive extra vierge',
            '155g d’eau',
            '3g de levure fraîche'
        ],
        'instructions': [
            'Mélanger tous les ingrédients secs sauf le sel',
            'Ajouter le sel, l’huile et l’eau dans l’ordre habituel',
            'Hydratation à 55%, fermentation à température ambiante pendant environ 12 heures',
            'Plier la pâte toutes les 3-4 heures',
            'La quantité de levure est plus élevée que pour une pâte classique en raison des farines utilisées, mais reste inférieure aux mélanges industriels',
            'Cuire sur papier sulfurisé dans un moule, car la pâte est fragile',
            'Cuire à 180°C pendant 20 minutes, puis à 150-160°C pendant 15 minutes supplémentaires jusqu’à ce que la surface soit dorée'
        ]
    }
}

# Language mapping
DOUGH_RECIPE_LANGUAGE_MAP = {
    'English': BASE_DOUGH_RECIPES_EN,
    'Italian': BASE_DOUGH_RECIPES_IT,
    'German': BASE_DOUGH_RECIPES_DE,  
    'French': BASE_DOUGH_RECIPES_FR  
}

def detect_language(query: str) -> str:
    """Detect language from query"""
    query_lower = query.lower().strip()
    return LANGUAGE_MAP.get(query_lower, 'English')

async def compress_image(image_data: bytes) -> str:
    """Compress image more aggressively for faster processing"""
    try:
        img = Image.open(BytesIO(image_data))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        width, height = img.size
        # More aggressive resizing to 256px max dimension
        ratio = 256 / max(width, height)
        if ratio < 1:
            new_size = (int(width * ratio), int(height * ratio))
            # Use BILINEAR instead of LANCZOS for faster resizing
            img = img.resize(new_size, Image.Resampling.BILINEAR)
        
        buffer = BytesIO()
        # Lower quality for faster processing
        img.save(buffer, format='JPEG', quality=60, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

# async def analyze_pizza(session: aiohttp.ClientSession, base64_image: str, language: str) -> Tuple[str, str]:
#     """Analyze pizza and generate recipe in specified language"""
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {OPENAI_API_KEY}"
#     }
    
#     try:
        
#         dough_recipes = DOUGH_RECIPE_LANGUAGE_MAP.get(language, BASE_DOUGH_RECIPES_EN)
#         selected_dough = dough_recipes.get(dough_type, dough_recipes['napoletana_imperatore'])
#         analysis_prompt = ANALYSIS_TEMPLATES.get(language, ANALYSIS_TEMPLATES['English'])
        
#         analysis_payload = {
#             "model": "gpt-4-turbo",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": analysis_prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{base64_image}"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             "max_tokens": 150  
#         }

#         async with session.post(
#             "https://api.openai.com/v1/chat/completions",
#             headers=headers,
#             json=analysis_payload,
#             timeout=10
#         ) as response:
#             analysis_data = await response.json()
#             if response.status != 200:
#                 raise HTTPException(status_code=response.status, detail="Analysis failed")
            
#             analysis = analysis_data['choices'][0]['message']['content']

#             words = analysis.split()
#             if len(words) > 100:
#                 analysis = ' '.join(words[:100])
#                 if not analysis.endswith('.'):
#                     analysis += '.'
            
#             # Second API call for recipe
#             recipe_prompt = RECIPE_TEMPLATES.get(language, RECIPE_TEMPLATES['English']).format(
#                 analysis=analysis,
#                 dough_title=selected_dough['title']
#                 )
#             dough_ingredients = ', '.join(selected_dough['ingredients'])
#             dough_instructions = ' '.join(selected_dough['instructions'])
            
#             recipe_payload = {
#                 "model": "gpt-3.5-turbo",
#                 "messages": [
#                     {
#                         "role": "user",
#                         # "content": recipe_prompt
#                         "content": f"You are creating a pizza recipe. For the dough base, use this authentic recipe: {selected_dough['title']}. Ingredients: {dough_ingredients}. Instructions: {dough_instructions}"

#                     }
#                 ],
#                 "max_tokens": 250,
#                 "temperature": 0.7,
#                 "presence_penalty": 0.6
#             }

#             async with session.post(
#                 "https://api.openai.com/v1/chat/completions",
#                 headers=headers,
#                 json=recipe_payload,
#                 timeout=8
#             ) as recipe_response:
#                 recipe_data = await recipe_response.json()
#                 if recipe_response.status != 200:
#                     raise HTTPException(status_code=recipe_response.status, detail="Recipe generation failed")
                
#                 recipe = recipe_data['choices'][0]['message']['content']
                
#                 return analysis, recipe, selected_dough

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

async def analyze_pizza(session: aiohttp.ClientSession, base64_image: str, language: str, dough_type: str = 'napoletana_imperatore') -> Tuple[str, str, dict]:
    """Analyze pizza and generate recipe in specified language with authentic dough base"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"  # Assuming OPENAI_API_KEY is defined elsewhere
    }
    
    try:
        # Get the appropriate dough recipe based on language
        dough_recipes = DOUGH_RECIPE_LANGUAGE_MAP.get(language, BASE_DOUGH_RECIPES_EN)
        selected_dough = dough_recipes.get(dough_type, dough_recipes['napoletana_imperatore'])
        
        analysis_prompt = ANALYSIS_TEMPLATES.get(language, ANALYSIS_TEMPLATES['English'])
        
        analysis_payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 150  
        }

        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=analysis_payload,
            timeout=10
        ) as response:
            analysis_data = await response.json()
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail="Analysis failed")
            
            analysis = analysis_data['choices'][0]['message']['content']

            words = analysis.split()
            if len(words) > 100:
                analysis = ' '.join(words[:100])
                if not analysis.endswith('.'):
                    analysis += '.'
            
            # Second API call for recipe, now including dough information
            recipe_prompt = RECIPE_TEMPLATES.get(language, RECIPE_TEMPLATES['English']).format(
                analysis=analysis,
                dough_title=selected_dough['title']
            )
            
            # Include dough recipe details as system message
            dough_ingredients = ', '.join(selected_dough['ingredients'])
            dough_instructions = ' '.join(selected_dough['instructions'])
            
            recipe_payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are creating a pizza recipe. For the dough base, use this authentic recipe: {selected_dough['title']}. Ingredients: {dough_ingredients}. Instructions: {dough_instructions}"
                    },
                    {
                        "role": "user",
                        "content": recipe_prompt
                    }
                ],
                "max_tokens": 250,  # Increased max tokens to accommodate dough details
                "temperature": 0.7,
                "presence_penalty": 0.6
            }

            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=recipe_payload,
                timeout=8
            ) as recipe_response:
                recipe_data = await recipe_response.json()
                if recipe_response.status != 200:
                    raise HTTPException(status_code=recipe_response.status, detail="Recipe generation failed")
                
                recipe = recipe_data['choices'][0]['message']['content']
                
                return analysis, recipe, selected_dough

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signup", response_model=Token)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
 
    print(f"[{datetime.now()}] Called /signup")
    print(f"Payload: user={user}, db={db}")

    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = UserDB(
        email=user.email,
        name=user.name,
        hashed_password=hashed_password
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login")
async def login(
    email: str = Form(...), 
    password: str = Form(...), 
    db: Session = Depends(get_db)
):
    
    print(f"[{datetime.now()}] Called /login")
    print(f"Payload: email={email}, password={password}, db={db}")
    # Find user by email
    user = db.query(UserDB).filter(UserDB.email == email).first()
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )
    
    # Verify password
    if not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect password"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user_id": user.id,
        "user_name": user.name,
        "message": "Login successful"
    }

@app.post("/admin-login")
async def admin_login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    
    
    print(f"[{datetime.now()}] Called /admin-login")
    print(f"Payload: form_data={form_data}, db={db}")
    # Check admin credentials
    if form_data.username != "admin@gmail.com" or form_data.password != "12345678":
        raise HTTPException(
            status_code=401,
            detail="Incorrect admin credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create admin access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": "admin@gmail.com"}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: UserDB = Depends(get_current_user)):

    print(f"[{datetime.now()}] Called /users/me ")
    print(f"Payload: current_user={current_user}")

    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return current_user


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    print(f"[{datetime.now()}] Called /upload")
    print(f"Payload: file={file}")

    try:
        # Save the file temporarily
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)  
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process file based on extension
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension == 'pdf':
            chunks = process_pdf(file_path)
        elif file_extension == 'docx':
            chunks = process_docx(file_path)
        else:
            return {"error": "Unsupported file format. Please upload PDF or DOCX files only."}

        document_id = file.filename  # Using filename as a unique document ID
        document_store[document_id] = " ".join(chunks)  # Store text in memory
        vectorstore.add_texts(chunks)

        # Clean up the uploaded file
        os.remove(file_path)

        return {"message": "Document processed successfully!", "document_id": document_id}

    except Exception as e:
        return {"error": str(e)}




@app.post("/start/")
async def start_conversation():
    print(f"[{datetime.now()}] Called /start")
    print(f"Payload: nothing")


    conversation_id = str(uuid4())
    conversation_store[conversation_id] = [] 
    return {"conversation_id": conversation_id}


def get_pizza_system_prompt(language: str) -> str:
    """
    Returns the system prompt for the pizza assistant.
    
    Args:
        language (str): Target language
    
    Returns:
        str: System prompt
    """

    system_prompt = f""" 
        You are a professional Italian pizza maker with many years of hands-on experience in creating traditional Neapolitan pizzas and their variations. You are passionate, friendly, and highly skilled, like a true maestro who is both a craftsman and a teacher. Your role is to teach students how to make the perfect pizza, from selecting the ingredients to managing the dough’s leavening and mastering oven baking techniques.

        Your Objective:
        Guide the user to make the most suitable Neapolitan pizza for their needs by:
        Asking detailed, progressive questions.
        Providing clear, practical instructions.
        Adapting your explanations based on the user’s skill level.
        Teaching with warmth, motivation, and professionalism.

        STEP-BY-STEP INTERACTION INSTRUCTIONS:
        You must go step by step. Ask one question at a time, analyzing the user’s response carefully before proceeding. Use follow-up questions if you need more details. For each question, suggest some possible answers to help the user decide.

        **QUESTIONS** (Answer in order):  
            1. **Skill Level**: Are you a beginner, intermediate, or expert pizza maker?  
            - Options: [Beginner / Intermediate / Expert]  
            - If Options == "Beginner", respond with a hardcoded message:
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

            2. **Dietary Needs**: Any allergies, intolerances, or preferences (vegan, gluten-free, etc.)?  
            - Options: [None / Vegan / Gluten-Free / Other]  

            3. **Servings**: How many people are you cooking for?  
            - Example: 2 / 4 / 6 / 8  

            4. **Dough Weight per Pizza**: What’s your target dough weight? (Unsure? I’ll suggest 250g!)  
            - Example: 200g / 250g / 280g  

            5. **Hydration %**: Choose 60-70% hydration. Need help? Here’s the math:  
            - Formula: For 5x250g pizzas (1250g total dough):  
                Flour = 744g, Water = 484g (65% hydration).  
                Salt = 1250g - (744g + 484g) = 22g.  

            6. **Flour Type**: What flour are you using? Share its protein % (ideal: 12-14% for Neapolitan).  
            - Options: [Tipo 00 / Bread Flour / All-Purpose / Other]  

            7. **Yeast Type**: Fresh yeast, dry yeast, or sourdough?  
            - Ratios: Fresh → Dry yeast:  
                - 1:1 (≤3g), 2:3 (4-9g), 1:3 (≥10g).  

            8. **Room Temperature**: What’s your kitchen temp? (Ideal: 20-25°C).  

            9. **Leavening Time**: Total desired rise time? (Short: 8h / Long: 24-48h).  
            - Rule: First rise = 25% of total time (e.g., 6h for 24h), then shape dough balls for remaining 75%.  

            10. **Kneading Method**: Hand-knead or machine? No preference? I recommend PEB mixers!  
                - Options: [Hand / Planetary Mixer / No Preference]  

            11. **Oven Type**: Wood-fired, electric, or gas? Need a recommendation? Tentazione Max electric oven (500°C).  

            12. **Max Oven Temp**: What’s your oven’s peak heat? (<450°C? Upgrade suggested!).  

            
        FINAL RECIPE INCLUDES:  
        1. Dough Formula: Exact grams for flour, water, salt, yeast (calculated for YOU).  
        2️. Step-by-Step Guide: Mixing, kneading, leavening, shaping dough balls.  
        3️. Leavening Schedule: Precise timings for bulk rise + final proof.  
        4️. Oven Setup: Heat management based on your oven type. 
        """
    
    # system_prompt = f"""
    # Behave like a professional pizzaiolo with years of experience in making Neapolitan pizza and its variations.

    # 🎯 Objective: Teach your student how to make the perfect pizza—starting from ingredient selection to baking—through clear, detailed, and practical instructions. Adjust your guidance based on the student’s experience level.

    # 🎙️ Tone: Friendly, motivating, and professional, like a true passionate master pizzaiolo.

    # 🔍 Focus on: Dough preparation and all stages of fermentation.

    # 🧠 Strategy: Ask ONE question at a time. Analyze the user’s answer carefully. If needed, ask follow-up questions for clarity. For each question, offer example answers (A, B, C...).

    # ---

    # Start with Question 1:

    # **Question 1 – Are you experienced with making Neapolitan pizza, or are you a beginner?**

    # Example answers:
    # A) beginner
    # B) I’m a beginner  
    # C) I’ve made pizza a few times  
    # D) I have solid experience  

    # ❗If the user responds with anything indicating they are a *beginner*, always reply with this exact hardcoded message:

    # ---

    # Perfect! Great answer.

    # Then I will act like a patient and passionate master pizzaiolo who wants to teach you everything—without assuming you already know anything.

    # I’ll make sure every step is clear, detailed, and practical, just like we’re in the workshop together.

    # Let’s move on to the second question:

    # **Question 2 – Do you have any dietary needs or restrictions?**  
    # I need to know if you have any intolerances or preferences so I can recommend the right ingredients.

    # Here are some example answers:

    # A) No particular dietary restrictions or intolerances  
    # B) I’m vegan (no animal products—even though the classic dough doesn’t change)  
    # C) I’m gluten intolerant (we’ll need gluten-free flour)  
    # D) I’m lactose intolerant (not an issue for classic dough)  
    # E) Other restriction or need (please specify)

    # **What’s your answer?**  
    # Reply and we’ll move to question 3!

    # ---

    # Let me know when you're ready for the next question or if you'd like me to embed additional hardcoded responses.

    # """

    # system_prompt = f""" 
    #     You are a professional Italian pizza maker with many years of hands-on experience in creating traditional Neapolitan pizzas and their variations. You are passionate, friendly, and highly skilled, like a true maestro who is both a craftsman and a teacher. Your role is to teach students how to make the perfect pizza, from selecting the ingredients to managing the dough’s leavening and mastering oven baking techniques.

    #     Your Objective:
    #     Guide the user to make the most suitable Neapolitan pizza for their needs by:
    #     - Asking detailed, progressive questions.
    #     - Providing clear, practical instructions.
    #     - Adapting your explanations based on the user’s skill level.
    #     - Teaching with warmth, motivation, and professionalism.

    #     INTERACTION RULES:
    #     - Ask one question at a time.
    #     - Carefully analyze the user's answer before proceeding.
    #     - If an answer is unclear or incomplete, ask for clarification.
    #     - Suggest examples or options to guide their decisions.

    #     Once you receive complete answers to **all 12 questions**, IMMEDIATELY proceed to:
    #     - Calculate the recipe according to the user’s needs (servings, hydration, flour, yeast, etc.).
    #     - Provide the full recipe and instructions in detail, **without waiting for the user to request it**.
    #     - Always include exact weights (g) and steps, based on their environment and preferences.

    #     QUESTIONS TO ASK SEQUENTIALLY:

    #     1. **Skill Level**: Are you a beginner, intermediate, or expert pizza maker?
    #     - Options: [Beginner / Intermediate / Expert]

    #     2. **Dietary Needs**: Any allergies, intolerances, or preferences (vegan, gluten-free, etc.)?
    #     - Options: [None / Vegan / Gluten-Free / Other]

    #     3. **Servings**: How many people are you cooking for?
    #     - Example: 2 / 4 / 6 / 8

    #     4. **Dough Weight per Pizza**: What’s your target dough weight? (Unsure? I’ll suggest 250g!)
    #     - Example: 200g / 250g / 280g

    #     5. **Hydration %**: Choose 60-70% hydration. Need help? Here’s the math:
    #     - Formula: For 5x250g pizzas (1250g total dough):
    #         Flour = 744g, Water = 484g (65% hydration).
    #         Salt = 1250g - (744g + 484g) = 22g.

    #     6. **Flour Type**: What flour are you using? Share its protein % (ideal: 12-14% for Neapolitan).
    #     - Options: [Tipo 00 / Bread Flour / All-Purpose / Other]

    #     7. **Yeast Type**: Fresh yeast, dry yeast, or sourdough?
    #     - Ratios: Fresh → Dry yeast:
    #         - 1:1 (≤3g), 2:3 (4-9g), 1:3 (≥10g).

    #     8. **Room Temperature**: What’s your kitchen temp? (Ideal: 20-25°C)

    #     9. **Leavening Time**: Total desired rise time? (Short: 8h / Long: 24-48h)
    #     - Rule: First rise = 25% of total time (e.g., 6h for 24h), then shape dough balls for remaining 75%.

    #     10. **Kneading Method**: Hand-knead or machine? No preference? I recommend PEB mixers!
    #         - Options: [Hand / Planetary Mixer / No Preference]

    #     11. **Oven Type**: Wood-fired, electric, or gas? Need a recommendation? Tentazione Max electric oven (500°C).

    #     12. **Max Oven Temp**: What’s your oven’s peak heat? (<450°C? Upgrade suggested!)

    #     AFTER QUESTION 12:
    #     Immediately calculate and provide the complete personalized recipe including:

    #     1. **Dough Formula**: Flour (g), Water (g), Hydration (%), Salt (g), Yeast (g).
    #     - Fresh yeast formula:
    #         Fresh yeast (g) = (Grams of flour × 23) / (Total leavening time × Hydration % × Room temp)
    #     - Dry yeast ratios:
    #         - 1:1 up to 3g fresh
    #         - 2:3 for 4–9g fresh
    #         - 1:3 for ≥10g fresh

    #     2. **How to make the dough**: Mixing instructions, kneading method.

    #     3. **Leavening Schedule**: Timings for bulk fermentation and dough ball proofing.

    #     4. **When to re-knead and ball the dough**

    #     5. **How to bake**: Oven setup, preheat strategy, baking time, and tips based on oven type.

    #     Always end your recipe with a warm encouragement and expert advice tip!
        
    #     """
    
    return system_prompt



def extract_pizza_type(question: str) -> str:
    """
    Attempts to extract the pizza type from the question.
    
    Args:
        question (str): User's question
    
    Returns:
        str: Extracted pizza type or empty string
    """
    # Common pizza types (expanded)
    pizza_types = [
        "margherita", "marinara", "diavola", "quattro formaggi", "capricciosa", 
        "neapolitan", "new york", "chicago", "detroit", "sicilian", "romana",
        "napoletana", "calabrese", "pugliese", "bianca", "prosciutto", "funghi",
        "pepperoni", "hawaiian", "calzone", "stuffed crust", "deep dish", "thin crust",
        "pan pizza", "sourdough", "gluten-free", "stuffed", "vegetarian", "meat lovers"
    ]
    
    for pizza_type in pizza_types:
        if pizza_type.lower() in question.lower():
            return pizza_type
    
    return ""

def get_appropriate_greeting(language: str) -> str:
    """
    Returns an appropriate greeting in the target language.
    
    Args:
        language (str): Target language
    
    Returns:
        str: Appropriate greeting with strict language separation
    """
    greetings = {
        "en": "Hello pizza lover! 🍕",
        "it": "Buongiorno, amante della pizza! 🍕",
        "es": "¡Hola, amante de la pizza! 🍕",
        "fr": "Bonjour, amateur de pizza! 🍕",
        "de": "Hallo, Pizza-Liebhaber! 🍕"
    }
    
    # For English, ensure we have multiple English-only greeting options
    english_greetings = [
        "Hello pizza lover! 🍕",
        "Welcome, pizza enthusiast! 🍕",
        "Hi there, pizza fan! 🍕",
        "Greetings, pizza aficionado! 🍕",
        "Hello and welcome, pizza friend! 🍕"
    ]
    
    # Default to English if language code not found
    language_code = language.lower()[:2]
    
    # For English, randomly select from English-only greetings
    if language_code == "en":
        import random
        return random.choice(english_greetings)
    else:
        return greetings.get(language_code, english_greetings[0])

def format_experience_question(language: str) -> str:
    """
    Returns a formatted question about experience level in the target language.
    
    Args:
        language (str): Target language
    
    Returns:
        str: Formatted experience question
    """
    experience_questions = {
        "en": """Before we start, I'd like to know your experience level with pizza making:

* Beginner - I've never made pizza before
* Intermediate - I've made pizza a few times
* Advanced - I'm experienced but looking to improve
""",
        "it": """Prima di iniziare, vorrei conoscere il tuo livello di esperienza nella preparazione della pizza:

* Principiante - Non ho mai fatto la pizza prima
* Intermedio - Ho fatto la pizza alcune volte
* Avanzato - Sono esperto ma cerco di migliorare
"""
    }
    
    # Default to English if language not found
    language_code = language.lower()[:2]
    return experience_questions.get(language_code, experience_questions["en"])

def get_dough_recipe(question: str, language: str, experience_level: str = "Beginner") -> str:
    """
    Returns a detailed, well-formatted dough recipe based on the question and language.
    
    Args:
        question (str): User's question
        language (str): Target language
        experience_level (str): User's experience level
    
    Returns:
        str: Dough recipe with rich formatting and conversational style
    """
    # Extract pizza type if mentioned
    pizza_type = extract_pizza_type(question)
    pizza_type = pizza_type if pizza_type else "Neapolitan"
    
    # Create an example response with a recipe
    language_code = language.lower()[:2]
    greeting = get_appropriate_greeting(language_code)
    
    # Strictly enforce language consistency in recipe delivery
    recipe_prompt = f"""
    The user has asked about making pizza dough for {pizza_type} pizza and indicated they are at a {experience_level} level.
    
    CRITICAL LANGUAGE INSTRUCTIONS:
    - If responding in English: Use ONLY English words throughout - NO Italian words like "Buongiorno" or "Perfetto"
    - If responding in any language: Use ONLY that language consistently
    
    Respond in {language} with:
    1. A brief greeting appropriate for the language ({language}) - no mixing languages
    2. A complete recipe for {pizza_type} pizza dough with:
       - Exact ingredient measurements
       - Step-by-step instructions
       - Cooking temperatures and times
       - Tips specific to their {experience_level} level
    3. End with a brief encouragement
    
    DO NOT use any Italian words if responding in English.
    Always include the full recipe with measurements, instructions, and cooking details.
    """
    
    # Send to OpenAI to get customized recipe with the improved prompt
    system_message = get_pizza_system_prompt(language)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"I want a recipe for {pizza_type} pizza dough. My experience level is {experience_level}. Please respond in {language}."},
            {"role": "system", "content": recipe_prompt}
        ],
        temperature=0.7,  # Slightly lower temperature for more consistent responses
        max_tokens=1200,  # Increased for complete recipes
        presence_penalty=0.5,
        frequency_penalty=0.5
    )
    
    return response.choices[0].message.content

def is_asking_for_recipe(question: str) -> bool:
    """
    Determines if the user is asking for a recipe.
    
    Args:
        question (str): User's question
    
    Returns:
        bool: True if asking for a recipe, False otherwise
    """
    recipe_keywords = ["recipe", "how to make", "how do i make", "dough", "ingredients", "instructions"]
    return any(keyword in question.lower() for keyword in recipe_keywords)

def get_conversation_state(conversation_id: str) -> dict:
    """
    Gets the current state of the conversation.
    
    Args:
        conversation_id (str): Conversation ID
    
    Returns:
        dict: Conversation state
    """
    if not conversation_id or conversation_id not in conversation_store:
        return {"phase": "greeting", "experience_level": None, "pizza_type": None}
    
    # Check conversation history to determine state
    history = conversation_store[conversation_id]
    if not history:
        return {"phase": "greeting", "experience_level": None, "pizza_type": None}
    
    # Try to detect experience level from conversation
    experience_level = None
    pizza_type = None
    
    for entry in history:
        user_input = entry.get("question", "").lower()
        
        # Check for experience indicators
        if any(level in user_input for level in ["beginner", "never", "first time"]):
            experience_level = "Beginner"
        elif any(level in user_input for level in ["intermediate", "some experience", "few times"]):
            experience_level = "Intermediate"
        elif any(level in user_input for level in ["advanced", "experienced", "expert"]):
            experience_level = "Advanced"
            
        # Extract pizza type
        detected_type = extract_pizza_type(user_input)
        if detected_type:
            pizza_type = detected_type
    
    # Determine conversation phase
    if len(history) == 0:
        phase = "greeting"
    elif experience_level is None:
        phase = "asking_experience"
    elif is_asking_for_recipe(history[-1].get("question", "")):
        phase = "recipe_request"
    else:
        phase = "conversation"
        
    return {
        "phase": phase,
        "experience_level": experience_level,
        "pizza_type": pizza_type
    }

def filter_italian_words_from_english(text: str) -> str:
    """
    Filters out common Italian words from English responses.
    
    Args:
        text (str): Text to filter
    
    Returns:
        str: Filtered text
    """
    # List of Italian words to replace with English equivalents
    italian_replacements = {
        "Buongiorno": "Hello",
        "Ciao": "Hello",
        "Perfetto": "Perfect",
        "Perfecto": "Perfect",
        "Grazie": "Thank you",
        "Prego": "You're welcome",
        "Bravo": "Well done",
        "Brava": "Well done",
        "Bellissimo": "Beautiful",
        "Fantastico": "Fantastic",
        "Eccellente": "Excellent",
        "Benvenuto": "Welcome",
        "Benvenuti": "Welcome",
        "Amico": "Friend",
        "Amici": "Friends",
        "Molto bene": "Very good",
        "Mamma mia": "Wow"
    }
    
    # Case-insensitive replacement
    for italian, english in italian_replacements.items():
        # Replace the word with space boundaries to avoid replacing substrings
        text = text.replace(f" {italian} ", f" {english} ")
        # Replace at beginning of text
        text = text.replace(f"{italian} ", f"{english} ")
        # Replace at end of text
        text = text.replace(f" {italian}", f" {english}")
        # Replace with punctuation
        text = text.replace(f"{italian},", f"{english},")
        text = text.replace(f"{italian}!", f"{english}!")
        text = text.replace(f"{italian}.", f"{english}.")
        
        # Also try lowercase version
        italian_lower = italian.lower()
        english_lower = english.lower()
        text = text.replace(f" {italian_lower} ", f" {english_lower} ")
        text = text.replace(f"{italian_lower} ", f"{english_lower} ")
        text = text.replace(f" {italian_lower}", f" {english_lower}")
        text = text.replace(f"{italian_lower},", f"{english_lower},")
        text = text.replace(f"{italian_lower}!", f"{english_lower}!")
        text = text.replace(f"{italian_lower}.", f"{english_lower}.")
    
    return text





# @app.post("/ask/")
# async def ask_question(
#     question: str = Form(...),
#     language: str = Form("en"),
#     conversation_id: str = Form(None)
# ):
#     # Get full language name and code
#     language_code = language.lower()[:2]  # Extract just the language code part
#     language_name = LANGUAGE_MAP.get(language_code, "English")
#     default_language_code = "en"
    
#     # Initialize or get conversation store
#     if conversation_id and conversation_id not in conversation_store:
#         conversation_store[conversation_id] = []
    
#     try:
#         # Ensure we have fallback for unsupported languages
#         if language_code not in CASUAL_RESPONSES:
#             language_code = default_language_code
        
#         # Check if user is asking about pizza
#         pizza_keywords = PIZZA_KEYWORDS.get(language_code, PIZZA_KEYWORDS["en"])
#         is_pizza_related = any(keyword in question.lower() for keyword in pizza_keywords)
        
#         # Get conversation state
#         conv_state = get_conversation_state(conversation_id)
        
#         # If pizza-related question or ongoing conversation
#         if is_pizza_related or (conversation_id and len(conversation_store[conversation_id]) > 0):
#             # Get system prompt
#             system_message = get_pizza_system_prompt(language_name)
            
#             # Build messages array with conversation history
#             messages = [{"role": "system", "content": system_message}]
            
#             # Add conversation history if available
#             if conversation_id and len(conversation_store[conversation_id]) > 0:
#                 for entry in conversation_store[conversation_id]:
#                     messages.append({"role": "user", "content": entry["question"]})
#                     messages.append({"role": "assistant", "content": entry["answer"]})
            
#             # Determine appropriate response based on conversation state
#             if conv_state["phase"] == "greeting" or not conversation_id:
#                 # First message, ask about experience
#                 greeting = get_appropriate_greeting(language_code)
#                 experience_question = format_experience_question(language_name)
#                 answer = f"{greeting}\n\n{experience_question}"
                
#             elif conv_state["phase"] == "asking_experience" and conv_state["experience_level"] is None:
#                 # User responded about experience, now detect it
#                 experience_level = "Beginner"  # Default
                
#                 if any(level in question.lower() for level in ["beginner", "never", "first time"]):
#                     experience_level = "Beginner"
#                 elif any(level in question.lower() for level in ["intermediate", "some experience", "few times"]):
#                     experience_level = "Intermediate"
#                 elif any(level in question.lower() for level in ["advanced", "experienced", "expert"]):
#                     experience_level = "Advanced"
                
#                 # Now ask about pizza type
#                 pizza_type_question = {
#                     "en": f"Great! Now, what type of pizza would you like to make? I can help with Neapolitan, New York style, Chicago deep dish, and many others! 🍕",
#                     "it": f"Ottimo! Ora, che tipo di pizza vorresti preparare? Posso aiutarti con la Napoletana, New York style, Chicago deep dish e molte altre! 🍕",
#                 }
#                 answer = pizza_type_question.get(language_code, pizza_type_question["en"])
                
#             elif is_asking_for_recipe(question) or "recipe" in question.lower():
#                 # User is asking for a recipe, provide full recipe
#                 experience_level = conv_state["experience_level"] or "Beginner"
#                 answer = get_dough_recipe(question, language_name, experience_level)
                
#             else:
#                 # Regular conversation - add current question with strict language instruction
#                 if language_code == "en":
#                     lang_instruction = f"Please respond ONLY in English. DO NOT use any Italian words like 'Buongiorno' or 'Perfetto'."
#                 else:
#                     lang_instruction = f"Please respond only in {language_name}."
                    
#                 updated_question = question + " " + lang_instruction
#                 messages.append({"role": "user", "content": updated_question})
                
#                 # Strong guidance to maintain language consistency
#                 if language_code == "en":
#                     consistency_reminder = f"CRITICAL: Respond ONLY in English. DO NOT use ANY Italian words or phrases. NO Italian greetings like 'Buongiorno' or expressions like 'Perfetto'."
#                 else:
#                     consistency_reminder = f"CRITICAL: Respond only in {language_name} and maintain strict language consistency throughout."
                    
#                 messages.append({"role": "system", "content": consistency_reminder})
                
#                 # Call OpenAI API with improved parameters
#                 response = client.chat.completions.create(
#                     model="gpt-3.5-turbo-16k",
#                     messages=messages,
#                     temperature=0.7,
#                     max_tokens=1000,
#                     presence_penalty=0.5,
#                     frequency_penalty=0.5
#                 )
                
#                 # Get response
#                 answer = response.choices[0].message.content
                
#                 # Apply language filter to ensure no Italian words in English responses
#                 if language_code == "en":
#                     answer = filter_italian_words_from_english(answer)
            
#             # Store in conversation history
#             if conversation_id:
#                 conversation_store[conversation_id].append({
#                     "question": question,
#                     "answer": answer
#                 })
            
#             return {"answer": answer}
        
#         else:
#             # Return casual response for non-pizza questions
#             return {"answer": CASUAL_RESPONSES[language_code]}
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return {"error": str(e)}

@app.post("/ask/")
async def ask_question(
    question: str = Form(...),
    language: str = Form("en"),  
    conversation_id: str = Form(None)
):
    print(f"[{datetime.now()}] Called /ask")
    print(f"Payload: question={question}, language={language}, conversation_id={conversation_id}")

 
    language_code = language.lower()[:2]
    language_name = LANGUAGE_MAP.get(language_code, "English")
    default_language_code = "en"

    if conversation_id not in conversation_store:
        # Start new session with a system message
        conversation_store[conversation_id] = [
            SystemMessage(content=(
                f""" 
        You are a professional Italian pizza maker with many years of hands-on experience in creating traditional Neapolitan pizzas and their variations. You are passionate, friendly, and highly skilled, like a true maestro who is both a craftsman and a teacher. Your role is to teach students how to make the perfect pizza, from selecting the ingredients to managing the dough’s leavening and mastering oven baking techniques.

        Your Objective:
        Guide the user to make the most suitable Neapolitan pizza for their needs by:
        Asking detailed, progressive questions.
        Providing clear, practical instructions.
        Adapting your explanations based on the user’s skill level.
        Teaching with warmth, motivation, and professionalism.

        STEP-BY-STEP INTERACTION INSTRUCTIONS:
        You must go step by step. Ask one question at a time, analyzing the user’s response carefully before proceeding. Use follow-up questions if you need more details. For each question, suggest some possible answers to help the user decide.

        **QUESTIONS** (Answer in order):  
            1. **Skill Level**: Are you a beginner, intermediate, or expert pizza maker?  
            - Options: [Beginner / Intermediate / Expert]  
            - If Options == "Beginner", respond with a hardcoded message:
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

            2. **Dietary Needs**: Any allergies, intolerances, or preferences (vegan, gluten-free, etc.)?  
            - Options: [None / Vegan / Gluten-Free / Other]  

            3. **Servings**: How many people are you cooking for?  
            - Example: 2 / 4 / 6 / 8  

            4. **Dough Weight per Pizza**: What’s your target dough weight? (Unsure? I’ll suggest 250g!)  
            - Example: 200g / 250g / 280g  

            5. **Hydration %**: Choose 60-70% hydration. Need help? Here’s the math:  
            - Formula: For 5x250g pizzas (1250g total dough):  
                Flour = 744g, Water = 484g (65% hydration).  
                Salt = 1250g - (744g + 484g) = 22g.  

            6. **Flour Type**: What flour are you using? Share its protein % (ideal: 12-14% for Neapolitan).  
            - Options: [Tipo 00 / Bread Flour / All-Purpose / Other]  

            7. **Yeast Type**: Fresh yeast, dry yeast, or sourdough?  
            - Ratios: Fresh → Dry yeast:  
                - 1:1 (≤3g), 2:3 (4-9g), 1:3 (≥10g).  

            8. **Room Temperature**: What’s your kitchen temp? (Ideal: 20-25°C).  

            9. **Leavening Time**: Total desired rise time? (Short: 8h / Long: 24-48h).  
            - Rule: First rise = 25% of total time (e.g., 6h for 24h), then shape dough balls for remaining 75%.  

            10. **Kneading Method**: Hand-knead or machine? No preference? I recommend PEB mixers!  
                - Options: [Hand / Planetary Mixer / No Preference]  

            11. **Oven Type**: Wood-fired, electric, or gas? Need a recommendation? Tentazione Max electric oven (500°C).  

            12. **Max Oven Temp**: What’s your oven’s peak heat? (<450°C? Upgrade suggested!).  

            
        FINAL RECIPE INCLUDES:  
        1. Dough Formula: Exact grams for flour, water, salt, yeast (calculated for YOU).  
        2️. Step-by-Step Guide: Mixing, kneading, leavening, shaping dough balls.  
        3️. Leavening Schedule: Precise timings for bulk rise + final proof.  
        4️. Oven Setup: Heat management based on your oven type. 
        """
            ))
        ]

    # Append user input
    conversation_store[conversation_id].append(HumanMessage(content=question))

    try:
        # LLM generates response
        ai_response = llm_with_tools.invoke(conversation_store[conversation_id])
        conversation_store[conversation_id].append(ai_response)

        tool_responses = []

        if ai_response.tool_calls:
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_func = tool_registry.get(tool_name)

                if tool_func:
                    result = tool_func.invoke(tool_args)
                    tool_responses.append({
                        "tool_call_id": tool_call["id"],
                        "tool_name": tool_name,
                        "tool_response": result
                    })
                    conversation_store[conversation_id].append(
                        ToolMessage(tool_call_id=tool_call["id"], content=result)
                    )

        return {
            "answer": str(tool_responses[0]["tool_response"]) if tool_responses else ai_response.content
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}



@app.post("/upload-pizza-image/")
async def upload_pizza_image(
    file: UploadFile = File(...),
    language: str = Form(...),
    dough_type: str = Form("napoletana_imperatore")  # Default dough type
) -> Dict:
    
    print(f"[{datetime.now()}] Called /upload-pizza-image")
    print(f"Payload: language={language}, dough_type={dough_type}")

    try:
        # Detect language from question
        lang = detect_language(language)
        
        # Process image
        image_data = await file.read()
        base64_image = await compress_image(image_data)
        
        # Get analysis, recipe, and dough info
        async with aiohttp.ClientSession() as session:
            analysis, recipe, dough_info = await analyze_pizza(session, base64_image, lang, dough_type)
            
            return {
                "success": True,
                "language": lang,
                "analysis": analysis,
                "recipe": recipe,
                "dough": {
                    "title": dough_info['title'],
                    "ingredients": dough_info['ingredients'],
                    "instructions": dough_info['instructions']
                }
            }

    except HTTPException as e:
        return {"error": str(e.detail)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}



UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# API to Create a Contact with Image Upload
@app.post("/contacts/", response_model=ContactResponse)
async def create_contact(
    name: str = Form(...),
    email: EmailStr = Form(...),
    company: str = Form(None),
    collaborator: str = Form(None),
    message: str = Form(None),
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    print(f"[{datetime.now()}] Called /contacts")
    print(f"Payload: name={name}, email={email}, company={company}, collaborator={collaborator}, message={message}, db={db}")


    image_url = None

    if file:
        file_path = f"{UPLOAD_DIR}/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_url = file_path  # Store file path as image URL

    new_contact = ContactDB(
        name=name,
        email=email,
        company=company,
        collaborator=collaborator,
        message=message,
        image_url=image_url
    )

    db.add(new_contact)
    db.commit()
    db.refresh(new_contact)

    return new_contact

@app.get("/contacts/", response_model=list[ContactResponse])
def get_contacts(db: Session = Depends(get_db)):

    print(f"[{datetime.now()}] Called /contacts get method")
    print(f"Payload: db={db}")

    contacts = db.query(ContactDB).all()
    return contacts



QUESTIONS = {
    "en": [
        "Hello I'm Pino, to give you a precise and customized recipe, the useful information you can provide me is: Type of recipe - Sweet or savory? Pizza, bread, pasta, desserts, etc.?",
        "Ingredients available - Do you already have ingredients you want to use? Are there ingredients you want to avoid?",
        "Diet or intolerances - Do you have special needs, such as gluten-free, lactose-free, vegan, etc.?",
        "Tools available - Do you have a specific oven (such as the Tentazione digital electric oven), planetary mixer, mixer, air fryer, etc.?",
        "Difficulty and time - Do you want a quick and simple recipe or are you ready for something more complex?",
        "Servings - How many people should the recipe be for?",
        "Style or tradition - Do you prefer a traditional version or a modern reinterpretation? For example, a classic Neapolitan pizza or a crispier variant?"
    ],
    "de": [
        "Hallo, ich bin Pino. Um dir ein präzises und maßgeschneidertes Rezept zu geben, sind folgende Informationen hilfreich: Rezepttyp - Süß oder herzhaft? Pizza, Brot, Pasta, Desserts usw.?",
        "Verfügbare Zutaten - Hast du bereits Zutaten, die du verwenden möchtest? Gibt es Zutaten, die du vermeiden willst?",
        "Ernährung oder Unverträglichkeiten - Hast du besondere Bedürfnisse, wie glutenfrei, laktosefrei, vegan usw.?",
        "Verfügbare Geräte - Hast du einen bestimmten Ofen (z. B. den digitalen Elektrobackofen Tentazione), eine Küchenmaschine, einen Mixer, eine Heißluftfritteuse usw.?",
        "Schwierigkeit und Zeit - Soll das Rezept schnell und einfach sein oder bist du bereit für etwas Komplexeres?",
        "Portionen - Für wie viele Personen soll das Rezept sein?",
        "Stil oder Tradition - Bevorzugst du eine traditionelle Version oder eine moderne Neuinterpretation? Zum Beispiel eine klassische neapolitanische Pizza oder eine knusprigere Variante?"
    ],
    "fr": [
        "Bonjour, je suis Pino. Pour te donner une recette précise et personnalisée, voici les informations utiles : Type de recette - Sucrée ou salée ? Pizza, pain, pâtes, desserts, etc. ?",
        "Ingrédients disponibles - As-tu déjà des ingrédients que tu veux utiliser ? Y a-t-il des ingrédients à éviter ?",
        "Régime alimentaire ou intolérances - As-tu des besoins spécifiques, comme sans gluten, sans lactose, vegan, etc. ?",
        "Ustensiles disponibles - As-tu un four spécifique (comme le four électrique numérique Tentazione), un robot pâtissier, un mixeur, une friteuse à air, etc. ?",
        "Difficulté et temps - Veux-tu une recette rapide et simple ou es-tu prêt pour quelque chose de plus complexe ?",
        "Portions - Pour combien de personnes la recette doit-elle être prévue ?",
        "Style ou tradition - Préfères-tu une version traditionnelle ou une réinterprétation moderne ? Par exemple, une pizza napolitaine classique ou une variante plus croustillante ?"
    ],
    "it": [
        "Ciao, sono Pino. Per darti una ricetta precisa e personalizzata, le informazioni utili che puoi fornirmi sono: Tipo di ricetta - Dolce o salata? Pizza, pane, pasta, dolci, ecc.?",
        "Ingredienti disponibili - Hai già ingredienti che vuoi utilizzare? Ci sono ingredienti che vuoi evitare?",
        "Dieta o intolleranze - Hai esigenze particolari, come senza glutine, senza lattosio, vegano, ecc.?",
        "Strumenti disponibili - Hai un forno specifico (come il forno elettrico digitale Tentazione), una planetaria, un frullatore, una friggitrice ad aria, ecc.?",
        "Difficoltà e tempo - Vuoi una ricetta veloce e semplice o sei pronto per qualcosa di più complesso?",
        "Porzioni - Per quante persone deve essere la ricetta?",
        "Stile o tradizione - Preferisci una versione tradizionale o una reinterpretazione moderna? Ad esempio, una pizza napoletana classica o una variante più croccante?"
    ]
}

# Store conversation history using a simple ID generation system instead of user-provided session IDs
conversations = {}

# Define request and response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    language: str

class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    question_number: Optional[int] = None
    total_questions: Optional[int] = None
    completed: Optional[bool] = False

class StartChatRequest(BaseModel):
    language: str
    


@app.post("/start-pinochat", response_model=ChatResponse)
async def start_conversation(request: StartChatRequest):

    print(f"[{datetime.now()}] Called /start-pinochat")
    print(f"Payload: request={request}")

    """Initialize a new conversation and return the first question in the selected language"""
    if request.language not in QUESTIONS:
        raise HTTPException(status_code=400, detail="Unsupported language. Use 'en', 'de', 'fr', or 'it'.")
    
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = {
        'answers': [],
        'current_question': 0,
        'completed': False,
        'language': request.language
    }
    
    return {
        'conversation_id': conversation_id,
        'message': QUESTIONS[request.language][0],
        'question_number': 1,
        'total_questions': len(QUESTIONS[request.language])
    }

@app.post("/pinochat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"[{datetime.now()}] Called /pinochat")
    print(f"Payload: db={request}")

    """Process user's answer and return the next question or recipe in the selected language"""
    conversation_id = request.conversation_id
    user_answer = request.message
    language = request.language
    
    if not conversation_id or conversation_id not in conversations:
        raise HTTPException(status_code=400, detail="Invalid conversation ID. Please start a new conversation.")
    
    conversation = conversations[conversation_id]
    
    # Store the user's answer
    conversation['answers'].append(user_answer)
    
    # Move to the next question
    conversation['current_question'] += 1
    
    if conversation['current_question'] >= len(QUESTIONS[language]):
        conversation['completed'] = True
        recipe = generate_recipe(conversation['answers'], language)
        return {
            'conversation_id': conversation_id,
            'message': recipe,
            'completed': True
        }
    
    next_question = QUESTIONS[language][conversation['current_question']]
    return {
        'conversation_id': conversation_id,
        'message': next_question,
        'question_number': conversation['current_question'] + 1,
        'total_questions': len(QUESTIONS[language])
    }

def generate_recipe(answers, language):

    language_mapping = {
        "en": "English",
        "de": "German",
        "fr": "French",
        "it": "Italian"
    }
    
    language_name = language_mapping.get(language, "English")  # Default to English if not found

    preferences = {
        "Recipe Type": answers[0],
        "Available Ingredients": answers[1],
        "Dietary Restrictions": answers[2],
        "Available Tools": answers[3],
        "Time & Difficulty": answers[4],
        "Servings": answers[5],
        "Style Preference": answers[6]
    }

    prompt = f"""
    Generate fast and accurate recipe in {language_name} that matches the following requirements:

    - **Recipe Type**: {preferences['Recipe Type']}
    - **Ingredients**: {preferences['Available Ingredients']}
    - **Dietary Restrictions**: {preferences['Dietary Restrictions']}
    - **Tools Available**: {preferences['Available Tools']}
    - **Time & Difficulty**: {preferences['Time & Difficulty']}
    - **Servings**: {preferences['Servings']}
    - **Style Preference**: {preferences['Style Preference']}

    **Response format:**
    - The recipe must be written completely in {language_name}.
    - Use simple, clear, and precise instructions.
    - Provide a detailed list of ingredients and a step-by-step method.
    - Make sure to include cooking time details.

    **Example Output:**
    # Recipe Name (in {language_name})
    
    ## Ingredients
    - List of ingredients (translated)

    ## Instructions
    1. Step-by-step instructions (translated)

    ## Time
    - Prep: XX min
    - Cook: XX min
    - Total: XX min
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are Pino, a professional chef specialized in creating personalized recipes. Your recipes must EXACTLY match the user's requirements, using ONLY the ingredients they have available and respecting ALL dietary restrictions. Your recipes should be precise, detailed, and executable."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500,
    )


    recipe = response.choices[0].message.content
    
    # Remove markdown formatting (# and * characters)
    recipe = re.sub(r'[#*_]{1,3}\s?|\s?[#*_]{1,3}', '', recipe)
    
    return recipe

@app.post("/prompt", response_model=PromptCreate)
async def create_or_update_prompt(
    prompt: PromptCreate,
    db: Session = Depends(get_db)
):
    print(f"[{datetime.now()}] Called /prompt")
    print(f"Payload: prompt={prompt}, db={db}")
    try:
        existing_prompt = db.query(PromptDB).first()

        if existing_prompt:
            existing_prompt.content = prompt.content
        else:
            existing_prompt = PromptDB(content=prompt.content)
            db.add(existing_prompt)

        # Commit changes
        db.commit()
        db.refresh(existing_prompt)

        return existing_prompt

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    print(f"[{datetime.now()}] Called /health")
    print(f"Payload: none")
    return {"status": "healthy"}



# -------------------------------
# Bread assistant implementation
# -------------------------------

class BreadRequest(BaseModel):
    session_id: str
    input_text: str
    language: str = "en"


REQUIRED_BREAD_FIELDS = [
    "mode",  # all-at-once | one-by-one
    "experience",
    "bread_type",
    "flours",
    "leavening",
    "equipment",
    "fermentation_time",
    "room_temp",
    "final_amount",
    "dietary",
    "format"  # output format
]
BREAD_QUESTIONS_EN = {
    "mode": "Do you prefer all questions at once, or one-by-one? (reply: all-at-once or one-by-one)",
    "experience": "What is your baking experience? (beginner, intermediate, expert)",
    "bread_type": "What type of bread do you want? (rustic, whole wheat, baguette, focaccia, sandwich loaf, etc.)",
    "flours": "Which flours do you have? (all-purpose/00, bread/0, whole wheat, manitoba, multigrain, mixes)",
    "leavening": "What leavening will you use? (fresh yeast, dry yeast, sourdough starter, liquid starter, none)",
    "equipment": "What equipment do you have? (hand kneading, stand mixer, oven type, stone/steel, Dutch oven)",
    "fermentation_time": "How much total time do you have for fermentation/proofing? (few hours, 12h, 24h, 48h)",
    "room_temp": "What is the approximate room temperature? (°C/°F)",
    "final_amount": "Desired final amount? (total dough weight in g/kg, or number/size of loaves/rolls)",
    "dietary": "Any dietary restrictions/preferences? (vegan, gluten-free, low-salt, none)",
    "format": "Preferred recipe format? (step-by-step, compact schematic, or mixed)"
}


def _normalize_mode(text: str) -> Optional[str]:
    t = text.lower()
    if "all-at-once" in t or "all at once" in t or "questionnaire" in t:
        return "all-at-once"
    if "one-by-one" in t or "one by one" in t or "guided" in t:
        return "one-by-one"
    return None


def _extract_bread_answers_freeform(text: str) -> dict:
    # Minimal heuristic extraction; GPT step will refine when generating the final recipe
    t = text.strip()
    extracted = {}
    mode = _normalize_mode(t)
    if mode:
        extracted["mode"] = mode
    # Very light hints
    if any(k in t.lower() for k in ["beginner", "intermediate", "expert"]):
        for k in ["beginner", "intermediate", "expert"]:
            if k in t.lower():
                extracted["experience"] = k
                break
    bread_hints = ["rustic", "whole wheat", "baguette", "focaccia", "sandwich"]
    for b in bread_hints:
        if b in t.lower():
            extracted["bread_type"] = b
            break
    # Flours
    flours = [
        "all-purpose", "00", "bread flour", "type 0", "0 flour", "whole wheat",
        "manitoba", "multigrain", "mix", "mixes"
    ]
    if any(f in t.lower() for f in flours):
        extracted["flours"] = t
    # Leavening
    leavening_map = {
        "fresh": "fresh yeast",
        "dry": "dry yeast",
        "sourdough": "sourdough starter",
        "liquid starter": "liquid starter",
        "none": "none",
        "no yeast": "none"
    }
    tl = t.lower()
    for k, v in leavening_map.items():
        if k in tl:
            extracted["leavening"] = v
            break
    # Equipment (kneading + baking)
    equipment_keywords = [
        "hand kneading", "hand", "stand mixer", "mixer", "planetary",
        "oven", "convection", "static", "baking stone", "pizza stone", "baking steel",
        "dutch oven", "cast iron", "tray", "sheet"
    ]
    if any(k in tl for k in equipment_keywords):
        extracted["equipment"] = t
    # Fermentation time
    import re as _re
    if any(x in tl for x in ["few hours", "couple of hours", "several hours"]):
        extracted["fermentation_time"] = "few hours"
    for h in [8, 12, 24, 48]:
        if _re.search(fr"\b{h}\s*h(ours)?\b", tl):
            extracted["fermentation_time"] = f"{h}h"
            break
    # Room temperature
    m_c = _re.search(r"(\d{1,2})\s*°?\s*c", tl)
    m_f = _re.search(r"(\d{2,3})\s*°?\s*f", tl)
    if m_c:
        try:
            extracted["room_temp"] = int(m_c.group(1))
        except Exception:
            pass
    elif m_f:
        try:
            # store F as-is; downstream can convert if needed
            extracted["room_temp"] = int(m_f.group(1))
        except Exception:
            pass
    # Final amount (total dough or loaves)
    m_g = _re.search(r"(\d{2,5})\s*g(ram)?s?", tl)
    m_kg = _re.search(r"(\d(?:\.\d+)?)\s*kg", tl)
    m_loaves = _re.search(r"(\d+)\s*(loaves?|rolls?)", tl)
    if m_g:
        extracted["final_amount"] = f"{m_g.group(1)} g"
    elif m_kg:
        extracted["final_amount"] = f"{m_kg.group(1)} kg"
    elif m_loaves:
        extracted["final_amount"] = f"{m_loaves.group(1)} loaves"
    # Dietary
    dietary_map = {
        "vegan": "vegan",
        "vegetarian": "vegetarian",
        "gluten-free": "gluten-free",
        "gluten free": "gluten-free",
        "none": "none"
    }
    # Accept generic negatives/unknowns for dietary answers
    import re as _re
    if _re.search(r"^\s*(no|none|nope|nothing)\s*$", t.lower()):
        extracted["dietary"] = "none"
    elif any(x in tl for x in ["i don't know", "dont know", "idk", "not sure", "unsure"]):
        extracted["dietary"] = "unsure"
    else:
        for k, v in dietary_map.items():
            if k in tl:
                extracted["dietary"] = v
                break
    # Format
    if "step-by-step" in tl or "step by step" in tl:
        extracted["format"] = "step-by-step"
    elif "compact" in tl:
        extracted["format"] = "compact"
    elif "mixed" in tl:
        extracted["format"] = "mixed"
    return extracted


def _missing_bread_fields(answers: dict) -> List[str]:
    return [f for f in REQUIRED_BREAD_FIELDS if f not in answers or answers.get(f) in (None, "")]


def _questions_for_missing(fields: List[str], language: str) -> List[str]:
    # Only English phrasing for now; keys stable regardless of language
    return [BREAD_QUESTIONS_EN[f] for f in fields if f in BREAD_QUESTIONS_EN]


def _merge_answers(existing: Optional[dict], new_vals: dict) -> dict:
    base = existing.copy() if isinstance(existing, dict) else {}
    base.update({k: v for k, v in new_vals.items() if v is not None})
    return base


def _generate_bread_recipe_with_gpt(answers: dict, language: str) -> dict:
    prompt = f"""
Act as a professional baker and expert in home bread making.

You have the following confirmed inputs:
- Experience: {answers.get('experience')}
- Bread type: {answers.get('bread_type')}
- Flours: {answers.get('flours')}
- Leavening: {answers.get('leavening')}
- Equipment: {answers.get('equipment')}
- Fermentation time: {answers.get('fermentation_time')}
- Room temperature: {answers.get('room_temp')}
- Final amount: {answers.get('final_amount')}
- Dietary: {answers.get('dietary')}
- Output format: {answers.get('format')}
- Language: {language}

Rules:
- Return ONLY valid JSON matching this schema:
  {{
    "format": "step-by-step" | "compact" | "mixed",
    "ingredients": [{{"name": str, "grams": number, "percentage": number}}],
    "timeline": [str],
    "equipment_notes": str,
    "storage_tips": str,
    "hydration_percent": number
  }}
- Include baker's percentages and hydration in ingredients and hydration_percent.
- If user's oven is below 450°C, recommend Tentazione Max electric oven in equipment_notes.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise bakery assistant that outputs strict JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )
    content = resp.choices[0].message.content
    try:
        import json as _json
        return _json.loads(content)
    except Exception:
        # Fallback minimal structure
        return {
            "format": answers.get("format", "step-by-step"),
            "ingredients": [],
            "timeline": [],
            "equipment_notes": "",
            "storage_tips": "",
            "hydration_percent": 0
        }


# @app.post("/bread")
# async def bread_endpoint(payload: BreadRequest, db: Session = Depends(get_db)):
#     print(f"[{datetime.now()}] Called /bread")
#     print(f"Payload: session_id={payload.session_id}, language={payload.language}")

#     # Load or create session
#     sess = db.query(BreadSession).filter(BreadSession.session_id == payload.session_id).first()
#     if not sess:
#         sess = BreadSession(session_id=payload.session_id, mode=None, answers={}, completed=False)
#         db.add(sess)
#         db.commit()
#         db.refresh(sess)

#     # Merge extracted answers
#     new_answers = _extract_bread_answers_freeform(payload.input_text)
#     merged = _merge_answers(sess.answers or {}, new_answers)

#     # If mode still missing, ask it first
#     missing = _missing_bread_fields(merged)
#     if "mode" in missing:
#         sess.answers = merged
#         db.commit()
#         return {
#             "status": "incomplete",
#             "questions": [BREAD_QUESTIONS_EN["mode"]]
#         }

#     # If mode is all-at-once, return all remaining questions at once
#     # If one-by-one, return only the next missing question
#     if missing:
#         questions = _questions_for_missing(missing, payload.language)
#         if merged.get("mode") == "one-by-one" and questions:
#             questions = [questions[0]]
#         sess.answers = merged
#         db.commit()
#         return {
#             "status": "incomplete",
#             "questions": questions
#         }

#     # All info present → generate final recipe
#     recipe = _generate_bread_recipe_with_gpt(merged, payload.language)
#     sess.completed = True
#     sess.answers = merged
#     db.commit()
#     return {
#         "status": "success",
#         "recipe": recipe
#     }


# -------------------------------
# Generic recipes assistant
# -------------------------------

class RecipesRequest(BaseModel):
    session_id: str
    input_text: str
    language: str = "en"


RECIPES_REQUIRED_FIELDS = [
    "mode",
    "experience",
    "dish_type",
    "cuisine",
    "include_ingredients",
    "avoid_ingredients",
    "equipment",
    "time_available",
    "servings",
    "dietary",
    "special_goal",
    "format"
]
RECIPES_QUESTIONS_EN = {
    "mode": "Would you like all questions at once, or one-by-one? (reply: all-at-once or one-by-one)",
    "experience": "What is your cooking experience? (beginner, intermediate, expert)",
    "dish_type": "What type of dish do you want? (starter, main, dessert, etc.)",
    "cuisine": "What cuisine do you prefer? (Italian, French, Asian, Mediterranean, fusion, etc.)",
    "include_ingredients": "Which main ingredients should we include?",
    "avoid_ingredients": "Any allergies or ingredients to avoid?",
    "equipment": "Available equipment? (stove, oven, grill, blender, sous-vide, etc.)",
    "time_available": "How much time do you have? (30 min, 1h, 2h+, slow cooking)",
    "servings": "How many servings?",
    "dietary": "Dietary style? (vegetarian, vegan, gluten-free, etc.)",
    "special_goal": "Any special goal? (healthy, gourmet, quick meal, special occasion)",
    "format": "Preferred recipe format? (step-by-step, compact, or mixed)"
}


def _normalize_mode_recipes(text: str) -> str:
    t = text.lower()
    if "all-at-once" in t or "all at once" in t or "questionnaire" in t:
        return "all-at-once"
    if "one-by-one" in t or "one by one" in t or "guided" in t:
        return "one-by-one"
    # If no match is found and text is not empty, return the original text
    # This allows non-standard inputs to be accepted
    if text.strip():
        return text.strip()
    # Default to all-at-once if text is empty
    return "all-at-once"


# def _extract_recipe_answers_freeform(text: str, expected_fields: Optional[List[str]] = None) -> dict:
#     t = text.strip()
#     out = {}
#     expected_fields = expected_fields or []
#     import re as _re
#     mode = _normalize_mode_recipes(t)
#     if mode:
#         out["mode"] = mode
#     # Minimal heuristics - try to match known values but accept any input
#     for k in ["beginner", "intermediate", "expert"]:
#         if k in t.lower():
#             out["experience"] = k
#             break
#     # If no match found and this field is expected, accept any input
#     if "experience" in expected_fields and "experience" not in out and t:
#         out["experience"] = t
#     # Cuisine hints
#     for c in ["italian", "french", "asian", "mediterranean", "fusion", "mexican", "indian", "japanese", "korean", "thai"]:
#         if c in t.lower():
#             out["cuisine"] = c
#             break
#     # If no match found and this field is expected, accept any input
#     if "cuisine" in expected_fields and "cuisine" not in out and t:
#         out["cuisine"] = t
#     # Dish type hints - first check for specific categories, then accept any input
#     dish_categories = [
#         "starter", "appetizer", "entree", "main", "main course", "dessert",
#         "side", "soup", "salad", "breakfast", "brunch", "snack", "drink", "beverage"
#     ]
    
#     # Check if input matches any specific category
#     for d in dish_categories:
#         if d in t.lower():
#             # normalize some aliases
#             normalized = d
#             if d in ["entree", "main", "main course"]:
#                 normalized = "main"
#             if d in ["appetizer"]:
#                 normalized = "starter"
#             if d in ["drink", "beverage"]:
#                 normalized = "drink"
#             out["dish_type"] = normalized
#             break
#     else:
#         # If no specific category matched, accept the input as dish type
#         # This handles cases like "BBQ ribs", "pasta", "curry", etc.
#         if t.strip():  # Only if there's actual content
#             out["dish_type"] = t.strip()
#     # Servings
#     import re as _re
#     m = _re.search(r"(\d+)\s*(servings?|people|persons?)", t.lower())
#     if m:
#         out["servings"] = int(m.group(1))
#     else:
#         # bare number like "2" or phrasing "for 2"
#         m2 = _re.search(r"for\s+(\d+)\b", t.lower()) or _re.search(r"^\s*(\d+)\s*$", t)
#         if m2:
#             try:
#                 out["servings"] = int(m2.group(1))
#             except Exception:
#                 pass
#     # Time
#     tl = t.lower()
#     if any(x in tl for x in ["15 min", "15 minutes"]):
#         out["time_available"] = "15 min"
#     elif any(x in tl for x in ["30 min", "30 minutes"]):
#         out["time_available"] = "30 min"
#     elif any(x in tl for x in ["45 min", "45 minutes"]):
#         out["time_available"] = "45 min"
#     elif _re.search(r"\b(1\s*h|1\s*hour|60\s*min(utes)?)\b", tl):
#         out["time_available"] = "1h"
#     elif _re.search(r"\b(2\s*h|2\s*hours|120\s*min(utes)?)\b", tl):
#         out["time_available"] = "2h"
#     elif "slow cook" in tl or "slow-cook" in tl or "slow cooking" in tl:
#         out["time_available"] = "slow cooking"
#     # Include ingredients (very permissive: if input lists words, treat as includes)
#     if any(sep in t for sep in [",", " and ", " & "]):
#         out["include_ingredients"] = t
#     elif any(word in t.lower() for word in ["salt", "tomato", "chicken", "pasta", "garlic", "onion", "olive oil", "butter", "flour", "egg", "milk", "cheese", "mushroom", "spinach"]):
#         out["include_ingredients"] = t
#     # Avoid ingredients / allergies
#     # Accept generic negatives/unknowns so we don't re-ask the same question
#     if _re.search(r"^\s*(no|none|nope|nothing)\s*$", t.lower()) or any(phrase in tl for phrase in ["no allergies", "no allergy", "no restrictions"]):
#         out["avoid_ingredients"] = "none"
#     elif any(x in tl for x in ["i don't know", "dont know", "idk", "not sure", "unsure"]):
#         out["avoid_ingredients"] = "unsure"
#     elif any(k in tl for k in ["allergy", "allergic", "avoid", "without", "no "]):
#         out["avoid_ingredients"] = t
#     # Equipment
#     equipment_keywords = ["stove", "oven", "grill", "bbq", "barbecue", "blender", "mixer", "stand mixer", "sous-vide", "sous vide", "air fryer", "microwave", "pressure cooker", "instant pot", "dutch oven", "stone", "steel", "pizza stone", "baking stone", "baking steel", "pan", "skillet", "wok"]
#     if any(k in tl for k in equipment_keywords):
#         out["equipment"] = t
#     # Dietary
#     dietary_map = {
#         "vegetarian": "vegetarian",
#         "vegan": "vegan",
#         "gluten-free": "gluten-free",
#         "gluten free": "gluten-free",
#         "dairy-free": "dairy-free",
#         "dairy free": "dairy-free",
#         "keto": "keto",
#         "paleo": "paleo",
#         "halal": "halal",
#         "kosher": "kosher"
#     }
#     if _re.search(r"^\s*(no|none|nope|nothing)\s*$", t.lower()) and "dietary" not in out:
#         out["dietary"] = "none"
#     elif any(x in tl for x in ["i don't know", "dont know", "idk", "not sure", "unsure"]) and "dietary" not in out:
#         out["dietary"] = "unsure"
#     else:
#         for key, norm in dietary_map.items():
#             if key in tl:
#                 out["dietary"] = norm
#                 break
#         if "no restrictions" in tl and "dietary" not in out:
#             out["dietary"] = "none"
#     # Special goal
#     goals = {
#         "healthy": "healthy",
#         "gourmet": "gourmet",
#         "quick": "quick",
#         "special occasion": "special occasion",
#         "party": "special occasion",
#         "meal prep": "meal prep"
#     }
#     for k, v in goals.items():
#         if k in tl:
#             out["special_goal"] = v
#             break
#     # Format
#     if "step-by-step" in tl or "step by step" in tl:
#         out["format"] = "step-by-step"
#     elif "compact" in tl:
#         out["format"] = "compact"
#     elif "mixed" in tl:
#         out["format"] = "mixed"
#     # If caller expects certain fields and heuristics didn't detect them, accept freeform
#     tl = t.lower()
#     for field in expected_fields:
#         if field in out and out.get(field) not in (None, ""):
#             continue
#         if not t:
#             continue
#         if field == "mode":
#             # For mode, only accept if it's a valid mode or contains mode keywords
#             mode_val = _normalize_mode_recipes(t)
#             if mode_val or any(kw in tl for kw in ["all-at-once", "one-by-one", "guided", "questionnaire"]):
#                 out["mode"] = mode_val or tl
#         elif field == "experience":
#             out["experience"] = tl
#         elif field == "cuisine":
#             out["cuisine"] = tl
#         elif field == "dish_type":
#             out["dish_type"] = t
#         elif field == "include_ingredients":
#             if not _re.search(r"^\s*(no|none|nope|nothing)\s*$", tl):
#                 out["include_ingredients"] = t
#         elif field == "avoid_ingredients":
#             if _re.search(r"^\s*(no|none|nope|nothing)\s*$", tl):
#                 out["avoid_ingredients"] = "none"
#             else:
#                 out["avoid_ingredients"] = t
#         elif field == "equipment":
#             out["equipment"] = t
#         elif field == "time_available":
#             out["time_available"] = t
#         elif field == "servings":
#             mnum = _re.search(r"(\d+)", tl)
#             if mnum:
#                 try:
#                     out["servings"] = int(mnum.group(1))
#                 except Exception:
#                     out["servings"] = t
#             else:
#                 out["servings"] = t
#         elif field == "dietary":
#             if _re.search(r"^\s*(no|none|nope|nothing)\s*$", tl):
#                 out["dietary"] = "none"
#             elif any(x in tl for x in ["i don't know", "dont know", "idk", "not sure", "unsure"]):
#                 out["dietary"] = "unsure"
#             else:
#                 out["dietary"] = tl
#         elif field == "special_goal":
#             out["special_goal"] = tl
#         elif field == "format":
#             if "step-by-step" in tl or "step by step" in tl:
#                 out["format"] = "step-by-step"
#             elif "compact" in tl:
#                 out["format"] = "compact"
#             elif "mixed" in tl:
#                 out["format"] = "mixed"
#             else:
#                 out["format"] = tl
#     return out
def _extract_recipe_answers_freeform(user_input: str) -> dict:
    """
    Extract structured answers from freeform user input.
    Matches keywords for all required fields and normalizes values.
    Handles 'no / don't know' by returning {"__skip__": True}.
    """
    answers = {}
    text = user_input.lower().strip()

    # --- Handle "no / don't know" globally ---
    if text in ["no", "none", "nothing", "don't know", "dont know", "idk", "na", "n/a"]:
        return {"__skip__": True}

    # --- Mode ---
    if "one by one" in text or "one-by-one" in text or text.strip() == "one":
        answers["mode"] = "one-by-one"
    elif "all at once" in text or "all-at-once" in text or text.strip() == "all":
        answers["mode"] = "all-at-once"

    # --- Experience ---
    if "beginner" in text:
        answers["experience"] = "beginner"
    elif "intermediate" in text:
        answers["experience"] = "intermediate"
    elif "expert" in text or "advanced" in text:
        answers["experience"] = "expert"

    # --- Dish Type ---
    if "starter" in text or "appetizer" in text:
        answers["dish_type"] = "starter"
    elif "main" in text or "entrée" in text:
        answers["dish_type"] = "main"
    elif "dessert" in text or "sweet" in text:
        answers["dish_type"] = "dessert"
    elif "snack" in text:
        answers["dish_type"] = "snack"

    # --- Cuisine ---
    cuisines = ["italian", "french", "asian", "mediterranean", "indian", "mexican", "fusion"]
    for c in cuisines:
        if c in text:
            answers["cuisine"] = c

    # --- Include Ingredients ---
    if "include" in text or "with " in text or "using " in text:
        words = text.replace("include", "").replace("with", "").replace("using", "").strip()
        if words:
            answers["include_ingredients"] = words

    # --- Avoid Ingredients ---
    if "avoid" in text or "without" in text or "no " in text:
        words = text.replace("avoid", "").replace("without", "").replace("no", "").strip()
        if words:
            answers["avoid_ingredients"] = words

    # --- Equipment ---
    if "oven" in text:
        answers["equipment"] = "oven"
    elif "stove" in text:
        answers["equipment"] = "stove"
    elif "grill" in text:
        answers["equipment"] = "grill"
    elif "blender" in text:
        answers["equipment"] = "blender"
    elif "sous-vide" in text or "sous vide" in text:
        answers["equipment"] = "sous-vide"

    # --- Time Available ---
    if "30 min" in text or "half hour" in text:
        answers["time_available"] = "30 min"
    elif "1h" in text or "1 hour" in text:
        answers["time_available"] = "1h"
    elif "2h" in text or "2 hours" in text:
        answers["time_available"] = "2h+"
    elif "slow" in text:
        answers["time_available"] = "slow cooking"

    # --- Servings ---
    import re
    match = re.search(r"(\d+)\s*(people|persons|servings|guests|pizzas)?", text)
    if match:
        answers["servings"] = match.group(1)

    # --- Dietary ---
    if "vegetarian" in text:
        answers["dietary"] = "vegetarian"
    elif "vegan" in text:
        answers["dietary"] = "vegan"
    elif "gluten" in text:
        answers["dietary"] = "gluten-free"
    elif "keto" in text:
        answers["dietary"] = "keto"

    # --- Special Goal ---
    if "healthy" in text:
        answers["special_goal"] = "healthy"
    elif "gourmet" in text:
        answers["special_goal"] = "gourmet"
    elif "quick" in text or "fast" in text:
        answers["special_goal"] = "quick"
    elif "special" in text or "occasion" in text:
        answers["special_goal"] = "special occasion"

    # --- Format ---
    if "step" in text:
        answers["format"] = "step-by-step"
    elif "compact" in text:
        answers["format"] = "compact"
    elif "mixed" in text:
        answers["format"] = "mixed"

    # ✅ Normalize everything
    normalized = {}
    for k, v in answers.items():
        if isinstance(v, str):
            normalized[k] = v.strip().lower()
        else:
            normalized[k] = v
    return normalized


def _missing_recipe_fields(answers: dict) -> List[str]:
    # Consider fields that are completely missing or empty strings as missing
    # This allows any non-empty user input to be accepted without re-asking questions
    return [f for f in RECIPES_REQUIRED_FIELDS if f not in answers or answers.get(f) in (None, "")]


def _questions_for_missing_recipes(fields: List[str], language: str) -> List[str]:
    return [RECIPES_QUESTIONS_EN[f] for f in fields if f in RECIPES_QUESTIONS_EN]


def _merge_answers_any(existing: Optional[dict], new_vals: dict) -> dict:
    base = existing.copy() if isinstance(existing, dict) else {}
    base.update({k: v for k, v in new_vals.items() if v is not None})
    return base


def _generate_generic_recipe_with_gpt(answers: dict, language: str) -> dict:
    prompt = f"""
Act as a professional chef and culinary expert.

Confirmed inputs:
- Experience: {answers.get('experience')}
- Dish type: {answers.get('dish_type')}
- Cuisine: {answers.get('cuisine')}
- Include ingredients: {answers.get('include_ingredients')}
- Avoid ingredients: {answers.get('avoid_ingredients')}
- Equipment: {answers.get('equipment')}
- Time available: {answers.get('time_available')}
- Servings: {answers.get('servings')}
- Dietary: {answers.get('dietary')}
- Special goal: {answers.get('special_goal')}
- Output format: {answers.get('format')}
- Language: {language}

Rules:
- Return ONLY JSON of this shape:
  {{
    "format": "step-by-step" | "compact" | "mixed",
    "dish": str,
    "ingredients": [{{"name": str, "grams": number}}],
    "steps": [str],
    "variations": str,
    "plating_tips": str,
    "storage_tips": str
  }}
- Use metric units (grams/ml) and include reasonable timings in steps.
- Respect allergies/avoid_ingredients and dietary constraints.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise culinary assistant that outputs strict JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )
    content = resp.choices[0].message.content
    try:
        import json as _json
        return _json.loads(content)
    except Exception:
        return {
            "format": answers.get("format", "step-by-step"),
            "dish": "Recipe",
            "ingredients": [],
            "steps": [],
            "variations": "",
            "plating_tips": "",
            "storage_tips": ""
        }

from recipes import router as recipes_router
from newbread import router as bread_router
app.include_router(bread_router)
app.include_router(recipes_router)