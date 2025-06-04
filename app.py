from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
import requests
import random
import nltk
from nltk.corpus import wordnet
from PIL import Image as PILImage
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from flask_caching import Cache
from datetime import datetime
import socket
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prompt_refinement.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure static/Images directory exists
IMAGE_DIR = os.path.join(app.static_folder, 'Images')
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load environment variables from the .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
logger.info("Loading .env file from: %s", env_path)
load_dotenv(env_path)

# Load API keys from environment variables
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

# Validate the Hugging Face API key
if not HUGGINGFACE_API_KEY:
    logger.error("Hugging Face API Key not provided.")
    raise ValueError("Hugging Face API Key is required to run the application.")

# Validate the Mistral API key
if not MISTRAL_API_KEY:
    logger.error("Mistral API Key not provided.")
    raise ValueError("Mistral API Key is required to run the application.")

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
IMAGE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
PARAPHRASING_API_URL = "https://api-inference.huggingface.co/models/prithivida/parrot_paraphraser_on_T5"

# Download WordNet
nltk.download('wordnet', quiet=True)

# Initialize Mistral model
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.85, max_tokens=450, api_key=MISTRAL_API_KEY)

# Configure caching
app.config['CACHE_TYPE'] = 'SimpleCache'
cache = Cache(app)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    images = db.relationship('Image', backref='user', lazy=True)
    prompts = db.relationship('PromptHistory', backref='user', lazy=True)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PromptHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Prompt Refinement Functions
def expand_prompt(prompt, variation_count=4):
    few_shot_template = """
    Expand and refine the following prompt by adding vivid, immersive details to enrich the scene while maintaining the original theme. 
    Generate exactly {variation_count} unique variations, each 100–120 words (500–600 tokens). 
    Ensure each variation is highly distinct in perspective (e.g., time of day, angle, mood, or context) to maximize diversity. 
    Format as a numbered list (e.g., "1. Variation text"). Avoid repetition and prioritize unique, creative details.

    ### Example 1:
    **Original Prompt:** "A dragon in a cave."
    **Expanded Variations:**
    1. At dawn, a dragon with emerald scales slumbers in a cave, sunlight filtering through a crystal fissure. Glowing mushrooms illuminate scattered bones, and a chilly mist swirls around its claws. The air hums with ancient magic, and distant water drips echo like a heartbeat.
    2. Under moonlight, a crimson dragon roars in a cavern, its fiery breath lighting jagged stalactites. A hoard of gold glitters, reflecting on obsidian walls. The sulfurous air crackles, and the ground trembles with the beast’s restless energy, evoking primal fear.
    3. From a knight’s view, a shadowy dragon coils in a cave, its sapphire eyes glinting in torchlight. Runes on the walls pulse faintly, and a pile of rusted armor tells of fallen heroes. The air is thick with the scent of damp stone and dread.
    4. In a forgotten era, a golden dragon guards a cave where vines drape over glowing crystals. The air is warm, scented with earth, and a gentle breeze carries whispers of ancient lore. Its scales shimmer, blending with the cave’s radiant, serene beauty.

    ### Example 2:
    **Original Prompt:** "A futuristic city."
    **Expanded Variations:**
    1. At dusk, a futuristic city glows with holographic billboards, skyscrapers piercing the violet sky. Hovercars zip through neon-lit streets, and the air buzzes with drone hums. Markets sell cybernetic implants, and a synthetic breeze carries the scent of ozone.
    2. From a rooftop, a futuristic city sprawls under a starry dome, its glass towers reflecting artificial auroras. Maglev trains glide silently, and robotic vendors hawk glowing trinkets. The air is crisp, filled with the pulse of a digital heartbeat.
    3. In morning light, a futuristic city awakens, its eco-towers blooming with vertical gardens. Solar panels gleam, and citizens in exosuits stroll through parks. The air is fresh, scented with engineered flowers, and a calm hum underscores the city’s harmony.
    4. During a festival, a futuristic city pulses with music and light, its streets alive with dancers in luminescent suits. Floating platforms host performances, and the air vibrates with energy. Holographic fireworks paint the sky, blending technology and celebration.

    ### Now, expand the following prompt:

    **Original Prompt:** "{prompt}"

    **Expanded Variations:**
    1.
    2.
    3.
    4.
    """
    try:
        formatted_prompt = PromptTemplate(input_variables=["prompt", "variation_count"], template=few_shot_template)
        llm_expand = ChatMistralAI(model="mistral-large-latest", temperature=0.85, max_tokens=550, api_key=MISTRAL_API_KEY)
        chain = LLMChain(prompt=formatted_prompt, llm=llm_expand)
        
        response = chain.invoke({"prompt": prompt, "variation_count": variation_count})
        refined_text = response.get("text", "").strip()
        variations = re.findall(r'^\d+\.\s*(.+)$', refined_text, re.MULTILINE)
        
        if len(variations) < variation_count:
            logger.warning("Expected %d variations, got %d. Using placeholders.", variation_count, len(variations))
            variations += [f"Expanded version {i+1} of '{prompt}'" for i in range(len(variations), variation_count)]
        
        return variations[:variation_count]
    
    except Exception as e:
        logger.error("Expand prompt error: %s", str(e))
        return [f"Expanded version {i+1} of '{prompt}'" for i in range(variation_count)]

def steer_prompt(prompt, user_steer, variation_count=4):
    few_shot_template = """
    Modify the following prompt based on the user's steer input to shift its context or setting while preserving the core theme. 
    Generate exactly {variation_count} unique variations, each 120–140 words (600–700 tokens). 
    Ensure each variation is highly distinct in perspective (e.g., environment, time, or emotional tone) to maximize diversity. 
    Format as a numbered list (e.g., "1. Variation text"). Incorporate the steer precisely and avoid repetition.

    ### Example 1:
    **Original Prompt:** "A dragon in a cave."
    **User's Steering Input:** "Set in a jungle."
    **Steered Variations:**
    1. In a dense jungle, a dragon with jade scales lounges in a vine-draped cave at sunrise. Ferns sway in the humid air, and parrots screech overhead. A waterfall cascades nearby, its mist catching the dawn’s golden rays, while the dragon’s eyes gleam with primal curiosity.
    2. At midnight, a dragon with obsidian scales prowls a jungle cave, its roar muffled by thick foliage. Bioluminescent plants cast eerie glows, and the air is heavy with the scent of wet earth. Monkeys chatter nervously as the dragon’s tail disturbs ancient ruins hidden in the undergrowth.
    3. From a explorer’s perspective, a dragon with amber scales rests in a jungle cave, surrounded by moss-covered idols. The air buzzes with insects, and sunlight filters through a canopy, dappling its hide. The cave’s entrance is choked with vines, hiding its secrets from the world.
    4. During a storm, a dragon with silver scales shelters in a jungle cave, lightning illuminating its shimmering body. Rain pounds the canopy, and the air crackles with electricity. The cave’s walls are etched with tribal carvings, and the dragon’s presence feels like a guardian of the wild.

    ### Example 2:
    **Original Prompt:** "A futuristic city."
    **User's Steering Input:** "Make it a desert outpost."
    **Steered Variations:**
    1. In a sun-scorched desert, a futuristic outpost hums with life, its domed structures glinting at dawn. Solar sails flutter, powering drones that patrol sandy streets. The air is dry, scented with dust, and traders in heat-resistant suits barter under a crimson sky.
    2. At night, a desert outpost glows faintly, its modular towers rising from dunes. Holographic signs flicker, and the air carries the hum of wind turbines. Nomads in cybernetic cloaks gather around a central oasis, where bioluminescent algae light the water.
    3. From a pilot’s view, a futuristic desert outpost sprawls across a dune sea, its geodesic domes reflecting starlight. Sandstorms swirl nearby, and the air is filled with the whine of hovercraft. Underground tunnels connect vital systems, hidden from the harsh surface.
    4. During a sandstorm, a desert outpost stands resilient, its reinforced spires cutting through the haze. The air is thick with grit, and automated defenses hum softly. Residents in sealed habitats monitor the storm, their lives a testament to survival in this arid, high-tech frontier.

    ### Now, modify the following prompt using the given steer:

    **Original Prompt:** "{prompt}"
    **User's Steering Input:** "{user_steer}"

    **Steered Variations:**
    1.
    2.
    3.
    4.
    """
    try:
        formatted_prompt = PromptTemplate(input_variables=["prompt", "user_steer", "variation_count"], template=few_shot_template)
        llm_steer = ChatMistralAI(model="mistral-large-latest", temperature=0.85, max_tokens=650, api_key=MISTRAL_API_KEY)
        chain = LLMChain(prompt=formatted_prompt, llm=llm_steer)
        
        response = chain.invoke({"prompt": prompt, "user_steer": user_steer, "variation_count": variation_count})
        refined_text = response.get("text", "").strip()
        variations = re.findall(r'^\d+\.\s*(.+)$', refined_text, re.MULTILINE)
        
        if len(variations) < variation_count:
            logger.warning("Expected %d variations, got %d. Using placeholders.", variation_count, len(variations))
            variations += [f"Steered version {i+1} of '{prompt}'" for i in range(len(variations), variation_count)]
        
        return variations[:variation_count]
    
    except Exception as e:
        logger.error("Steer prompt error: %s", str(e))
        return [f"Steered version {i+1} of '{prompt}'" for i in range(variation_count)]

def style_prompt(prompt, user_style, variation_count=4):
    few_shot_template = """
    Enhance the artistic style of the following prompt while preserving its core theme. 
    Apply the requested style to create visually immersive, aesthetically aligned descriptions. 
    Generate exactly {variation_count} unique variations, each 120–140 words (600–700 tokens). 
    Ensure each variation is highly distinct in perspective (e.g., mood, lighting, or composition) to maximize diversity. 
    Format as a numbered list (e.g., "1. Variation text"). Avoid repetition.

    ### Example 1:
    **Original Prompt:** "A dragon in a cave."
    **User's Preferred Style:** "Baroque, dramatic."
    **Styled Variations:**
    1. In a baroque-inspired cave, a dragon with ruby scales basks in golden candlelight, its form draped in opulent shadows. Ornate chandeliers hang from stalactites, casting intricate patterns. The air is rich with the scent of wax, and the dragon’s wings shimmer like velvet in a dramatic, regal tableau.
    2. A dragon with gilded scales coils in a cave, lit by torchlight that mimics a baroque painting’s chiaroscuro. Carved marble columns frame its hoard of jeweled goblets, and the air vibrates with a mournful harpsichord echo. The scene is heavy with theatrical grandeur, each scale a brushstroke of divine artistry.
    3. From above, a dragon with sapphire scales lounges in a baroque cave, its form bathed in a divine glow from a crystal skylight. Velvet drapes of moss hang from the walls, and the air is thick with incense. The dragon’s presence is a masterpiece of opulence, evoking a cathedral’s solemn majesty.
    4. At twilight, a dragon with emerald scales rests in a baroque cave, where gilded frescoes depict ancient battles. Candlelight flickers, casting dramatic shadows that dance across its hide. The air hums with the weight of history, and the dragon’s eyes glow like sacred relics in this ornate, theatrical sanctuary.

    ### Example 2:
    **Original Prompt:** "A futuristic city."
    **User's Preferred Style:** "Cyberpunk, neon-drenched."
    **Styled Variations:**
    1. A cyberpunk city pulses at midnight, its skyscrapers drenched in neon pink and blue. Holographic ads flicker, and cybernetic citizens navigate rainy streets. The air crackles with static, and a gritty synth beat underscores the chaotic vibrancy of this neon-soaked urban sprawl.
    2. From a hacker’s den, a cyberpunk city glows with neon greens and purples, its towers shrouded in digital fog. Data streams light the sky, and the air smells of burnt circuits. Hoverbikes roar through alleys, and the city’s pulse is a relentless, electric dream of rebellion and tech.
    3. In pouring rain, a cyberpunk city shimmers with neon reflections on wet asphalt. Augmented reality overlays guide crowds through bustling markets, and the air hums with drone chatter. Towering megacorporations loom, their signs casting a surreal glow over this gritty, vibrant dystopia.
    4. At dawn, a cyberpunk city awakens, its neon reds and yellows fading against a smoggy sky. Cyber-enhanced street vendors hawk glowing wares, and the air carries the tang of metal. Holographic graffiti pulses on walls, blending art and anarchy in this vivid, high-tech underworld.

    ### Now, modify the following prompt using the given style:

    **Original Prompt:** "{prompt}"
    **User's Preferred Style:** "{user_style}"

    **Styled Variations:**
    1.
    2.
    3.
    4.
    """
    try:
        formatted_prompt = PromptTemplate(input_variables=["prompt", "user_style", "variation_count"], template=few_shot_template)
        llm_style = ChatMistralAI(model="mistral-large-latest", temperature=0.85, max_tokens=650, api_key=MISTRAL_API_KEY)
        chain = LLMChain(prompt=formatted_prompt, llm=llm_style)
        
        response = chain.invoke({"prompt": prompt, "user_style": user_style, "variation_count": variation_count})
        refined_text = response.get("text", "").strip()
        variations = re.findall(r'^\d+\.\s*(.+)$', refined_text, re.MULTILINE)
        
        if len(variations) < variation_count:
            logger.warning("Expected %d variations, got %d. Using placeholders.", variation_count, len(variations))
            variations += [f"Styled version {i+1} of '{prompt}'" for i in range(len(variations), variation_count)]
        
        return variations[:variation_count]
    
    except Exception as e:
        logger.error("Style prompt error: %s", str(e))
        return [f"Styled version {i+1} of '{prompt}'" for i in range(variation_count)]

def negative_prompt(prompt, negative_prompt, variation_count=4):
    few_shot_template = """
    Refine the following prompt by removing all elements specified in the negative constraints while preserving the core theme. 
    Generate exactly {variation_count} unique variations, each 120–140 words (600–700 tokens). 
    Ensure each variation is highly distinct in perspective (e.g., environment, time, or mood) to maximize diversity. 
    Format as a numbered list (e.g., "1. Variation text"). Reimagine the scene vividly without the negative elements.

    ### Example 1:
    **Original Prompt:** "A dragon in a cave with fire and treasure."
    **Negative Constraints:** "No fire, no treasure."
    **Refined Variations:**
    1. At sunrise, a dragon with azure scales rests in a cave, sunlight streaming through a skylight. Lush moss coats the walls, and a gentle stream sparkles nearby. The air is fresh, filled with the scent of wet stone, and the dragon’s calm presence blends with the serene, natural beauty.
    2. In moonlight, a dragon with silver scales lounges in a cave, its walls aglow with luminescent crystals. A soft breeze carries the scent of wildflowers, and the ground is carpeted with ferns. The dragon’s scales shimmer, creating a tranquil scene untouched by wealth or flames.
    3. From a traveler’s view, a dragon with green scales nests in a cave, surrounded by vines and glowing fungi. The air hums with the buzz of insects, and a clear pool reflects the dragon’s form. The scene is peaceful, a natural haven free of fiery or opulent distractions.
    4. During a quiet afternoon, a dragon with golden scales curls in a cave, where sunlight dapples through a canopy of leaves. The air is warm, scented with earth, and the walls are etched with ancient, non-magical carvings. The dragon’s presence is majestic, unmarred by riches or heat.

    ### Example 2:
    **Original Prompt:** "A futuristic city with neon lights and drones."
    **Negative Constraints:** "No neon lights, no drones."
    **Refined Variations:**
    1. At dawn, a futuristic city awakens, its eco-towers gleaming in soft sunlight. Pedestrian walkways bustle with citizens in smart fabrics, and the air is fresh with blooming rooftop gardens. Solar-powered trams glide quietly, blending technology with serene urban harmony.
    2. In twilight, a futuristic city glows with natural starlight, its glass spires reflecting a clear sky. Tree-lined avenues host vibrant markets, and the air carries the scent of engineered flowers. The city’s calm pulse is driven by sustainable systems, free of chaotic tech.
    3. From a balcony, a futuristic city sprawls, its minimalist towers bathed in morning mist. Electric bikes hum through green plazas, and the air is crisp with dew. Public art installations pulse softly, creating a tranquil, human-centric metropolis without garish distractions.
    4. During a festival, a futuristic city thrives with music and laughter, its streets lit by bioluminescent plants. Residents in flowing garments dance in open squares, and the air vibrates with acoustic melodies. The city’s beauty lies in its organic, tech-light celebration.

    ### Now, refine the following prompt by removing the negative elements:

    **Original Prompt:** "{prompt}"
    **Negative Constraints:** "{negative_prompt}"

    **Refined Variations:**
    1.
    2.
    3.
    4.
    """
    try:
        formatted_prompt = PromptTemplate(input_variables=["prompt", "negative_prompt", "variation_count"], template=few_shot_template)
        llm_negative = ChatMistralAI(model="mistral-large-latest", temperature=0.85, max_tokens=650, api_key=MISTRAL_API_KEY)
        chain = LLMChain(prompt=formatted_prompt, llm=llm_negative)
        
        response = chain.invoke({"prompt": prompt, "negative_prompt": negative_prompt, "variation_count": variation_count})
        refined_text = response.get("text", "").strip()
        variations = re.findall(r'^\d+\.\s*(.+)$', refined_text, re.MULTILINE)
        
        if len(variations) < variation_count:
            logger.warning("Expected %d variations, got %d. Using placeholders.", variation_count, len(variations))
            variations += [f"Refined version {i+1} of '{prompt}'" for i in range(len(variations), variation_count)]
        
        return variations[:variation_count]
    
    except Exception as e:
        logger.error("Negative prompt error: %s", str(e))
        return [f"Refined version {i+1} of '{prompt}'" for i in range(variation_count)]

# Image Generation Functions
def get_synonym(word):
    synonyms = wordnet.synsets(word)
    if synonyms and synonyms[0].lemmas():
        return random.choice(synonyms[0].lemmas()).name()
    return word

def synonym_replacement(prompt):
    words = prompt.split()
    modified_words = [get_synonym(word) if random.random() > 0.5 else word for word in words]
    return " ".join(modified_words)

def paraphrase_prompt(prompt):
    data = {"inputs": prompt}
    logger.info("Paraphrase request headers: %s", headers)
    try:
        response = requests.post(PARAPHRASING_API_URL, headers=headers, json=data, timeout=60)
        if response.status_code == 200 and isinstance(response.json(), list) and response.json():
            return response.json()[0].get("generated_text", prompt)
        elif response.status_code == 402:
            logger.error("Paraphrasing failed: Monthly credit limit exceeded (402). Check Hugging Face account credits.")
            return prompt
        else:
            logger.warning("Paraphrasing failed with status %d: %s", response.status_code, response.text)
            return prompt
    except requests.exceptions.RequestException as e:
        logger.error("Paraphrasing failed: %s", str(e))
        if isinstance(e, requests.exceptions.ConnectionError) and "NameResolutionError" in str(e):
            logger.error("DNS resolution failed for %s. Check network connectivity.", PARAPHRASING_API_URL)
        return prompt

@cache.memoize(timeout=3600)
def fetch_image(final_prompt, seed, index, max_retries=5, initial_delay=5, timeout=60):
    data = {"inputs": final_prompt, "parameters": {"seed": seed}}
    logger.info("Image generation request headers: %s", headers)
    for attempt in range(max_retries):
        try:
            response = requests.post(IMAGE_API_URL, headers=headers, json=data, timeout=timeout)
            if response.status_code == 200:
                img = PILImage.open(BytesIO(response.content))
                img_path = os.path.join(IMAGE_DIR, f"image_{index}_{seed}.jpg")
                img.save(img_path)
                logger.info("✅ Image %d saved successfully at %s", index, img_path)
                return img_path
            elif response.status_code == 402:
                logger.error("Image generation failed: Monthly credit limit exceeded (402). Check Hugging Face account credits.")
                return None
            elif response.status_code == 429:  # Rate limit
                delay = initial_delay * (2 ** attempt)
                logger.warning("⚠️ Image %d, Attempt %d: Rate limit (429), retrying in %.1fs", index, attempt+1, delay)
                time.sleep(delay)
            elif response.status_code == 503:  # Model loading
                delay = initial_delay * (2 ** attempt)
                logger.warning("⚠️ Image %d, Attempt %d: Model loading (503), retrying in %.1fs", index, attempt+1, delay)
                time.sleep(delay)
            else:
                logger.warning("⚠️ Image %d, Attempt %d: Failed with status %d, Response: %s", 
                                index, attempt+1, response.status_code, response.text)
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
        except requests.exceptions.RequestException as e:
            delay = initial_delay * (2 ** attempt)
            logger.warning("⚠️ Image %d, Attempt %d: Error - %s, retrying in %.1fs", index, attempt+1, str(e), delay)
            if isinstance(e, requests.exceptions.ConnectionError) and "NameResolutionError" in str(e):
                logger.error("DNS resolution failed for %s. Check network connectivity.", IMAGE_API_URL)
            time.sleep(delay)
    logger.error("❌ Failed to generate image %d after %d attempts. Possible network, DNS, or credit limit issue.", index, max_retries)
    return None

def generate_images_parallel(prompt, num_images):
    prompt_variations = generate_prompt_variations(prompt, num_images)
    image_paths = []
    failed_tasks = []
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(fetch_image, variation, random.randint(10000, 99999), i, max_retries=5, initial_delay=5, timeout=60)
            for i, variation in enumerate(prompt_variations)
        ]
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if result:
                    image_paths.append(result)
                else:
                    failed_tasks.append((prompt_variations[i], i))
            except Exception as e:
                logger.error("Error in image generation for index %d: %s", i, str(e))
                failed_tasks.append((prompt_variations[i], i))
    
    for variation, index in failed_tasks:
        logger.info("Retrying failed image %d", index)
        result = fetch_image(variation, random.randint(10000, 99999), index, max_retries=5, initial_delay=5, timeout=60)
        if result:
            image_paths.append(result)
        else:
            logger.error("Failed to generate image %d after retry", index)
    
    if len(image_paths) < num_images:
        logger.warning("Generated %d images out of %d requested", len(image_paths), num_images)
    
    return image_paths

def generate_prompt_variations(prompt, num_variations):
    variations = []
    structure_variations = [
        "{prompt} under a mysterious twilight sky.",
        "A cinematic rendering of {prompt}, viewed from a dynamic angle.",
        "An alternate reality version of {prompt}, where colors and physics are surreal.",
        "{prompt}, captured in an unexpected moment of action.",
        "A painterly vision of {prompt}, infused with impressionistic detail."
    ]
    
    for _ in range(num_variations):
        choice = random.choice(["synonym", "paraphrase", "both", "structure"])
        if choice == "synonym":
            new_prompt = synonym_replacement(prompt)
        elif choice == "paraphrase":
            new_prompt = paraphrase_prompt(prompt)
        elif choice == "both":
            new_prompt = synonym_replacement(paraphrase_prompt(prompt))
        else:
            structure = random.choice(structure_variations)
            new_prompt = structure.format(prompt=prompt)
        variations.append(new_prompt)
    return variations

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-to-use')
def how_to_use():
    return render_template('how_to_use.html')

@app.route('/history')
def history():
    if current_user.is_authenticated:
        prompts = PromptHistory.query.filter_by(user_id=current_user.id).order_by(PromptHistory.created_at.desc()).all()
    else:
        prompts = []
    return render_template('history.html', prompts=prompts)

@app.route('/save_prompt', methods=['POST'])
@login_required
def save_prompt():
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    new_prompt = PromptHistory(user_id=current_user.id, prompt=prompt)
    db.session.add(new_prompt)
    db.session.commit()
    return jsonify({'message': 'Prompt saved to history'})

@app.route('/delete_all_prompts', methods=['POST'])
@login_required
def delete_all_prompts():
    PromptHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'message': 'All prompts deleted'})

@app.route('/delete_selected_prompts', methods=['POST'])
@login_required
def delete_selected_prompts():
    data = request.get_json()
    prompt_ids = data.get('prompt_ids', [])
    if not prompt_ids:
        return jsonify({'error': 'No prompts selected'}), 400
    PromptHistory.query.filter(PromptHistory.id.in_(prompt_ids), PromptHistory.user_id == current_user.id).delete()
    db.session.commit()
    return jsonify({'message': f'Deleted {len(prompt_ids)} prompts'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
        elif User.query.filter_by(username=username).first():
            flash('Username already taken.', 'danger')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, email=email, password_hash=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/gallery')
def gallery():
    if current_user.is_authenticated:
        images = Image.query.filter_by(user_id=current_user.id).order_by(Image.created_at.desc()).all()
    else:
        images = []
    return render_template('gallery.html', images=images)

@app.route('/save_image', methods=['POST'])
@login_required
def save_image():
    data = request.get_json()
    file_path = data.get('file_path')
    prompt = data.get('prompt')
    if not file_path or not prompt:
        return jsonify({'error': 'File path and prompt required'}), 400
    
    filename = os.path.basename(file_path)
    standardized_path = f"static/Images/{filename}"
    full_path = os.path.join(IMAGE_DIR, filename)
    
    logger.info("Saving image with file_path: %s", standardized_path)
    if not os.path.exists(full_path):
        logger.error("Image file not found at: %s", full_path)
        return jsonify({'error': 'Image not found'}), 404
    
    new_image = Image(user_id=current_user.id, file_path=standardized_path, prompt=prompt)
    db.session.add(new_image)
    db.session.commit()
    logger.info("Image saved to database with file_path: %s", standardized_path)
    return jsonify({'message': 'Image saved to gallery'})

@app.route('/expand_prompt', methods=['POST'])
@login_required
def expand():
    data = request.get_json()
    user_prompt = data.get('prompt', '')
    if not user_prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    try:
        variations = expand_prompt(user_prompt)
        logger.info("Generated variations: %s", variations)
        return jsonify({'variations': variations})
    except Exception as e:
        logger.error("Expand prompt error: %s", str(e))
        return jsonify({'error': 'Failed to generate variations', 'variations': []}), 500

@app.route('/steer_prompt', methods=['POST'])
@login_required
def steer():
    data = request.get_json()
    user_prompt = data.get('prompt', '')
    user_steer = data.get('steer', '')
    if not user_prompt or not user_steer:
        return jsonify({'error': 'Prompt and steer input required'}), 400
    try:
        variations = steer_prompt(user_prompt, user_steer)
        logger.info("Generated variations: %s", variations)
        return jsonify({'variations': variations})
    except Exception as e:
        logger.error("Steer prompt error: %s", str(e))
        return jsonify({'error': 'Failed to generate variations', 'variations': []}), 500

@app.route('/style_prompt', methods=['POST'])
@login_required
def style():
    data = request.get_json()
    user_prompt = data.get('prompt', '')
    user_style = data.get('style', '')
    if not user_prompt or not user_style:
        return jsonify({'error': 'Prompt and style input required'}), 400
    try:
        variations = style_prompt(user_prompt, user_style)
        logger.info("Generated variations: %s", variations)
        return jsonify({'variations': variations})
    except Exception as e:
        logger.error("Style prompt error: %s", str(e))
        return jsonify({'error': 'Failed to generate variations', 'variations': []}), 500

@app.route('/negative_prompt', methods=['POST'])
@login_required
def negative():
    data = request.get_json()
    user_prompt = data.get('prompt', '')
    negative_prompt_input = data.get('negative', '')
    if not user_prompt or not negative_prompt_input:
        return jsonify({'error': 'Prompt and negative input required'}), 400
    try:
        variations = negative_prompt(user_prompt, negative_prompt_input)
        logger.info("Generated variations: %s", variations)
        return jsonify({'variations': variations})
    except Exception as e:
        logger.error("Negative prompt error: %s", str(e))
        return jsonify({'error': 'Failed to generate variations', 'variations': []}), 500

@app.route('/generate_images', methods=['POST'])
@login_required
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    num_images = data.get('num_images', 1)
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    try:
        num_images = int(num_images)
        if num_images < 1 or num_images > 10:
            return jsonify({'error': 'Number of images must be between 1 and 10'}), 400
        image_paths = generate_images_parallel(prompt, num_images)
        if not image_paths:
            logger.error("No images generated. Possible network, DNS, or credit limit issue.")
            return jsonify({
                'error': 'Failed to generate images. This may be due to a network issue or exhausted Hugging Face API credits. Please check your Hugging Face account or try again later.'
            }), 500
        image_urls = [f"/static/Images/{os.path.basename(path)}" for path in image_paths]
        response_data = {'image_urls': image_urls, 'prompt': prompt}
        if len(image_urls) < num_images:
            response_data['warning'] = f'Generated {len(image_urls)} out of {num_images} images due to API limitations or credit limits'
            return jsonify(response_data), 206
        return jsonify(response_data)
    except ValueError:
        return jsonify({'error': 'Invalid number of images provided'}), 400
    except Exception as e:
        logger.error("Image generation error: %s", str(e))
        return jsonify({
            'error': 'Failed to generate images. This may be due to a network issue or exhausted Hugging Face API credits. Please check your Hugging Face account or try again later.'
        }), 500

@app.route('/download_image/<filename>', methods=['POST'])
def download_image(filename):
    data = request.get_json()
    width = int(data.get('width', 512))
    height = int(data.get('height', 512))
    img_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(img_path):
        return jsonify({'error': 'Image not found'}), 404
    img = PILImage.open(img_path)
    img_resized = img.resize((width, height), PILImage.LANCZOS)
    output = BytesIO()
    img_resized.save(output, format="JPEG")
    output.seek(0)
    return send_file(output, mimetype='image/jpeg', as_attachment=True, download_name=f"{filename}")

@app.route('/delete_image/<filename>', methods=['POST'])
@login_required
def delete_image(filename):
    try:
        logger.info("Attempting to delete image with filename: %s", filename)
        image = Image.query.filter_by(user_id=current_user.id, file_path=f"static/Images/{filename}").first()
        if not image:
            image = Image.query.filter_by(user_id=current_user.id, file_path=f"/static/Images/{filename}").first()
        if not image:
            image = Image.query.filter_by(user_id=current_user.id, file_path=filename).first()
        
        if not image:
            logger.error("Image not found in database for user %s with filename %s", current_user.id, filename)
            return jsonify({'success': False, 'message': 'Image not found'}), 404
        
        logger.info("Found image in database with file_path: %s", image.file_path)
        image_path = os.path.join(app.root_path, 'static', 'Images', filename)
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.info("Image file deleted from filesystem: %s", image_path)
        else:
            logger.warning("Image file not found on server (possibly already deleted): %s", image_path)
        
        db.session.delete(image)
        db.session.commit()
        logger.info("Image record deleted from database for user %s", current_user.id)
        flash('Image deleted successfully!', 'success')
        return jsonify({'success': True, 'message': 'Image deleted successfully'})
    except Exception as e:
        db.session.rollback()
        logger.error("Delete image error: %s", str(e))
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/delete_all_images', methods=['POST'])
@login_required
def delete_all_images():
    try:
        images = Image.query.filter_by(user_id=current_user.id).all()
        if not images:
            logger.info("No images to delete for user %s", current_user.id)
            return jsonify({'success': True, 'message': 'No images to delete'})
        
        for image in images:
            image_path = os.path.join(app.root_path, image.file_path)
            if os.path.exists(image_path):
                os.remove(image_path)
                logger.info("Deleted image file: %s", image_path)
        
        Image.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        logger.info("All image records deleted from database for user %s", current_user.id)
        flash('All images deleted successfully!', 'success')
        return jsonify({'success': True, 'message': 'All images deleted successfully'})
    except Exception as e:
        db.session.rollback()
        logger.error("Delete all images error: %s", str(e))
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/clear_cache')
def clear_cache():
    cache.clear()
    logger.info("Cache cleared successfully")
    return jsonify({'message': 'Cache cleared successfully'})

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)