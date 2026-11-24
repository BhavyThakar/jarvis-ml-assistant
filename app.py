import numpy as np
import pickle
import threading
import pyttsx3
import os
import re
import time
import json
from pathlib import Path

from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import webbrowser
import dateutil.parser
import subprocess
import wikipedia

# ===== Extra imports for new features =====
import cv2
import base64
import requests

# Optional imports (handled safely)
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

try:
    import pvporcupine
    import pyaudio
except ImportError:
    pvporcupine = None
    pyaudio = None

# ================ FLASK APP INIT ============================
app = Flask(__name__)

# ========== LOAD CHATBOT MODEL & TOKENIZER ==========
def _build_chatbot_model(vocab_size: int, max_len: int, num_classes: int):
    # Must match train_chatbot.py architecture
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.003),
        metrics=["accuracy"],
    )
    return model


def load_chatbot_model():
    with open('tokenizer_inp.pkl', 'rb') as f:
        _tokenizer_inp = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        _label_encoder = pickle.load(f)

    # 1) Try direct load (best case)
    try:
        mdl = tf.keras.models.load_model(
            'chatbot_model.h5',
            compile=False,
            safe_mode=False,
        )
        return mdl, _tokenizer_inp, _label_encoder
    except TypeError:
        pass
    except Exception:
        pass

    try:
        mdl = tf.keras.models.load_model(
            'chatbot_model.h5',
            compile=False,
        )
        return mdl, _tokenizer_inp, _label_encoder
    except Exception:
        pass

    # 2) Fallback: reconstruct architecture and load weights
    vocab_size = len(_tokenizer_inp.word_index) + 1
    max_len = None
    try:
        if os.path.exists('training_data.txt'):
            _inputs = []
            with open('training_data.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    if "=" not in line:
                        continue
                    q, _a = line.split("=", 1)
                    q = q.strip().lower()
                    if q:
                        _inputs.append(q)
            seqs = _tokenizer_inp.texts_to_sequences(_inputs)
            max_len = max(len(s) for s in seqs) if seqs else 10
    except Exception:
        max_len = None

    if not max_len:
        max_len = 10

    num_classes = len(_label_encoder.classes_)
    mdl = _build_chatbot_model(vocab_size=vocab_size, max_len=max_len, num_classes=num_classes)

    try:
        mdl.load_weights('chatbot_model.h5')
    except Exception:
        try:
            mdl.load_weights('chatbot_model.h5', by_name=True, skip_mismatch=True)
        except Exception:
            pass

    return mdl, _tokenizer_inp, _label_encoder


model, tokenizer_inp, label_encoder = load_chatbot_model()
max_len_inp = int(model.input_shape[1])


# ========= LOAD EMOTION MODEL (FACE EMOTION) =========
try:
    emotion_model = load_model("emotion_model.h5")
    print("✅ Emotion model loaded.")
except Exception as e:
    print("❌ Could not load emotion model:", e)
    emotion_model = None

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Haar cascades for face & smile
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

current_face_emotion = "neutral"
current_smile = False

# ================ TTS (pyttsx3 - PURE ENGLISH VOICE) =======
speaking = False
def speak(text):
    def run():
        global speaking
        if speaking:
            return
        speaking = True
        try:
            # Strip emojis/non-ASCII symbols so TTS reads cleanly
            clean_text = re.sub(r'[^\x00-\x7F]+', '', str(text)).strip()
            if not clean_text:
                return

            engine = pyttsx3.init()
            voices = engine.getProperty('voices')

            # Prefer a male English voice (David/Mark/George) for Jarvis
            selected_voice = None
            for v in voices:
                name_lower = v.name.lower()
                if "english" in name_lower and any(k in name_lower for k in ("david", "mark", "george")):
                    selected_voice = v.id
                    break
            if not selected_voice:
                for v in voices:
                    if "english" in v.name.lower() or "en-us" in v.id.lower() or "en-gb" in v.id.lower():
                        selected_voice = v.id
                        break
            if selected_voice:
                engine.setProperty('voice', selected_voice)

            engine.setProperty('rate', 175)
            engine.setProperty('volume', 1.0)
            engine.say(clean_text)
            engine.runAndWait()
            engine.stop()
        finally:
            speaking = False
    threading.Thread(target=run, daemon=True).start()

# ================ BASIC PREDICTION =========================
def predict_response(user_text):
    seq = tokenizer_inp.texts_to_sequences([user_text.lower()])
    seq_padded = pad_sequences(seq, maxlen=max_len_inp, padding='post')
    pred = model.predict(seq_padded)
    pred_index = np.argmax(pred)
    response = label_encoder.inverse_transform([pred_index])[0]
    return response

# ================ MEMORY SYSTEM (FEATURE 1) ================
MEMORY_FILE = "jarvis_memory.json"
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)
else:
    memory = {}

def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def handle_memory(user_text: str):
    """
    Persistent JSON memory system:
    - 'remember that ...'
    - 'remember my X is Y'
    - 'what is my X'
    - 'what did you remember'
    """
    text = user_text.lower()

    # remember that ...
    if "remember that" in text:
        fact = user_text.lower().split("remember that", 1)[1].strip()
        if not fact:
            return "What would you like me to remember?"
        memory.setdefault("notes", []).append(fact)
        save_memory()
        return "Got it. I have saved that to memory."

    # remember my X is Y
    if "remember my" in text and " is " in text:
        try:
            after = text.split("remember my", 1)[1].strip()
            key, val = after.split(" is ", 1)
            key = key.strip()
            val = val.strip()
            memory.setdefault("profile", {})[key] = val
            save_memory()
            return f"Understood. I will remember that your {key} is {val}."
        except Exception:
            return "I didn't catch that properly. Try saying 'remember my birthday is May 5'."

    # what is my X
    if text.startswith("what is my"):
        key = text.replace("what is my", "").strip()
        if "profile" in memory and key in memory["profile"]:
            return f"Your {key} is {memory['profile'][key]}."
        else:
            return f"I don't have your {key} saved in memory yet."

    # what did you remember
    if "what did you remember" in text or "what do you remember" in text:
        notes = memory.get("notes", [])
        prof = memory.get("profile", {})
        if not notes and not prof:
            return "I don't have anything saved in memory yet."
        msg_parts = []
        if prof:
            for k, v in prof.items():
                msg_parts.append(f"Your {k} is {v}")
        if notes:
            msg_parts.append("I remember these notes: " + "; ".join(notes))
        return ". ".join(msg_parts) + "."

    return None

# ========== ADVANCED CONVERSATION MEMORY (last 10 msgs) ==========
conversation_memory = []

def add_to_conversation(user, bot):
    conversation_memory.append({
        "user": user,
        "bot": bot,
        "time": time.time()
    })
    if len(conversation_memory) > 10:
        conversation_memory.pop(0)

def get_memory_context():
    context = ""
    for item in conversation_memory:
        context += f"User: {item['user']}\nJarvis: {item['bot']}\n"
    return context

# ================ EMOTION-BASED REPLY (TEXT) =================
def detect_mood_reply(user_text: str):
    text = user_text.lower()
    sad_words = ["sad", "down", "depressed", "upset", "lonely", "broken", "hurt"]
    happy_words = ["happy", "excited", "awesome", "great", "amazing"]
    stress_words = ["stressed", "anxious", "tired", "exhausted"]

    if any(w in text for w in sad_words):
        return "I'm sorry you're feeling down. I'm right here if you want to talk about what's going on."
    if any(w in text for w in happy_words):
        return "Love that energy! Keep that momentum going. What would you like to build next?"
    if any(w in text for w in stress_words):
        return "You sound a bit stressed. Take a short break and grab some water. How can I help lighten your workload?"

    return None

# Tone modifier based on face emotion + smile
def tone_based_reply(bot_reply, detected_emotion, smile_flag):
    if smile_flag:
        return bot_reply + " Great to see you smiling! Love that energy."

    if detected_emotion == "happy":
        return bot_reply + " You're looking happy today!"
    elif detected_emotion == "sad":
        return bot_reply + " You seem a bit down. Let me know how I can help."
    elif detected_emotion == "angry":
        return bot_reply + " Take a deep breath, we'll sort this out."
    elif detected_emotion == "fear":
        return bot_reply + " Don't worry, everything is under control."
    elif detected_emotion == "surprise":
        return bot_reply + " You look surprised!"
    else:
        return bot_reply

# ================ SYSTEM COMMANDS (FEATURE 2) ================
def handle_system_commands(user_text: str):
    text = user_text.lower()

    if "shutdown" in text or "shut down" in text:
        os.system("shutdown /s /t 30")
        return "System shutdown scheduled in 30 seconds. Run 'shutdown /a' in Command Prompt to cancel."

    if "restart" in text or "reboot" in text:
        os.system("shutdown /r /t 30")
        return "System restart scheduled in 30 seconds. Run 'shutdown /a' in Command Prompt to cancel."

    if "lock screen" in text:
        os.system("rundll32.exe user32.dll,LockWorkStation")
        return "Locking your screen now."

    return None

# ================ FILE SEARCH & OPEN (FEATURE 6) ===========
def search_file(filename: str, base_dir: Path):
    filename = filename.lower()
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if filename in f.lower():
                return os.path.join(root, f)
    return None

def handle_file_commands(user_text: str):
    text = user_text.lower()
    home = Path.home()

    if text.startswith("find file"):
        name = text.replace("find file", "").strip()
        if not name:
            return "Which file should I search for?"
        path = search_file(name, home)
        if path:
            return f"I found a file matching '{name}' at: {path}"
        else:
            return f"I couldn't find any file named '{name}'."

    if "open downloads folder" in text:
        downloads = home / "Downloads"
        if downloads.exists():
            os.startfile(str(downloads))
            return "Opening your Downloads folder."
        else:
            return "I couldn't locate your Downloads folder."

    if text.startswith("open folder"):
        name = text.replace("open folder", "").strip()
        if not name:
            return "Which folder should I open?"
        guessed = home / name
        if guessed.exists():
            os.startfile(str(guessed))
            return f"Opening folder {guessed}."
        else:
            return f"I couldn't find a folder named {name}."

    return None

# ================ PDF READING (FEATURE 8 & 5) ==================
def read_pdf(path: str):
    if PyPDF2 is None:
        return "The PyPDF2 library is not installed."
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages_to_read = min(2, len(reader.pages))
            text_out = ""
            for i in range(pages_to_read):
                text_out += reader.pages[i].extract_text() + "\n"
        return text_out.strip()
    except Exception as e:
        return f"Sorry, I couldn't read that PDF: {e}"

def handle_pdf_reading(user_text: str):
    text = user_text.lower()
    home = Path.home()

    if "read pdf" in text:
        name = text.split("read pdf", 1)[1].replace("named", "").strip()
        if not name:
            return "Which PDF should I read?"

        if not name.lower().endswith(".pdf"):
            name = name + ".pdf"

        path = search_file(name, home)
        if not path:
            return f"I couldn't find any PDF named {name}."

        pdf_text = read_pdf(path)
        speak("Reading from your PDF now. " + pdf_text[:1500])
        return f"Reading the first section of {os.path.basename(path)}."

    return None

# ================ KNOWLEDGE / GPT-LIKE MODE (FEATURE 15) ===
def handle_knowledge_queries(user_text: str):
    text = user_text.lower()
    triggers = ["who is", "what is", "tell me about", "define", "explain"]

    if any(text.startswith(t) for t in triggers) or any(t in text for t in triggers):
        try:
            wikipedia.set_lang("en")
            for t in triggers:
                text = text.replace(t, "")
            query = text.strip()
            if not query:
                return None
            summary = wikipedia.summary(query, sentences=3)
            return summary
        except Exception:
            return None

    return None

# ============= NEW KNOWLEDGE TOOLS: WEATHER, STOCK, GITHUB, ERROR =============
def get_weather(city: str):
    try:
        url = f"https://wttr.in/{city}?format=j1"
        data = requests.get(url, timeout=5).json()
        temp = data['current_condition'][0]['temp_C']
        desc = data['current_condition'][0]['weatherDesc'][0]['value']
        return f"The weather in {city} is {desc} with a temperature of {temp} degrees Celsius."
    except Exception:
        return "Sorry, I couldn't fetch the weather data right now."

def get_stock_price(symbol: str):
    if yf is None:
        return "The yfinance library is not installed."
    try:
        data = yf.Ticker(symbol)
        hist = data.history(period="1d")
        if hist.empty:
            return "I couldn't find stock data for that symbol. Please check the ticker."
        price = hist["Close"].iloc[0]
        return f"{symbol.upper()} current price is {round(float(price), 2)}."
    except Exception:
        return "I encountered an error while fetching that stock price."

def github_search(query: str):
    try:
        url = f"https://api.github.com/search/repositories?q={query}"
        data = requests.get(url, timeout=5).json()
        items = data.get("items", [])
        if not items:
            return "I couldn't find any matching repositories on GitHub."
        repo = items[0]
        return (
            f"Top GitHub repository for '{query}': {repo['name']} "
            f"with {repo['stargazers_count']} stars. URL: {repo['html_url']}"
        )
    except Exception:
        return "I ran into an error while searching GitHub."

def get_cricket_score_placeholder():
    return "Live cricket scores require an active sports API key."

def fix_code_error(err_msg: str):
    if "SyntaxError" in err_msg:
        return "This is a SyntaxError. Check for missing brackets, colons, quotes, or incorrect indentation."
    if "NameError" in err_msg:
        return "This is a NameError. A variable or function name is being referenced before it is defined."
    if "TypeError" in err_msg:
        return "This is a TypeError. An operation or function was given an argument of the wrong data type."
    if "IndexError" in err_msg:
        return "This is an IndexError. You are trying to access an index outside the range of a list or string."
    if "KeyError" in err_msg:
        return "This is a KeyError. That key does not exist in the dictionary."
    return (
        f"Here is the error reported: {err_msg}. "
        "Check the line number, variable names, data types, and indentation."
    )

# ================ EXISTING SPECIAL TASKS + NEW ONES =========
def handle_special_tasks(user_text):
    # 1) MEMORY
    mem_resp = handle_memory(user_text)
    if mem_resp:
        return mem_resp

    # 2) SYSTEM COMMANDS
    sys_resp = handle_system_commands(user_text)
    if sys_resp:
        return sys_resp

    # 3) FILE COMMANDS
    file_resp = handle_file_commands(user_text)
    if file_resp:
        return file_resp

    # 4) PDF READING
    pdf_resp = handle_pdf_reading(user_text)
    if pdf_resp:
        return pdf_resp

    # 5) NEW KNOWLEDGE FEATURES
    lower = user_text.lower()

    if lower.startswith("weather in"):
        city = user_text.split("in", 1)[1].strip()
        return get_weather(city)

    if "cricket score" in lower:
        return get_cricket_score_placeholder()

    if lower.startswith("stock "):
        symbol = user_text.split("stock", 1)[1].strip()
        return get_stock_price(symbol)

    if lower.startswith("github search"):
        query = user_text.split("github search", 1)[1].strip()
        return github_search(query)

    if lower.startswith("fix error"):
        err = user_text.split("fix error", 1)[1].strip()
        return fix_code_error(err)

    # 6) DESKTOP & WEB COMMANDS
    user_text = lower

    if "current time" in user_text or "what time is it" in user_text:
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."

    elif "open whatsapp" in user_text:
        webbrowser.open("https://web.whatsapp.com/")
        return "Opening WhatsApp for you."

    elif "open youtube" in user_text:
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube for you."

    elif "play" in user_text and "youtube" in user_text:
        song = user_text.split("play")[-1].replace("on youtube", "").strip()
        url = f"https://www.youtube.com/results?search_query={song.replace(' ', '+')}"
        webbrowser.open(url)
        return f"Playing {song} on YouTube."

    elif "open google" in user_text:
        webbrowser.open("https://www.google.com")
        return "Opening Google for you."

    elif user_text.startswith("search") and "on google" in user_text:
        query = user_text.replace("search", "").replace("on google", "").strip()
        if query:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(url)
            return f"Searching Google for '{query}'."
        else:
            return "Please specify what you want me to search on Google."

    elif "open vs code" in user_text or "open visual studio code" in user_text:
        try:
            os.startfile("code")
            return "Opening Visual Studio Code."
        except Exception as e:
            return f"Sorry, I couldn't open VS Code: {e}"

    elif "open spotify" in user_text:
        try:
            os.startfile("spotify")
            return "Opening Spotify for you."
        except Exception as e:
            return f"Sorry, I couldn't open Spotify: {e}"

    elif "open calculator" in user_text:
        try:
            os.system("calc")
            return "Opening Calculator."
        except Exception as e:
            return f"Sorry, I couldn't open Calculator: {e}"

    elif "open notepad" in user_text:
        try:
            os.system("notepad")
            return "Opening Notepad."
        except Exception as e:
            return f"Sorry, I couldn't open Notepad: {e}"

    elif "write a note" in user_text or "take a note" in user_text:
        return "write_note_mode"

    return None

# ================ REMINDER SYSTEM (EXISTING) ================
reminders = []
reminders_lock = threading.Lock()

def parse_reminder(text):
    try:
        lower_text = text.lower()
        if "remind me to" in lower_text and "at" in lower_text:
            task_part = lower_text.split("remind me to", 1)[1]
            task, time_part = task_part.split("at", 1)
            task = task.strip()
            time_str = time_part.strip()
            reminder_time = dateutil.parser.parse(time_str, default=datetime.now())
            if reminder_time < datetime.now():
                reminder_time += timedelta(days=1)
            return task, reminder_time
        else:
            return None, None
    except Exception as e:
        print(f"Error parsing reminder: {e}")
        return None, None

def reminder_checker():
    while True:
        now = datetime.now()
        with reminders_lock:
            due_reminders = [r for r in reminders if r['time'] <= now]
            for r in due_reminders:
                print(f"Reminder triggered: {r['task']} at {r['time'].strftime('%H:%M')}")
                speak(f"Reminder: {r['task']}")
                reminders.remove(r)
        time.sleep(5)

threading.Thread(target=reminder_checker, daemon=True).start()

# ================ NOTE WRITING SUPPORT ======================
is_note_mode = False

def open_notepad_async(path):
    subprocess.run(["notepad.exe", path])

# ================ FILE UPLOAD + OCR (FEATURE 5) =============
def explain_file(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        txt = read_pdf(path)
        return "From this PDF, here is what I extracted:\n\n" + txt[:1500]

    if pytesseract is not None and Image is not None and ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        try:
            text = pytesseract.image_to_string(Image.open(path))
            if not text.strip():
                return "I couldn't detect any readable text in that image."
            return "From this image, I read:\n\n" + text[:1500]
        except Exception as e:
            return f"I ran into an issue reading that image: {e}"

    return "I don't support reading that file type yet."

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"reply": "No file was received."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"reply": "The uploaded file has an empty filename."}), 400

    save_path = os.path.join(os.getcwd(), "uploaded_" + file.filename)
    file.save(save_path)

    reply = explain_file(save_path)
    speak(reply)
    return jsonify({"reply": reply})

# ================ HOME ROUTE ================================
@app.route("/")
def home():
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()

    if weekday == 5 or weekday == 6:
        if 5 <= hour < 12:
            greeting = "Good morning Bhavy, happy weekend! How are you today?"
        elif 12 <= hour < 18:
            greeting = "Good afternoon Bhavy, happy weekend! How are you today?"
        else:
            greeting = "Good evening Bhavy, happy weekend! How are you today?"
    else:
        if 5 <= hour < 12:
            greeting = "Good morning Bhavy, how are you today?"
        elif 12 <= hour < 18:
            greeting = "Good afternoon Bhavy, how are you today?"
        else:
            greeting = "Good evening Bhavy, how are you today?"

    speak(greeting)
    return render_template("index.html")

# ================ CHAT ROUTE ================================
@app.route("/chat", methods=["POST"])
def chat():
    global is_note_mode, current_face_emotion, current_smile
    user_message = request.json.get("message", "")

    # NOTE MODE
    if is_note_mode:
        note = user_message
        note_path = os.path.abspath("note.txt")
        with open(note_path, "w", encoding="utf-8") as file:
            file.write(note)
        threading.Thread(target=open_notepad_async, args=(note_path,)).start()

        reply = "Your note has been written and opened in Notepad."
        speak(reply)
        is_note_mode = False
        add_to_conversation(user_message, reply)
        now = datetime.now()
        return jsonify({
            "reply": reply,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S")
        })

    # SPECIAL TASKS
    special_response = handle_special_tasks(user_message)
    if special_response:
        if special_response == "write_note_mode":
            reply = "Please tell me what to write."
            speak("What should I write?")
            is_note_mode = True
        else:
            reply = special_response
            speak(reply)
        add_to_conversation(user_message, reply)
        now = datetime.now()
        return jsonify({
            "reply": reply,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S")
        })

    # REMINDERS
    task, reminder_time = parse_reminder(user_message)
    if task and reminder_time:
        with reminders_lock:
            reminders.append({"task": task, "time": reminder_time})
        reply = f"Reminder set for {reminder_time.strftime('%I:%M %p')}: {task}"
        speak(reply)
        add_to_conversation(user_message, reply)
        now = datetime.now()
        return jsonify({
            "reply": reply,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S")
        })

    # EMOTION CHECK (FROM TEXT)
    mood_reply = detect_mood_reply(user_message)
    if mood_reply:
        reply = mood_reply
        speak(reply)
        add_to_conversation(user_message, reply)
        now = datetime.now()
        return jsonify({
            "reply": reply,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S")
        })

    # KNOWLEDGE VIA WIKIPEDIA
    knowledge_reply = handle_knowledge_queries(user_message)
    if knowledge_reply:
        reply = knowledge_reply
        speak(reply)
        add_to_conversation(user_message, reply)
        now = datetime.now()
        return jsonify({
            "reply": reply,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S")
        })

    # FALLBACK MODEL
    reply = predict_response(user_message)
    reply = tone_based_reply(reply, current_face_emotion, current_smile)
    speak(reply)
    add_to_conversation(user_message, reply)

    now = datetime.now()
    return jsonify({
        "reply": reply,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S")
    })

# ================ CAMERA EMOTION + SMILE ROUTE ==============
def decode_base64_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

@app.route("/analyze_emotion", methods=["POST"])
def analyze_emotion():
    global current_face_emotion, current_smile

    data = request.get_json() or {}
    frame = data.get("image")

    if not frame:
        return jsonify({"success": False, "emotion": "neutral", "smile": False})

    img = decode_base64_image(frame)
    if img is None:
        return jsonify({"success": False, "emotion": "neutral", "smile": False})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotion = "neutral"
    smile_detected = False

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # Smile detection
        smiles = smile_cascade.detectMultiScale(face_roi, 1.7, 22)
        smile_detected = len(smiles) > 0

        # Emotion detection
        if emotion_model is not None:
            face_resized = cv2.resize(face_roi, (48, 48))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)
            preds = emotion_model.predict(face_resized)[0]
            idx = int(np.argmax(preds))
            if 0 <= idx < len(EMOTIONS):
                emotion = EMOTIONS[idx]
            else:
                emotion = "neutral"
        else:
            emotion = "neutral"

    current_face_emotion = emotion
    current_smile = smile_detected

    return jsonify({
        "success": True,
        "emotion": emotion,
        "smile": smile_detected
    })

# ================ HOTWORD LISTENER (OPTIONAL) ===============
def hotword_listener():
    if pvporcupine is None or pyaudio is None:
        print("Hotword libs not installed, skipping hotword listener.")
        return
    try:
        porcupine = pvporcupine.create(keywords=["jarvis"])
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        print("🎧 Hotword listener started. Say 'Jarvis'...")
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = np.frombuffer(pcm, dtype=np.int16)
            result = porcupine.process(pcm)
            if result >= 0:
                speak("Yes Bhavy, I'm listening.")
    except Exception as e:
        print("Error in hotword listener:", e)

# ================ RUN APP ================================
if __name__ == "__main__":
    threading.Thread(target=hotword_listener, daemon=True).start()
    app.run(debug=True)