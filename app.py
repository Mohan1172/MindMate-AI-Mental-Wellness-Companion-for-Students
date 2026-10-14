# MindMate first-run launcher
# If dependencies are missing, automatically create a project-local virtual
# environment, install requirements there, and restart this app with that
# environment. This avoids modifying uv/system-managed Python installations.
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REQUIREMENTS = BASE_DIR / "requirements.txt"
VENV_DIR = BASE_DIR / ".venv"


def _module_missing(module_name):
    return importlib.util.find_spec(module_name) is None


def _venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _dependencies_missing():
    required = ("flask", "numpy", "sklearn")
    return [name for name in required if _module_missing(name)]


def _run(command, label):
    print(f"\nMindMate setup: {label}")
    subprocess.check_call(command, cwd=BASE_DIR)


def _ensure_project_environment():
    missing = _dependencies_missing()
    if not missing:
        return

    # If we are already running from the project venv, install into that venv.
    vpy = _venv_python()
    running_from_project_venv = Path(sys.executable).resolve() == vpy.resolve() if vpy.exists() else False

    if not running_from_project_venv:
        print("\nMindMate first-run setup: missing packages detected:")
        print("  " + ", ".join(missing))
        print("Creating a private .venv inside the MindMate project...")

        if not vpy.exists():
            try:
                _run([sys.executable, "-m", "venv", str(VENV_DIR)], "Creating project virtual environment")
            except subprocess.CalledProcessError as exc:
                print("\nCould not create the project virtual environment.")
                print("Please install Python with the standard venv component, then run app.py again.")
                raise SystemExit(exc.returncode)

        if not vpy.exists():
            raise SystemExit("MindMate could not find the new virtual-environment Python executable.")

        # Restart the same app.py using the project-local interpreter.
        env = os.environ.copy()
        env["MINDMATE_VENV_BOOTSTRAPPED"] = "1"
        print(f"Installing dependencies with: {vpy}")
        try:
            _run([str(vpy), "-m", "pip", "install", "--upgrade", "pip"], "Updating pip")
            _run([str(vpy), "-m", "pip", "install", "-r", str(REQUIREMENTS)], "Installing MindMate dependencies")
        except subprocess.CalledProcessError as exc:
            print("\nMindMate dependency installation failed.")
            print("You can run this manually:")
            print(f'  "{vpy}" -m pip install -r "{REQUIREMENTS}"')
            raise SystemExit(exc.returncode)

        os.execve(str(vpy), [str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]], env)

    # We are inside .venv. If packages are still missing, install normally.
    missing = _dependencies_missing()
    if missing:
        print("\nMindMate: installing missing packages inside the project .venv...")
        try:
            _run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)], "Installing dependencies")
        except subprocess.CalledProcessError as exc:
            print("\nInstallation failed. Run:")
            print(f'  "{sys.executable}" -m pip install -r "{REQUIREMENTS}"')
            raise SystemExit(exc.returncode)


_ensure_project_environment()


from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import sqlite3
from datetime import datetime, timedelta
import threading
import time
import webbrowser

from ml_engine import WellnessEngine

MODEL_DIR = BASE_DIR / "models"
DATASET = BASE_DIR / "data" / "emotion_dataset.csv"
DB_PATH = BASE_DIR / "database" / "wellness.db"

app = Flask(__name__)
app.secret_key = os.environ.get("MINDMATE_SECRET_KEY", "mindmate-local-development-key")
engine = WellnessEngine(DATASET, MODEL_DIR)

RESPONSES = {
    "food_nutrition": (
        "For stress support, try regular balanced meals rather than skipping food. "
        "Good options include oats, bananas, yogurt/curd, nuts and seeds, leafy greens, "
        "beans, berries or other fruit, and protein-rich meals. Water is also important. "
        "For studying, fruit + nuts or yogurt + fruit can be a simple snack. "
        "Food can support wellbeing, but it is not a treatment for anxiety or stress."
    ),
    "sleep": (
        "If stress is affecting sleep, try a consistent bedtime, reduce caffeine late in the day, "
        "dim screens before bed, and do a short breathing or relaxation exercise. If sleep problems "
        "keep happening or seriously affect your day, consider speaking with a healthcare professional."
    ),
    "study_pressure": (
        "For exam or college stress, make the next step very small: choose one topic, study for 20–25 minutes, "
        "take a short break, then continue. Write the top 3 tasks for today instead of trying to finish everything at once. "
        "If you tell me the subject or exam problem, I can help make a simple plan."
    ),
    "breathing": (
        "Try this for about 2 minutes: inhale gently through your nose for 4 seconds, then exhale slowly for 6 seconds. "
        "Repeat without forcing the breath. If counting makes you uncomfortable, simply make the exhale a little longer than the inhale."
    ),
    "grounding": (
        "Try the 5-4-3-2-1 grounding exercise: notice 5 things you can see, 4 you can touch, 3 you can hear, "
        "2 you can smell, and 1 you can taste. Move slowly and focus on the present moment."
    ),
    "exercise": (
        "Gentle movement can be useful for stress. A 10–20 minute walk, stretching, or a light workout is a reasonable place to start. "
        "Choose something comfortable rather than treating exercise as another task you must perfect."
    ),
    "focus": (
        "For concentration, remove one distraction, pick one small task, and use a 25-minute focus block followed by a 5-minute break. "
        "Keep your phone away if possible. If you tell me what you are trying to study, I can help break it into steps."
    ),
    "motivation": (
        "When motivation is low, do not wait to feel fully motivated. Start with a 5-minute version of the task. "
        "Once you begin, decide whether to continue for another 10–20 minutes. Small progress still counts."
    ),
    "loneliness": (
        "Feeling lonely can be heavy. Consider one low-pressure connection today: message a friend, classmate, family member, "
        "or someone you trust. You do not need to explain everything; even a simple conversation can help."
    ),
    "anger": (
        "Before responding while angry, pause for a few minutes and create some physical distance from the situation if you can. "
        "Slow breathing, a short walk, or writing down what triggered you can help you respond rather than react."
    ),
    "self_care": (
        "For everyday stress, try one small reset: slow breathing, a short walk, stretching, music, a shower, talking to someone you trust, "
        "or taking a short break from the task. If you tell me what is causing the stress, I can make the suggestion more specific."
    ),
    "definition": (
        "Stress is the body's and mind's response to demands or pressure. A certain amount can be useful, but ongoing or intense stress can affect "
        "sleep, concentration, mood, and energy. If you tell me what you are experiencing, I can explain it in a more relevant way."
    ),
    "general_wellbeing": (
        "I’m here with you. I can help with stress, anxiety, sleep, study pressure, food and daily wellbeing. "
        "Ask a specific question and I’ll try to answer that question directly."
    ),
}


def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS checkins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        mood INTEGER NOT NULL,
        stress INTEGER NOT NULL,
        energy INTEGER NOT NULL,
        sleep REAL NOT NULL,
        note TEXT,
        emotion TEXT,
        risk_level TEXT,
        wellness_score INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        user_message TEXT NOT NULL,
        bot_message TEXT NOT NULL,
        emotion TEXT,
        risk_level TEXT,
        intent TEXT,
        confidence REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(chats)").fetchall()}
    if "intent" not in columns:
        conn.execute("ALTER TABLE chats ADD COLUMN intent TEXT")
    if "confidence" not in columns:
        conn.execute("ALTER TABLE chats ADD COLUMN confidence REAL")
    conn.commit()
    conn.close()


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return user


def get_recent_checkins(user_id, days=14):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = db()
    rows = conn.execute(
        "SELECT * FROM checkins WHERE user_id=? AND created_at>=? ORDER BY created_at DESC",
        (user_id, cutoff),
    ).fetchall()
    conn.close()
    return rows


def calculate_wellness(mood, stress, energy, sleep):
    score = ((mood / 10) * 30 + ((10 - stress) / 10) * 30 + (energy / 10) * 20 + min(sleep / 8, 1) * 20)
    return max(0, min(100, round(score)))


def risk_from_checkins(rows):
    if not rows:
        return "Low"
    recent = rows[:7]
    avg_stress = sum(r["stress"] for r in recent) / len(recent)
    avg_energy = sum(r["energy"] for r in recent) / len(recent)
    avg_sleep = sum(r["sleep"] for r in recent) / len(recent)
    avg_mood = sum(r["mood"] for r in recent) / len(recent)
    if avg_stress >= 8 or (avg_stress >= 7 and avg_sleep < 5):
        return "High"
    if avg_stress >= 6 or avg_energy <= 4 or avg_mood <= 4:
        return "Moderate"
    return "Low"


def make_response(message, analysis, history):
    if analysis["crisis"]:
        return (
            "I'm really sorry you're going through this. If you might hurt yourself or someone else, "
            "please move to a safer place and contact a trusted person who can stay with you. "
            "In India, Tele-MANAS can be reached at 14416 or 1800-89-14416, and emergency services are 112. "
            "This app is not a crisis service."
        )

    intent = analysis["intent"]
    reply = RESPONSES.get(intent, RESPONSES["general_wellbeing"])

    if intent == "general_wellbeing" and history:
        recent = history[:3]
        avg_stress = sum(r["stress"] for r in recent) / len(recent)
        if avg_stress >= 7:
            reply += " Your recent check-ins also show higher stress, so keeping the next step small may be especially useful."
    return reply


@app.route("/")
def index():
    return render_template("index.html", user=current_user())


@app.route("/start", methods=["POST"])
def start():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("index"))
    conn = db()
    cur = conn.execute("INSERT INTO users(name, created_at) VALUES (?, ?)", (name, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    session["user_id"] = cur.lastrowid
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/checkin", methods=["GET", "POST"])
def checkin():
    user = current_user()
    if not user:
        return redirect(url_for("index"))
    if request.method == "POST":
        mood = int(request.form["mood"])
        stress = int(request.form["stress"])
        energy = int(request.form["energy"])
        sleep = float(request.form["sleep"])
        note = request.form.get("note", "").strip()
        analysis = engine.analyze(note) if note else {"emotion": "Neutral", "risk": "Low", "crisis": False}
        score = calculate_wellness(mood, stress, energy, sleep)
        conn = db()
        conn.execute(
            "INSERT INTO checkins(user_id,mood,stress,energy,sleep,note,emotion,risk_level,wellness_score,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (user["id"], mood, stress, energy, sleep, note, analysis["emotion"], analysis["risk"], score, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        return redirect(url_for("dashboard"))
    return render_template("checkin.html", user=user)


@app.route("/chat")
def chat():
    user = current_user()
    if not user:
        return redirect(url_for("index"))
    conn = db()
    messages = conn.execute("SELECT * FROM chats WHERE user_id=? ORDER BY created_at ASC", (user["id"],)).fetchall()
    conn.close()
    return render_template("chat.html", user=user, messages=messages)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    user = current_user()
    if not user:
        return jsonify({"error": "Please start a session first."}), 401
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400
    history = get_recent_checkins(user["id"])
    analysis = engine.analyze(message, history)
    reply = make_response(message, analysis, history)

    conn = db()
    conn.execute(
        "INSERT INTO chats(user_id,user_message,bot_message,emotion,risk_level,intent,confidence,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (user["id"], message, reply, analysis["emotion"], analysis["risk"], analysis["intent"], analysis["intent_confidence"], datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return jsonify({
        "reply": reply,
        "emotion": analysis["emotion"],
        "risk": analysis["risk"],
        "intent": analysis["intent"],
        "confidence": analysis["intent_confidence"],
        "crisis": analysis["crisis"],
    })


@app.route("/dashboard")
def dashboard():
    user = current_user()
    if not user:
        return redirect(url_for("index"))
    rows = get_recent_checkins(user["id"], 30)
    chart_rows = list(reversed(rows))
    overall_risk = risk_from_checkins(rows)
    latest = rows[0] if rows else None
    avg_stress = round(sum(r["stress"] for r in rows) / len(rows), 1) if rows else 0
    avg_mood = round(sum(r["mood"] for r in rows) / len(rows), 1) if rows else 0
    avg_sleep = round(sum(r["sleep"] for r in rows) / len(rows), 1) if rows else 0
    avg_energy = round(sum(r["energy"] for r in rows) / len(rows), 1) if rows else 0
    labels = [datetime.fromisoformat(r["created_at"]).strftime("%d %b") for r in chart_rows]
    mood_data = [r["mood"] for r in chart_rows]
    stress_data = [r["stress"] for r in chart_rows]
    wellness_data = [r["wellness_score"] for r in chart_rows]
    return render_template(
        "dashboard.html", user=user, rows=rows, latest=latest, overall_risk=overall_risk,
        avg_stress=avg_stress, avg_mood=avg_mood, avg_sleep=avg_sleep, avg_energy=avg_energy,
        labels=labels, mood_data=mood_data, stress_data=stress_data, wellness_data=wellness_data,
    )


@app.route("/mindfulness")
def mindfulness():
    user = current_user()
    if not user:
        return redirect(url_for("index"))
    return render_template("mindfulness.html", user=user)


@app.route("/api/summary")
def summary():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    rows = get_recent_checkins(user["id"], 7)
    return jsonify({
        "checkins": len(rows),
        "risk": risk_from_checkins(rows),
        "avg_stress": round(sum(r["stress"] for r in rows) / len(rows), 1) if rows else 0,
        "avg_mood": round(sum(r["mood"] for r in rows) / len(rows), 1) if rows else 0,
        "avg_sleep": round(sum(r["sleep"] for r in rows) / len(rows), 1) if rows else 0,
    })


def open_browser_when_ready(host, port, timeout=15):
    """Open the browser only after Flask is actually accepting connections."""
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(f"http://{host}:{port}")
                return
        except OSError:
            time.sleep(0.25)
    # Even if the readiness check times out, give the user the URL.
    webbrowser.open(f"http://{host}:{port}")


if __name__ == "__main__":
    init_db()
    host = "127.0.0.1"
    port = int(os.environ.get("MINDMATE_PORT", "5000"))
    threading.Thread(target=open_browser_when_ready, args=(host, port), daemon=True).start()
    print("\nMindMate is starting...")
    print(f"Open in your browser: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
