# MindMate — AI Mental Wellness Companion for Students

MindMate is a local Flask wellness-support web app with student check-ins, an intent-aware NLP chatbot, stress/risk flagging, mindfulness exercises, and a personal dashboard.

> Educational wellness-support prototype only. It does not diagnose or treat mental-health conditions and is not an emergency service.

## The important improvement: one-click startup

You **do not have to manually run `pip install` before pressing Run**.

When `app.py` is started directly from VS Code, PyCharm, IDLE, another IDE, or a terminal, it checks for the required Python packages and automatically installs anything missing from `requirements.txt` using the same Python interpreter that launched the app.

After dependencies are available, it starts Flask and automatically opens the browser at:

`http://127.0.0.1:5000`

The browser is opened only after the Flask server is ready, so the first page should not open too early.

### VS Code

1. Open the `MindMate_Improved` folder.
2. Open `app.py`.
3. Select your normal Python interpreter if VS Code asks.
4. Click **Run ▶**.
5. On a new computer, the first run installs dependencies automatically.
6. The MindMate webpage opens automatically.

### Other IDEs

Run `app.py` normally. The same automatic dependency setup is performed before Flask starts.

### Double-click option on Windows

You can also double-click:

`run.bat`

For an explicit first-time setup, double-click:

`setup.bat`

## Requirements

- Python 3.11+ recommended.
- Internet access is required the first time dependencies need to be installed automatically.
- Later launches can run without reinstalling packages.

The requirements use scikit-learn 1.8+ on Python 3.11–3.14 and a 1.7.x fallback for older supported Python versions. scikit-learn 1.8 supports Python 3.11–3.14.

## Chatbot improvements

The chatbot now separates **intent/topic detection** from emotion detection. This prevents a question such as:

> best food for reducing stress

from being answered only with a generic stress/exhaustion message.

It detects topics such as:

- food and nutrition
- sleep
- exam/study pressure
- breathing
- grounding
- exercise
- focus
- motivation
- loneliness
- anger/frustration
- self-care/stress relief
- stress/anxiety definitions
- general wellbeing
- crisis/safety

It combines TF-IDF + Logistic Regression with high-precision keyword overrides for obvious topics and safety-critical phrases.

## Project structure

```text
MindMate_Improved/
├── app.py
├── ml_engine.py
├── requirements.txt
├── run.bat
├── setup.bat
├── README.md
├── .gitignore
├── data/
│   └── emotion_dataset.csv
├── models/
│   └── generated model files
├── database/
│   └── wellness.db (created automatically)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── checkin.html
│   ├── chat.html
│   ├── dashboard.html
│   └── mindfulness.html
└── static/
    ├── css/style.css
    └── js/chat.js
```

## GitHub

The repository can be cloned or downloaded as a ZIP. Open the downloaded folder in an IDE and run `app.py`. Missing Python packages are installed automatically on first launch.

Do **not** upload generated `database/*.db`, Python cache files, or generated model pickle files to GitHub; `.gitignore` excludes them.

## How the AI works

```text
Student message
      ↓
Text normalization
      ↓
Crisis/safety check
      ↓
Emotion classifier ──────────┐
                             ├──> risk + intent-aware response
Intent classifier ───────────┘
      ↓
High-precision topic overrides
      ↓
Relevant response
```

## Safety

The chatbot is not a substitute for a qualified professional. If a user is in immediate danger, use local emergency services. The India-focused demo provides Tele-MANAS (14416 / 1800-89-14416) and emergency services (112); verify current resources before real-world deployment.
