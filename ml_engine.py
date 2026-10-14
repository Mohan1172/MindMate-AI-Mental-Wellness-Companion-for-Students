from pathlib import Path
import csv
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class WellnessEngine:
    """Local intent + emotion engine for a student wellness companion.

    Intent classification is deliberately lightweight and deterministic enough for a
    local student project. It uses a curated training set, TF-IDF + Logistic
    Regression, and high-precision keyword overrides for safety-critical intents.
    """

    def __init__(self, dataset_path, model_dir):
        self.dataset_path = Path(dataset_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = None
        self.emotion_model = None
        self.intent_vectorizer = None
        self.intent_model = None
        self._train()

    def _load_data(self):
        texts, emotions, intents = [], [], []
        with open(self.dataset_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                texts.append(row["text"].strip())
                emotions.append(row["emotion"].strip())
                intents.append(row["intent"].strip())
        return texts, emotions, intents

    def _train(self):
        texts, emotions, intents = self._load_data()

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=5000,
            strip_accents="unicode",
        )
        X = self.vectorizer.fit_transform(texts)
        self.emotion_model = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
        self.emotion_model.fit(X, emotions)

        self.intent_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            max_features=8000,
            strip_accents="unicode",
        )
        XI = self.intent_vectorizer.fit_transform(texts)
        self.intent_model = LogisticRegression(max_iter=2500, random_state=42, class_weight="balanced")
        self.intent_model.fit(XI, intents)

        with open(self.model_dir / "emotion_tfidf.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.model_dir / "emotion_model.pkl", "wb") as f:
            pickle.dump(self.emotion_model, f)
        with open(self.model_dir / "intent_tfidf.pkl", "wb") as f:
            pickle.dump(self.intent_vectorizer, f)
        with open(self.model_dir / "intent_model.pkl", "wb") as f:
            pickle.dump(self.intent_model, f)

    @staticmethod
    def _normalise(text):
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _crisis(self, text):
        t = self._normalise(text)
        patterns = [
            r"\bsuicide\b", r"\bkill myself\b", r"\bend my life\b",
            r"\btake my life\b", r"\bself harm\b", r"\bself-harm\b",
            r"\bhurt myself\b", r"\bwant to die\b", r"\bwish i was dead\b",
            r"\bcan't go on\b", r"\bcannot go on\b", r"\bi don't want to live\b",
            r"\bno reason to live\b",
        ]
        return any(re.search(p, t) for p in patterns)

    def _risk(self, text, history=None):
        t = self._normalise(text)
        if self._crisis(t):
            return "Crisis"
        high_terms = [
            "hopeless", "can't cope", "cannot cope", "completely exhausted",
            "no sleep", "not sleeping", "panic attack", "breaking down",
            "feel unsafe", "out of control",
        ]
        moderate_terms = [
            "stressed", "stress", "anxious", "anxiety", "worried", "burnout",
            "tired", "exhausted", "pressure", "overwhelmed", "panic",
        ]
        personal_distress = any(x in t for x in [
            "i feel", "i'm", "im ", "my stress", "my anxiety", "i am ", "i can't", "i cannot",
            "i keep", "i have", "i'm feeling", "im feeling"
        ])
        if any(x in t for x in high_terms):
            return "High"
        if personal_distress and any(x in t for x in moderate_terms):
            return "Moderate"
        if history:
            recent = history[:7]
            avg_stress = sum(r["stress"] for r in recent) / len(recent)
            avg_sleep = sum(r["sleep"] for r in recent) / len(recent)
            avg_mood = sum(r["mood"] for r in recent) / len(recent)
            if avg_stress >= 8 or (avg_stress >= 7 and avg_sleep < 5):
                return "High"
            if avg_stress >= 6 or avg_mood <= 4:
                return "Moderate"
        return "Low"

    def _keyword_intent(self, t):
        """High precision overrides for obvious questions where generic emotion is misleading."""
        rules = [
            ("breathing", ["breathing exercise", "breathing technique", "breathe", "breath exercise"]),
            ("food_nutrition", ["food", "eat", "eating", "meal", "diet", "snack", "nutrition", "drink", "banana", "oats"]),
            ("sleep", ["sleep", "insomnia", "can't sleep", "cannot sleep", "sleeping", "awake all night"]),
            ("study_pressure", ["exam", "exams", "study", "studying", "assignment", "college", "semester", "marks", "grades", "cgpa", "deadline"]),
            ("grounding", ["grounding", "5-4-3-2-1", "panic", "panicking"]),
            ("exercise", ["exercise", "workout", "walk", "walking", "gym", "physical activity"]),
            ("focus", ["focus", "concentrate", "concentration", "distracted", "procrastinat"]),
            ("motivation", ["motivat", "lazy", "unmotivated", "can't start", "cannot start"]),
            ("loneliness", ["lonely", "alone", "no friends", "friendless", "isolated"]),
            ("anger", ["angry", "anger", "mad", "irritated", "irritation", "frustrated"]),
            ("self_care", ["self care", "self-care", "relax", "relaxation", "calm down", "reduce stress", "stress relief"]),
            ("definition", ["what is stress", "what is anxiety", "meaning of stress", "explain stress", "why do i feel stressed"]),
        ]
        for intent, words in rules:
            for w in words:
                if " " in w or any(ch in w for ch in "-?="):
                    matched = w in t
                elif len(w) <= 5:
                    matched = bool(re.search(r"\b" + re.escape(w) + r"\b", t))
                else:
                    # Longer stems such as "procrastinat" or "motivat" are intentional.
                    matched = w in t
                if matched:
                    return intent
        return None

    def analyze(self, text, history=None):
        text = str(text or "").strip()
        t = self._normalise(text)
        crisis = self._crisis(t)

        XE = self.vectorizer.transform([text])
        emotion_probs = self.emotion_model.predict_proba(XE)[0]
        emotion_idx = emotion_probs.argmax()
        emotion = self.emotion_model.classes_[emotion_idx]
        emotion_confidence = float(emotion_probs[emotion_idx])

        XI = self.intent_vectorizer.transform([text])
        intent_probs = self.intent_model.predict_proba(XI)[0]
        intent_idx = intent_probs.argmax()
        predicted_intent = self.intent_model.classes_[intent_idx]
        intent_confidence = float(intent_probs[intent_idx])

        override = self._keyword_intent(t)
        if override:
            intent = override
            intent_confidence = max(intent_confidence, 0.92)
        elif intent_confidence < 0.38:
            intent = "general_wellbeing"
        else:
            intent = predicted_intent

        risk = self._risk(text, history)
        if crisis:
            emotion = "Crisis"
            intent = "crisis"
            risk = "Crisis"

        return {
            "emotion": emotion,
            "emotion_confidence": round(emotion_confidence, 3),
            "intent": intent,
            "intent_confidence": round(intent_confidence, 3),
            "risk": risk,
            "crisis": crisis,
        }
