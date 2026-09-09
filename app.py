import os
import sys
import csv
import subprocess
from datetime import datetime, timedelta
 
import joblib
import pandas as pd
import psycopg2
import psycopg2.extras
from flask import Flask, render_template, request, redirect, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
 
model = joblib.load("model.pkl")
 
app = Flask(__name__)
# Reads from environment in production; falls back to a dev-only key locally.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
 
DATABASE_URL = os.environ["DATABASE_URL"]
 
 
def get_db():
    """Open a fresh PostgreSQL connection per call. Data lives on Render's
    persistent database service now, so it survives restarts/redeploys."""
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn
 
 
def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id SERIAL PRIMARY KEY,
            time TEXT,
            comment TEXT,
            result TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
 
    # Create a default admin account, but with a HASHED password so it can
    # actually pass check_password_hash() at login time.
    cur.execute("SELECT * FROM users WHERE username=%s", ("admin",))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            ("admin", generate_password_hash("1234")),
        )
        conn.commit()
 
    cur.close()
    conn.close()
 
 
init_db()
 
# Basic brute-force protection: tracks failed login attempts per username.
# Resets when the app restarts (fine for a small project; a real production
# app would store this in the DB or a cache like Redis instead).
MAX_ATTEMPTS = 5
LOCKOUT_MINUTES = 5
failed_attempts = {}  # { username: {"count": int, "locked_until": datetime or None} }
 
 
def is_locked_out(username):
    entry = failed_attempts.get(username)
    if entry and entry["locked_until"] and datetime.now() < entry["locked_until"]:
        return True
    return False
 
 
def record_failed_attempt(username):
    entry = failed_attempts.setdefault(username, {"count": 0, "locked_until": None})
    entry["count"] += 1
    if entry["count"] >= MAX_ATTEMPTS:
        entry["locked_until"] = datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)
        entry["count"] = 0
 
 
def reset_attempts(username):
    failed_attempts.pop(username, None)
 
 
def fetch_history(filter_type="all", search_query=""):
    """Always reads straight from the DB, so it never drifts out of sync
    (fixes the old global in-memory `history` list)."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, time, comment, result FROM comments ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
 
    rows = [dict(r) for r in rows]
 
    if filter_type != "all":
        rows = [r for r in rows if r["result"] == filter_type]
 
    if search_query:
        q = search_query.lower()
        rows = [
            r for r in rows
            if q in r["comment"].lower() or q in r["result"].lower()
        ]
 
    return rows
 
 
@app.route("/", methods=["GET", "POST"])
def home():
    if "logged_in" not in session:
        return redirect("/login")
 
    confidence = 0
    result = ""
    last_comment = ""
    prediction_details = {}
 
    if request.method == "POST":
        comment = request.form.get("comment", "").strip()
        last_comment = comment
 
        if comment:
            result = model.predict([comment])[0]
            probabilities = model.predict_proba([comment])[0]
            classes = model.classes_
            prediction_details = dict(zip(classes, probabilities))
            confidence = max(probabilities) * 100
 
            current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
 
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO comments (time, comment, result) VALUES (%s, %s, %s)",
                (current_time, comment, result),
            )
            conn.commit()
            cur.close()
            conn.close()
 
    filter_type = request.args.get("filter", "all")
    search_query = request.args.get("search", "")
    history = fetch_history(filter_type, search_query)
 
    all_history = fetch_history()  # unfiltered, for accurate counts
    safe_count = sum(1 for r in all_history if r["result"] == "safe")
    toxic_count = sum(1 for r in all_history if r["result"] == "toxic")
    spam_count = sum(1 for r in all_history if r["result"] == "spam")
    total_comments = len(all_history)
 
    return render_template(
        "index.html",
        result=result,
        history=history,
        safe_count=safe_count,
        toxic_count=toxic_count,
        spam_count=spam_count,
        username=session.get("username"),
        total_comments=total_comments,
        filter_type=filter_type,
        search_query=search_query,
        last_comment=last_comment,
        confidence=confidence,
        prediction_details=prediction_details,
        model_name="Multinomial Naive Bayes",
    )
 
 
@app.route("/clear")
def clear():
    if "logged_in" not in session:
        return redirect("/login")
 
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM comments")
    conn.commit()
    cur.close()
    conn.close()
    return redirect("/")
 
 
@app.route("/delete/<int:comment_id>")
def delete(comment_id):
    if "logged_in" not in session:
        return redirect("/login")
 
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM comments WHERE id = %s", (comment_id,))
    conn.commit()
    cur.close()
    conn.close()
    return redirect("/")
 
 
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
 
        if is_locked_out(username):
            return render_template(
                "login.html",
                error=f"Too many failed attempts. Try again in a few minutes."
            )
 
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        conn.close()
 
        if user and check_password_hash(user["password"], password):
            reset_attempts(username)
            session["logged_in"] = True
            session["username"] = username
            return redirect("/")
 
        record_failed_attempt(username)
        return render_template("login.html", error="Invalid username or password")
 
    return render_template("login.html", error="")
 
 
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
 
        if not username or not password:
            return render_template("register.html", error="Username and password are required.")
 
        if len(password) < 6:
            return render_template("register.html", error="Password must be at least 6 characters.")
 
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
 
        if cur.fetchone():
            conn.close()
            return render_template("register.html", error="Username already exists!")
 
        cur.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, generate_password_hash(password)),
        )
        conn.commit()
        conn.close()
        return redirect("/login")
 
    return render_template("register.html", error="")
 
 
@app.route("/", methods=["GET", "POST"])
def home():
    if "logged_in" not in session:
        return render_template("landing.html")
 
    global model
    # sys.executable instead of "python" - works reliably on hosts like
    # Render where the command may only be available as "python3".
    result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
 
    if result.returncode != 0:
        return f"Retraining failed:<br><pre>{result.stderr}</pre>"
 
    model = joblib.load("model.pkl")
    return redirect("/")
 
 
@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    session.pop("username", None)
    return redirect("/login")
 
 
@app.route("/upload", methods=["POST"])
def upload():
    if "logged_in" not in session:
        return redirect("/login")
 
    if "csvfile" not in request.files:
        return "No file selected."
 
    file = request.files["csvfile"]
    if file.filename == "":
        return "No file selected."
 
    try:
        df = pd.read_csv(file)
    except Exception:
        return "Could not read that file. Please upload a valid CSV."
 
    if "comment" not in df.columns:
        return "CSV must contain a 'comment' column."
 
    df["comment"] = df["comment"].fillna("").astype(str)
 
    try:
        predictions = model.predict(df["comment"])
    except Exception as e:
        return f"Prediction failed: {e}"
 
    df["prediction"] = predictions
 
    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)
    return send_file(output_file, as_attachment=True)
 
 
@app.route("/export")
def export():
    if "logged_in" not in session:
        return redirect("/login")
 
    history = fetch_history()
    with open("history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Comment", "Result"])
        for row in history:
            writer.writerow([row["time"], row["comment"], row["result"]])
 
    return send_file("history.csv", as_attachment=True)
 
 
# ---------------------------------------------------------------------
# Public API endpoint — lets other apps check a comment programmatically
# without logging into the website. Protected by an API key so random
# people can't spam your model for free.
# ---------------------------------------------------------------------
API_KEY = os.environ.get("API_KEY", "change-this-key-12345")
 
 
@app.route("/api/check", methods=["POST"])
def api_check():
    provided_key = request.headers.get("X-API-Key")
    if provided_key != API_KEY:
        return {"error": "Invalid or missing API key"}, 401
 
    data = request.get_json(silent=True)
    if not data or "comment" not in data:
        return {"error": "Request must be JSON with a 'comment' field"}, 400
 
    comment = str(data["comment"]).strip()
    if not comment:
        return {"error": "Comment cannot be empty"}, 400
 
    result = model.predict([comment])[0]
    probabilities = model.predict_proba([comment])[0]
    classes = model.classes_
    confidence = max(probabilities) * 100
 
    return {
        "comment": comment,
        "result": result,
        "confidence": round(confidence, 2),
        "probabilities": {cls: round(p * 100, 2) for cls, p in zip(classes, probabilities)}
    }
 
 
if __name__ == "__main__":
    app.run(debug=True)
 