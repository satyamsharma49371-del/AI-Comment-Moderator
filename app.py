import os
import sys
import csv
import sqlite3
import subprocess
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash

model = joblib.load("model.pkl")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")

DB_PATH = "comments.db"


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            comment TEXT,
            result TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()

    cur.execute("SELECT * FROM users WHERE username='admin'")
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            ("admin", generate_password_hash("1234")),
        )
        conn.commit()

    conn.close()


init_db()


def fetch_history(filter_type="all", search_query=""):
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
            conn.execute(
                "INSERT INTO comments (time, comment, result) VALUES (?, ?, ?)",
                (current_time, comment, result),
            )
            conn.commit()
            conn.close()

    filter_type = request.args.get("filter", "all")
    search_query = request.args.get("search", "")
    history = fetch_history(filter_type, search_query)

    all_history = fetch_history()
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
    conn.execute("DELETE FROM comments")
    conn.commit()
    conn.close()
    return redirect("/")


@app.route("/delete/<int:comment_id>")
def delete(comment_id):
    if "logged_in" not in session:
        return redirect("/login")

    conn = get_db()
    conn.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
    conn.commit()
    conn.close()
    return redirect("/")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["logged_in"] = True
            session["username"] = username
            return redirect("/")

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html", error="")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        if not username or not password:
            return render_template("register.html", error="Username and password are required.")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))

        if cur.fetchone():
            conn.close()
            return render_template("register.html", error="Username already exists!")

        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        conn.commit()
        conn.close()
        return redirect("/login")

    return render_template("register.html", error="")


@app.route("/retrain")
def retrain():
    if "logged_in" not in session:
        return redirect("/login")

    global model
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


if __name__ == "__main__":
    app.run(debug=True)