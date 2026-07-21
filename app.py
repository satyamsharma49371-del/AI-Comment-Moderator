from flask import Flask, render_template, request, redirect,session,send_file
import joblib
from datetime import datetime
import csv
import sqlite3
import subprocess
from werkzeug.security import generate_password_hash, check_password_hash
model = joblib.load("model.pkl")
import pandas as pd
app = Flask(__name__)
app.secret_key = "AI_COMMENT_MODERATOR_2026"
conn = sqlite3.connect("comments.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    comment TEXT,
    result TEXT
)
""")
conn.commit()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("SELECT * FROM users WHERE username='admin'")

if cursor.fetchone() is None:
    cursor.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        ("admin", "1234")
    )
    conn.commit()

cursor.execute("SELECT time, comment, result FROM comments")
history = list(cursor.fetchall())

@app.route("/", methods=["GET", "POST"])
def home():

    if "logged_in" not in session:
        return redirect("/login")

    confidence = 0
    result = ""
    last_comment = ""
    prediction_details = {}

    if request.method == "POST":

        comment = request.form["comment"]
        last_comment = comment

        result = model.predict([comment])[0]

        probabilities = model.predict_proba([comment])[0]
        classes = model.classes_

        prediction_details = dict(zip(classes, probabilities))

        confidence = max(probabilities) * 100

        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        history.append((current_time, comment, result))

        cursor.execute(
            "INSERT INTO comments (time, comment, result) VALUES (?, ?, ?)",
            (current_time, comment, result)
        )
        conn.commit()

    safe_count = sum(1 for t, c, r in history if r == "safe")
    toxic_count = sum(1 for t, c, r in history if r == "toxic")
    spam_count = sum(1 for t, c, r in history if r == "spam")
    total_comments = len(history)

    filter_type = request.args.get("filter", "all")
    filtered_history = history

    if filter_type != "all":
        filtered_history = [
            (t, c, r)
            for t, c, r in filtered_history
            if r == filter_type
        ]

    search_query = request.args.get("search", "").lower()

    if search_query:
        filtered_history = [
            (t, c, r)
            for t, c, r in filtered_history
            if search_query in c.lower() or search_query in r.lower()
        ]

    return render_template(
        "index.html",
        result=result,
        history=filtered_history,
        safe_count=safe_count,
        toxic_count=toxic_count,
        spam_count=spam_count,
        username=session.get("username"),
        total_comments=total_comments,
        filter_type=filter_type,
        last_comment=last_comment,
        confidence=confidence,
        search_query=search_query,
        prediction_details=prediction_details,
        model_name="Multinomial Naive Bayes"
    )

@app.route("/clear")
def clear():
    history.clear()

    cursor.execute("DELETE FROM comments")
    conn.commit()

    return render_template(
        "index.html",
        result="",
        history=[],
        safe_count=0,
        toxic_count=0,
        spam_count=0,
        total_comments=0,
        filter_type="all",
        last_comment="",
        confidence=None,
        search_query="",
        model_name="Multinomial Naive Bayes"
    )
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        cursor.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        )

        user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            session["logged_in"] = True
            session["username"] = username
            return redirect("/")

        return render_template(
            "login.html",
            error="Invalid Username or Password"
        )

    return render_template(
        "login.html",
        error=""
    )

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        cursor.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        )

        if cursor.fetchone():
            return render_template(
                "register.html",
                error="Username already exists!"
            )

        hashed_password = generate_password_hash(password)

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password)
        )

        conn.commit()

        return redirect("/login")

    return render_template(
        "register.html",
        error=""
    )

    @app.route("/delete/<int:index>")
    def delete(index):
     if 0 <= index < len(history):
        history.pop(index)

        cursor.execute("DELETE FROM comments")

        for time, comment, result in history:
            cursor.execute(
                "INSERT INTO comments (time, comment, result) VALUES (?, ?, ?)",
                (time, comment, result)
            )

        conn.commit()

    return redirect("/")

@app.route("/retrain")
def retrain():

    subprocess.run(["python", "train_model.py"])

    global model
    model = joblib.load("model.pkl")

    return redirect("/")
@app.route("/logout")
def logout():

    session.pop("logged_in", None)

    return redirect("/login")
@app.route("/upload", methods=["POST"])
def upload():

    if "csvfile" not in request.files:
        return "No file selected."

    file = request.files["csvfile"]

    if file.filename == "":
        return "No file selected."

    df = pd.read_csv(file)

    if "comment" not in df.columns:
        return "CSV must contain a 'comment' column."

    predictions = model.predict(df["comment"])

    df["prediction"] = predictions

    output_file = "predictions.csv"

    df.to_csv(output_file, index=False)
    return send_file(
    output_file,
    as_attachment=True
)
@app.route("/export")
def export():
    with open("history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["Time", "Comment", "Result"])

        for time, comment, result in history:
            writer.writerow([time, comment, result])

    return "CSV Exported Successfully!"


if __name__ == "__main__":
    app.run(debug=True)