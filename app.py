from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    flash,
)
import json
from dotenv import load_dotenv
import os
import time
import logging
import re
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from sqlalchemy import text
from openai import OpenAI
from datetime import datetime, timedelta
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv(
    "SECRET_KEY", "aced3162316523rdea121723"
) 

database_url = os.getenv("DATABASE_URL", "sqlite:///robot_control.db")
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default="user")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


openai_api_key = (
    os.getenv("OPENAI_API_KEY")
    or "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"
)

command_history = {}

rate_limits = {
    "admin": {"requests": 50, "period": 3600},  
    "user": {"requests": 20, "period": 3600},  
}

login_attempts = {}  
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_TIME_MINUTES = 15

PASSWORD_MIN_LENGTH = 8
PASSWORD_PATTERN = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&#])[A-Za-z\d@$!%*?&#]{8,}$"
)


def init_db():
    db.create_all()
    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin", role="admin")
        admin.set_password("admin")
        db.session.add(admin)
        db.session.commit()
        logger.info("Created initial admin user")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))

        user = User.query.get(session["user_id"])
        if not user or user.role != "admin":
            flash("Admin access required")
            return redirect(url_for("home"))

        return f(*args, **kwargs)

    return decorated_function


def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401

        user = User.query.get(session["user_id"])
        if not user:
            return jsonify({"error": "Authentication required"}), 401

        role = user.role
        limit = rate_limits.get(role, rate_limits["user"])

        user_id = str(user.id)

        if user_id not in command_history:
            command_history[user_id] = []

        current_time = time.time()
        command_history[user_id] = [
            t
            for t in command_history[user_id]
            if isinstance(t, float) and current_time - t < limit["period"]
        ]

        if len(command_history[user_id]) >= limit["requests"]:
            return (
                jsonify(
                    {
                        "error": f"Rate limit exceeded. Maximum {limit['requests']} requests per {limit['period']//3600} hour(s).",
                        "retry_after": limit["period"]
                        - (current_time - command_history[user_id][0]),
                    }
                ),
                429,
            )

        command_history[user_id].append(current_time)

        return f(*args, **kwargs)

    return decorated_function


def validate_password(password):
    """
    Validates password against the password policy.
    Returns a list of error messages, or an empty list if password is valid.
    """
    errors = []

    if len(password) < PASSWORD_MIN_LENGTH:
        errors.append(
            f"Password must be at least {PASSWORD_MIN_LENGTH} characters long"
        )

    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")

    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter")

    if not re.search(r"\d", password):
        errors.append("Password must contain at least one digit")

    if not re.search(r"[@$!%*?&#]", password):
        errors.append("Password must contain at least one special character (@$!%*?&#)")

    return errors


def interpret_command(command, previous_commands=None):
    """
    Enhanced function to interpret human commands with context from previous commands.
    Improved to handle directional commands more logically.
    """
    system_prompt = """You are an AI that converts natural language movement instructions into structured JSON commands for a 4-wheeled robot.

    You MUST ONLY output valid JSON. No explanations, text, or markdown formatting.

    Input: Natural language instructions for robot movement
    Output: JSON object representing the commands

    **Supported Movements:**
    - Linear motion: Use "direction": "forward" or "backward", with speed (m/s) and either distance (m) or time (s)
    - Rotation: Use "direction": "left" or "right", with degrees
    - Arc movements: Use "mode": "arc" for curved paths with specified radius and direction

    **Output Format:**
    {
      "commands": [
        {
          "mode": "linear|rotate|arc|stop",
          "direction": "forward|backward|left|right",
          "speed": float,  // meters per second (0.1-2.0)
          "distance": float,  // meters 
          "time": float,  // seconds (if applicable)
          "rotation": float,  // degrees (if applicable)
          "turn_radius": float,  // meters (for arc movements)
          "avoid_obstacle": "true | false"  // to stop or not to stop and avoid the obstacle (if applicable)
        },
        // Additional commands for sequences
      ],
      "description": "Brief human-readable description of what the robot will do"
    }

    **IMPORTANT RULES:**
    1. For rotation movements:
       - Use "mode": "rotate" with "direction": "left" or "right"
       - Always specify a rotation value in degrees 
       1.2) for arc mode specify: turn radius and distance(distance of the arc)

    2. For linear movements:
       - Use "mode": "linear" with "direction": "forward" or "backward"
       - Never use "left" or "right" as direction for linear movements
       - For "go right" type instructions, interpret as "rotate right, then go forward"
       - For "go left quickly for 5 meters", interpret as "rotate left, then go forward for 5 meters"

    For shapes, break them down into appropriate primitive movements:
    - Square: 4 forward movements with 4 90° right/left turns
    - Circle: A arc that forms a complete 360° path
    - Triangle: 3 forward movements with 120° turns
    - Rectangle: 2 pairs of different-length forward movements with 90° turns
    - Figure-eight: Two connected circles in opposite directions
    -question mark figure(complrx shape example) : half circle then downwards line movement
    Always provide complete, valid JSON that a robot can execute immediately.

    """
    user_prompt = f'Convert this command into a structured robot command: "{command}"'

    if previous_commands and len(previous_commands) > 0:
        recent_commands = previous_commands[-3:]  
        context = "Previous commands for context:\n" + "\n".join(
            [f"- {cmd}" for cmd in recent_commands]
        )
        user_prompt = context + "\n\n" + user_prompt

    try:
        client = OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        logger.info(f"Raw LLM output: {raw_output}")

        try:
            parsed_data = json.loads(raw_output)

            if "timestamp" in parsed_data:
                del parsed_data["timestamp"]

            if "sequence_type" in parsed_data:
                del parsed_data["sequence_type"]

            parsed_data["original_command"] = command

            if "commands" not in parsed_data:
                parsed_data["commands"] = [
                    {
                        "mode": "stop",
                        "description": "Invalid command structure - missing commands array",
                    }
                ]

            try:
                print("Im trying to talk to the robot")
                response = requests.get("http://192.168.0.20/")
                print(response.text)
                ip_address = "192.168.0.20"
                url = f"http://{ip_address}/json"
                json_str = json.dumps(parsed_data)
                payload = f"plain={json_str}"

                headers = {
                    "Content-Type": "application/x-www-form-urlencoded",
                }
                print(payload)
                response = requests.post(url, data=payload, headers=headers)
                print("*" * 20)
                print(response.text)
                print("*" * 20)
            except:
                logger.error("Failed to send command to ESP32")
                pass
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}, raw output: {raw_output}")

            json_match = re.search(r"```json(.*?)```", raw_output, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1).strip()
                    return json.loads(json_str)
                except:
                    pass

            return {
                "error": "Failed to parse response as JSON",
                "commands": [
                    {
                        "mode": "stop",
                        "description": "Command parsing error - robot stopped",
                    }
                ],
                "description": "Error in command processing",
            }

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return {
            "error": str(e),
            "commands": [{"mode": "stop", "description": "API error - robot stopped"}],
            "description": "Error in API communication",
        }


@app.route("/")
def login():
    if "user_id" in session:
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/auth", methods=["POST"])
def auth():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    ip_address = request.remote_addr

    if username in login_attempts:
        attempts, timestamp = login_attempts[username]

        if attempts >= MAX_LOGIN_ATTEMPTS:
            lockout_until = timestamp + timedelta(minutes=LOCKOUT_TIME_MINUTES)
            if datetime.now() < lockout_until:
                remaining_minutes = (
                    int((lockout_until - datetime.now()).total_seconds() / 60) + 1
                )
                flash(
                    f"Account is locked due to too many failed attempts. Try again in {remaining_minutes} minutes.",
                    "error",
                )
                return redirect(url_for("login"))
            else:
                login_attempts[username] = [0, datetime.now()]

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        if username in login_attempts:
            login_attempts.pop(username)

        session["user_id"] = user.id
        session["username"] = user.username
        return redirect(url_for("home"))
    else:
        if username in login_attempts:
            attempts, _ = login_attempts[username]
            login_attempts[username] = [attempts + 1, datetime.now()]
        else:
            login_attempts[username] = [1, datetime.now()]

        attempts = login_attempts[username][0]
        remaining_attempts = MAX_LOGIN_ATTEMPTS - attempts

        if remaining_attempts <= 0:
            flash(
                f"Too many failed login attempts. Account locked for {LOCKOUT_TIME_MINUTES} minutes.",
                "error",
            )
        else:
            flash(
                f"Invalid username or password. {remaining_attempts} attempts remaining before account lockout.",
                "error",
            )

        return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not password:
            flash("Username and password are required", "error")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for("register"))

        password_errors = validate_password(password)
        if password_errors:
            for error in password_errors:
                flash(error, "error")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
            return redirect(url_for("register"))

        new_user = User(username=username, role="user")
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    return render_template("home.html")


@app.route("/send_command", methods=["POST"])
@login_required
@rate_limit
def send_command():
    try:
        command = request.form.get("command", "").strip()
        user_id = str(session.get("user_id"))

        if not command:
            return jsonify({"error": "No command provided"})

        user_commands = []
        if user_id in command_history:
            user_commands = [
                item["original_command"]
                for item in command_history[user_id]
                if isinstance(item, dict) and "original_command" in item
            ]

        interpreted_command = interpret_command(command, user_commands)

        if user_id not in command_history:
            command_history[user_id] = []
        command_history[user_id].append(interpreted_command)

        if len(command_history[user_id]) > 10:
            command_history[user_id] = command_history[user_id][-10:]

        return jsonify(interpreted_command)

    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    users = User.query.all()

    db_type = "SQLite"
    db_location = app.config["SQLALCHEMY_DATABASE_URI"]

    if "postgresql" in db_location:
        db_type = "PostgreSQL"
        db_location = re.sub(r":[^@/]+@", ":***@", db_location)

    query_result = session.pop("query_result", None)

    return render_template(
        "admin.html",
        users=users,
        db_type=db_type,
        db_location=db_location,
        query_result=query_result,
    )


@app.route("/admin/edit_user/<int:user_id>", methods=["GET", "POST"])
@login_required
@admin_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)

    if request.method == "POST":
        username = request.form.get("username")
        role = request.form.get("role")
        new_password = request.form.get("password")

        if not username:
            flash("Username is required", "error")
            return redirect(url_for("edit_user", user_id=user_id))

        existing_user = User.query.filter_by(username=username).first()
        if existing_user and existing_user.id != user.id:
            flash("Username already exists", "error")
            return redirect(url_for("edit_user", user_id=user_id))

        user.username = username
        user.role = role

        if new_password:
            password_errors = validate_password(new_password)
            if password_errors:
                for error in password_errors:
                    flash(error, "error")
                return redirect(url_for("edit_user", user_id=user_id))

            user.set_password(new_password)

        try:
            db.session.commit()
            flash("User updated successfully", "success")
            return redirect(url_for("admin_panel"))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating user: {str(e)}")
            flash(f"Error updating user: {str(e)}", "error")
            return redirect(url_for("edit_user", user_id=user_id))

    return render_template("edit_user.html", user=user)


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)

    if user.role == "admin" and User.query.filter_by(role="admin").count() <= 1:
        flash("Cannot delete the last admin user", "error")
        return redirect(url_for("admin_panel"))

    try:
        db.session.delete(user)
        db.session.commit()
        flash("User deleted successfully", "success")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting user: {str(e)}")
        flash(f"Error deleting user: {str(e)}", "error")

    return redirect(url_for("admin_panel"))


@app.route("/run_query", methods=["POST"])
@login_required
@admin_required
def run_query():
    query = request.form.get("query", "").strip()

    if not query.lower().startswith("select"):
        flash("Only SELECT queries are allowed", "error")
        return redirect(url_for("admin_panel"))

    try:
        result = {}
        with db.engine.connect() as conn:
            result_proxy = conn.execute(text(query))
            result["columns"] = result_proxy.keys()
            result["rows"] = [list(row) for row in result_proxy]

        flash(
            f'Query executed successfully: {len(result["rows"])} rows returned',
            "success",
        )

        session["query_result"] = result

    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        flash(f"Error executing query: {str(e)}", "error")
        session["query_result"] = {"message": f"Error: {str(e)}"}

    return redirect(url_for("admin_panel"))


# # ESP32 API Endpoint
# @app.route("/api/robot_command", methods=["GET", "POST"])
# def robot_command():
#     # Simple authentication using API key instead of session-based auth
#     api_key = request.headers.get("X-API-Key")
#     if not api_key or api_key != os.getenv("ROBOT_API_KEY", "1234"):
#         return jsonify({"error": "Invalid API key"}), 401

#     # For GET requests, return the latest command for the robot
#     if request.method == "GET":
#         # This could be any user's most recent command
#         for user_id, commands in command_history.items():
#             if commands:
#                 # Find the most recent valid command
#                 for item in reversed(commands):
#                     if isinstance(item, dict) and "commands" in item:
#                         return jsonify(item)

#         return jsonify({"error": "No commands available"}), 404

#     # For POST requests, allow the ESP32 to send status updates
#     elif request.method == "POST":
#         try:
#             data = request.get_json()
#             # Process status update from ESP32
#             logger.info(f"Received status update from ESP32: {data}")

#             # Store the status update if needed
#             if "status" in data and "commandId" in data:
#                 status_update = {
#                     "timestamp": time.time(),
#                     "status": data["status"],
#                     "commandId": data["commandId"],
#                 }

#                 # You could store this in a database or in memory
#                 if "esp32_status" not in command_history:
#                     command_history["esp32_status"] = []

#                 command_history["esp32_status"].append(status_update)

#                 # Keep only the last 20 status updates
#                 if len(command_history["esp32_status"]) > 20:
#                     command_history["esp32_status"] = command_history["esp32_status"][
#                         -20:
#                     ]

#             return jsonify({"status": "received"}), 200
#         except Exception as e:
#             logger.error(f"Error processing ESP32 status update: {str(e)}")
#             return jsonify({"error": str(e)}), 400


with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
