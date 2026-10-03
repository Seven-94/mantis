"""HTTP route layer for the document portal."""

from flask import Flask, jsonify, request, send_file

import admintools
import auth
import database
import fetcher
import helpers
import storage

app = Flask(__name__)


@app.route("/api/user")
def lookup_user():
    name = request.args.get("name", "")
    return jsonify(database.find_user(name))


@app.route("/api/products")
def list_products():
    category = request.args.get("category", "general")
    return jsonify(helpers.safe_query(category))


@app.route("/docs/view")
def view_document():
    doc = request.args.get("doc", "welcome.txt")
    return send_file(storage.fetch_document(doc))


@app.route("/admin/backup", methods=["POST"])
def admin_backup():
    label = request.form.get("label", "nightly")
    if not auth.is_admin(request.headers.get("X-Auth-Token", "")):
        return jsonify({"error": "forbidden"}), 403
    return jsonify(admintools.run_backup(label))


@app.route("/api/preview")
def preview_link():
    url = request.args.get("url", "")
    return jsonify({"preview": fetcher.preview(url)})


@app.route("/api/register", methods=["POST"])
def register():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    record = auth.create_account(username, password)
    return jsonify({"id": record["id"]})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": helpers.VERSION})
