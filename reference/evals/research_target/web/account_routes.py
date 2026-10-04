"""Account and session routes."""

from flask import Blueprint, jsonify, redirect, request

from services import accounts
from utils import urls

account_bp = Blueprint("account", __name__)


@account_bp.route("/account/update", methods=["POST"])
def update_account():
    user_id = request.form.get("user_id", "0")
    fields = {k: v for k, v in request.form.items() if k != "user_id"}
    return jsonify(accounts.update_profile(user_id, fields))


@account_bp.route("/account/reset", methods=["POST"])
def request_reset():
    username = request.form.get("username", "")
    token = accounts.start_password_reset(username)
    return jsonify({"sent": True, "token_preview": token[:4]})


@account_bp.route("/login/done")
def login_done():
    target = request.args.get("next", "/")
    if urls.is_relative_url(target):
        return redirect(target)
    return redirect("/")
