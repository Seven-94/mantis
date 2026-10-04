"""Webhook subscription routes."""

from flask import Blueprint, jsonify, request

from integrations import webhooks

hooks_bp = Blueprint("hooks", __name__)

_SUBSCRIPTIONS = {}


@hooks_bp.route("/hooks/register", methods=["POST"])
def register_hook():
    merchant = request.form.get("merchant", "")
    callback = request.form.get("callback_url", "")
    _SUBSCRIPTIONS[merchant] = callback
    return jsonify({"merchant": merchant, "registered": bool(callback)})


@hooks_bp.route("/hooks/test", methods=["POST"])
def test_hook():
    merchant = request.form.get("merchant", "")
    callback = _SUBSCRIPTIONS.get(merchant, "")
    if not callback:
        return jsonify({"error": "no callback registered"}), 404
    return jsonify(webhooks.ping(callback))
