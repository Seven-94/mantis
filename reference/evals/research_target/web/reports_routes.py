"""Report export routes."""

from flask import Blueprint, jsonify, request

from services import reports

reports_bp = Blueprint("reports", __name__)


@reports_bp.route("/reports/render", methods=["POST"])
def render_report():
    layout = request.form.get("layout", "")
    period = request.form.get("period", "month")
    summary = {"period": period, "total": 0}
    return jsonify({"body": reports.render_summary(layout, summary)})


@reports_bp.route("/reports/footer")
def report_footer():
    return jsonify({"footer": reports.render_footer()})
