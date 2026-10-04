"""Order dashboard routes."""

from flask import Blueprint, jsonify, request

from services import imports, notify, orders

orders_bp = Blueprint("orders", __name__)


@orders_bp.route("/orders/grid")
def orders_grid():
    sort = request.args.get("sort")
    direction = request.args.get("dir", "DESC")
    return jsonify(orders.search_orders(sort, direction))


@orders_bp.route("/orders/<int:order_id>/notify", methods=["POST"])
def notify_customer(order_id):
    recipient = request.form.get("recipient", "")
    subject = request.form.get("subject", f"Update for order {order_id}")
    note = request.form.get("note", "")
    notify.send_status_mail(recipient, subject, note)
    return jsonify({"sent": True, "order": order_id})


@orders_bp.route("/orders/<int:order_id>")
def order_detail(order_id):
    return jsonify(orders.order_details(order_id))


@orders_bp.route("/orders/import", methods=["POST"])
def import_batch():
    batch_id = request.form.get("batch_id", "")
    payload = request.get_data()
    return jsonify(imports.stage_order_batch(batch_id, payload))


@orders_bp.route("/orders/import/<batch_id>/review")
def review_batch(batch_id):
    return jsonify(imports.load_order_batch(batch_id))


@orders_bp.route("/orders/import/bundle", methods=["POST"])
def import_bundle():
    upload = request.files["bundle"]
    saved = f"/srv/portal/uploads/{upload.filename}"
    upload.save(saved)
    return jsonify(imports.unpack_supplier_bundle(saved))
