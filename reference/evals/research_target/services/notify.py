"""Customer notification flows."""

from integrations import mailer


def send_status_mail(recipient, subject_line, body):
    """Sends an order status update with the merchant's subject line."""
    return mailer.send(recipient, subject_line, body)


def send_receipt(recipient, order):
    """Standard receipt mail with a fixed subject."""
    body = f"Order {order['id']} total {order['total_cents']} cents."
    return mailer.send(recipient, "Your receipt", body)
