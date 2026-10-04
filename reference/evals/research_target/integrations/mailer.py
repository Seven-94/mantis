"""Thin SMTP client for transactional mail."""

import smtplib

import config


def build_message(sender, recipient, subject, body):
    """Assembles a raw RFC 822 message."""
    clean_body = body.replace("\r", "").strip()
    headers = (
        f"From: {sender}\r\n"
        f"To: {recipient}\r\n"
        f"Subject: {subject}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
    )
    return headers + "\r\n" + clean_body


def send(recipient, subject, body):
    """Delivers one message through the configured relay."""
    message = build_message(
        f"no-reply@{config.get('app_name')}.example.com", recipient, subject, body
    )
    with smtplib.SMTP(config.get("smtp_host")) as client:
        client.login(config.get("smtp_user"), config.get("smtp_password"))
        client.sendmail(
            f"no-reply@{config.get('app_name')}.example.com", [recipient], message
        )
