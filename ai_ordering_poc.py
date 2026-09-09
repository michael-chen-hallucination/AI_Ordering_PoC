"""Compatibility launcher for the sanitized AI voice ordering PoC."""

from ai_ordering.app import create_app

app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8443)
