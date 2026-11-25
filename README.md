# Shor Lab Hardware

Type a number, and Shor's algorithm runs on a simulator, BlueQubit, or a real IBM quantum chip.
The factors come back in the page.

Run locally:

    pip install -r requirements.txt
    python shor_app.py        # open http://localhost:8000

Deploy free on Render: New > Web Service > connect this repo > Runtime: Docker > Instance type: Free.

Visitors bring their own API keys. Keys are sent with each request, used for that request only,
and never stored or logged. The server's own environment keys are ignored (`SHOR_PUBLIC=1`).
