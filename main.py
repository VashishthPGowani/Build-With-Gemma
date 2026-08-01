"""PitBrain live server entry point.

Run:  uv run main.py           (server on http://127.0.0.1:8000)
Then: uv run scripts/replay_race.py   (in a second terminal)
"""

import uvicorn


def main() -> None:
    uvicorn.run("f1coach.api:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
