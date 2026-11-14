import csv
from pathlib import Path
from uuid import uuid4


class DataCollector:
    # Колонки майбутнього датасету
    FIELDNAMES = [
        "session_id",
        "step",
        "bird_y",
        "bird_speed",
        "pipe_x",
        "gap_y",
        "distance_to_pipe",
        "distance_to_gap",
        "score",
        "action",
    ]

    def __init__(self, filename="flappy_data.csv"):
        self.filename = Path(filename)

        # Створюємо окремий ідентифікатор запуску гри
        self.session_id = uuid4().hex[:8]

        self.step = 0

    def collect(
        self,
        bird_y,
        bird_speed,
        pipe_x,
        gap_y,
        bird_x,
        score,
        action,
    ):
        self.step += 1

        row = {
            "session_id": self.session_id,
            "step": self.step,
            "bird_y": round(bird_y, 2),
            "bird_speed": round(bird_speed, 2),
            "pipe_x": pipe_x,
            "gap_y": gap_y,
            "distance_to_pipe": pipe_x - bird_x,
            "distance_to_gap": round(
                gap_y - bird_y,
                2
            ),
            "score": score,
            "action": action,
        }

        self.save(row)

    def save(self, row):
        file_exists = (
            self.filename.exists()
            and self.filename.stat().st_size > 0
        )

        with self.filename.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)