from pygame import *
import random

# НОВЕ: бібліотеки для завантаження та використання моделі
from pathlib import Path
import joblib
import pandas as pd

from collect_data import DataCollector


init()

WIDTH = 600
HEIGHT = 500
FPS = 60

screen = display.set_mode(
    (WIDTH, HEIGHT)
)

display.set_caption(
    "Simple Flappy Bird"
)

clock = time.Clock()
game_font = font.Font(None, 40)

# НОВЕ: менший шрифт для режиму керування
info_font = font.Font(None, 28)


# Пташка
bird_x = 120
bird_y = 250
bird_radius = 18

bird_speed = 0
gravity = 0.5
jump_power = -8


# Перешкода
pipe_x = WIDTH
pipe_width = 70
pipe_gap = 170
pipe_speed = 4

gap_y = random.randint(
    120,
    HEIGHT - 120
)


score = 0
running = True


# Створюємо збирач даних
collector = DataCollector(
    "flappy_data.csv"
)

# Записуємо один приклад кожні 10 кадрів
collect_interval = 10
frame_count = 0


# НОВЕ: шлях до навченої моделі
MODEL_FILE = (
    Path(__file__).resolve().parent
    / "flappy_model.joblib"
)


# НОВЕ: ознаки, які використовувались під час навчання
DEFAULT_FEATURE_COLUMNS = [
    "bird_y",
    "bird_speed",
    "distance_to_pipe",
    "distance_to_gap",
]


# НОВЕ: завантаження моделі
def load_ai_model():
    if not MODEL_FILE.exists():
        print(
            "Модель не знайдено:",
            MODEL_FILE
        )

        print(
            "Гра працюватиме тільки "
            "в ручному режимі."
        )

        return None, DEFAULT_FEATURE_COLUMNS

    model_data = joblib.load(
        MODEL_FILE
    )

    # Модель була збережена разом
    # зі списком її ознак
    if isinstance(model_data, dict):
        loaded_model = model_data["model"]

        feature_columns = model_data.get(
            "feature_columns",
            DEFAULT_FEATURE_COLUMNS,
        )

        return loaded_model, feature_columns

    # Запасний варіант, якщо у файлі
    # була збережена лише модель
    return model_data, DEFAULT_FEATURE_COLUMNS


# НОВЕ: модель завантажується один раз
ai_model, ai_feature_columns = load_ai_model()


# НОВЕ: початковий режим — ручне керування
if ai_model is None:
    control_mode = "human"
else:
    control_mode = "ai"


# НОВЕ: модель приймає рішення
# не на кожному кадрі
ai_decision_interval = 5
ai_decision_counter = 0


# НОВЕ: затримка між стрибками AI
ai_jump_cooldown = 0
ai_jump_cooldown_frames = 8


# НОВЕ: остання дія моделі для відображення
last_ai_action = "wait"


# НОВЕ: створення стану для прогнозування
def create_game_state():
    state_values = {
        "bird_y": bird_y,
        "bird_speed": bird_speed,
        "distance_to_pipe": pipe_x - bird_x,
        "distance_to_gap": gap_y - bird_y,
    }

    # Колонки створюються в тому самому порядку,
    # у якому модель бачила їх під час навчання
    state = {
        column: state_values[column]
        for column in ai_feature_columns
    }

    return pd.DataFrame(
        [state]
    )


while running:
    clock.tick(FPS)

    frame_count += 1

    # За замовчуванням пташка не стрибає
    action = "wait"

    for current_event in event.get():
        if current_event.type == QUIT:
            running = False

        if current_event.type == KEYDOWN:
            # НОВЕ: клавіша M перемикає режим керування
            if current_event.key == K_m:
                if ai_model is None:
                    print(
                        "Неможливо ввімкнути AI: "
                        "модель не знайдена."
                    )

                elif control_mode == "human":
                    control_mode = "ai"

                    ai_decision_counter = (
                        ai_decision_interval
                    )

                    ai_jump_cooldown = 0

                    print(
                        "Режим керування: AI"
                    )

                else:
                    control_mode = "human"

                    print(
                        "Режим керування: HUMAN"
                    )

            # SPACE працює тільки
            # в ручному режимі
            if (
                current_event.key == K_SPACE
                and control_mode == "human"
            ):
                # Записуємо стан до виконання стрибка
                action = "jump"

                collector.collect(
                    bird_y=bird_y,
                    bird_speed=bird_speed,
                    pipe_x=pipe_x,
                    gap_y=gap_y,
                    bird_x=bird_x,
                    score=score,
                    action=action,
                )

                bird_speed = jump_power

    # НОВЕ: керування моделлю
    if (
        control_mode == "ai"
        and ai_model is not None
    ):
        ai_decision_counter += 1

        if ai_jump_cooldown > 0:
            ai_jump_cooldown -= 1

        if (
            ai_decision_counter
            >= ai_decision_interval
        ):
            ai_decision_counter = 0

            game_state = create_game_state()

            last_ai_action = ai_model.predict(
                game_state
            )[0]

            if (
                last_ai_action == "jump"
                and ai_jump_cooldown == 0
            ):
                bird_speed = jump_power

                ai_jump_cooldown = (
                    ai_jump_cooldown_frames
                )

    # Записуємо приклади без стрибка
    # тільки під час ручного керування
    if (
        control_mode == "human"
        and frame_count % collect_interval == 0
        and action == "wait"
    ):
        collector.collect(
            bird_y=bird_y,
            bird_speed=bird_speed,
            pipe_x=pipe_x,
            gap_y=gap_y,
            bird_x=bird_x,
            score=score,
            action=action,
        )

    # Гравітація
    bird_speed += gravity
    bird_y += bird_speed

    # Рух перешкоди
    pipe_x -= pipe_speed

    # Створення нової перешкоди
    if pipe_x + pipe_width < 0:
        pipe_x = WIDTH

        gap_y = random.randint(
            120,
            HEIGHT - 120
        )

        score += 1

    top_pipe = Rect(
        pipe_x,
        0,
        pipe_width,
        gap_y - pipe_gap // 2
    )

    bottom_pipe = Rect(
        pipe_x,
        gap_y + pipe_gap // 2,
        pipe_width,
        HEIGHT
    )

    bird_rect = Rect(
        bird_x - bird_radius,
        bird_y - bird_radius,
        bird_radius * 2,
        bird_radius * 2
    )

    if (
        bird_rect.colliderect(top_pipe)
        or bird_rect.colliderect(bottom_pipe)
    ):
        running = False

    if (
        bird_y - bird_radius < 0
        or bird_y + bird_radius > HEIGHT
    ):
        running = False

    screen.fill(
        (120, 200, 255)
    )

    draw.circle(
        screen,
        (255, 220, 0),
        (bird_x, int(bird_y)),
        bird_radius
    )

    draw.rect(
        screen,
        (40, 180, 70),
        top_pipe
    )

    draw.rect(
        screen,
        (40, 180, 70),
        bottom_pipe
    )

    score_text = game_font.render(
        f"Score: {score}",
        True,
        (255, 255, 255)
    )

    data_text = game_font.render(
        f"Rows: {collector.step}",
        True,
        (255, 255, 255)
    )

    # НОВЕ: текст поточного режиму
    if control_mode == "human":
        mode_name = "HUMAN"
    else:
        mode_name = "AI"

    mode_text = info_font.render(
        f"Mode: {mode_name} [M]",
        True,
        (255, 255, 255)
    )

    screen.blit(
        score_text,
        (20, 20)
    )

    screen.blit(
        data_text,
        (20, 55)
    )

    screen.blit(
        mode_text,
        (350, 20)
    )

    # НОВЕ: показуємо останню дію моделі
    if control_mode == "ai":
        prediction_text = info_font.render(
            f"AI action: {last_ai_action}",
            True,
            (255, 255, 255)
        )

        screen.blit(
            prediction_text,
            (350, 50)
        )

    display.update()


quit()