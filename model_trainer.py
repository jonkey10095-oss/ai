 """
model_trainer.py
навчання моделі для Simple Flappy Bird
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "flappy_data.csv"
MODEL_FILE = BASE_DIR / "flappy_model.joblib"


# Ознаки, які описують поточний стан гри
FEATURE_COLUMNS = [
    "bird_y",
    "bird_speed",
    "distance_to_pipe",
    "distance_to_gap",
]


# Дія, яку модель повинна прогнозувати:
# jump або wait
TARGET_COLUMN = "action"


def load_data():
    """Завантажує датасет із CSV-файлу."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Файл flappy_data.csv не знайдено. "
            "Спочатку запустіть гру та зберіть дані."
        )

    df = pd.read_csv(DATA_FILE)

    print("Перші рядки датасету:")
    print(df.head())

    print("\nКількість рядків:")
    print(len(df))

    print("\nКількість прикладів кожної дії:")
    print(df[TARGET_COLUMN].value_counts())

    return df


def prepare_data(df):
    """Готує ознаки та цільову колонку."""

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    # Залишаємо тільки потрібні для моделі колонки
    df = df[required_columns].copy()

    # Видаляємо рядки з порожніми значеннями
    df = df.dropna()

    # X містить стани гри
    X = df[FEATURE_COLUMNS]

    # y містить дії гравця
    y = df[TARGET_COLUMN]

    return X, y


def split_data(X, y):
    """Розділяє дані на навчальну та тестову частини."""

    if len(X) < 30:
        raise ValueError(
            "У датасеті замало рядків. "
            "Зіграйте ще декілька разів."
        )

    if y.nunique() < 2:
        raise ValueError(
            "У датасеті повинні бути обидві дії: "
            "jump і wait."
        )

    action_counts = y.value_counts()

    if action_counts.min() < 2:
        raise ValueError(
            "Кожна дія повинна зустрічатися "
            "хоча б два рази."
        )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def train_model(X_train, y_train):
    """Створює та навчає дерево рішень."""

    model = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Перевіряє точність навченої моделі."""

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    train_accuracy = model.score(
        X_train,
        y_train,
    )

    print("\nРезультати навчання")
    print("-------------------")

    print(
        "Навчальних прикладів:",
        len(X_train),
    )

    print(
        "Тестових прикладів:",
        len(X_test),
    )

    print(
        "Точність на train:",
        round(train_accuracy, 2),
    )

    print(
        "Точність на test:",
        round(accuracy, 2),
    )

    results = pd.DataFrame({
        "real_action": y_test,
        "predicted_action": predictions,
    })

    print("\nПерші прогнози:")
    print(results.head(10))

    print("\nТаблиця помилок:")
    print(
        pd.crosstab(
            results["real_action"],
            results["predicted_action"],
        )
    )


def test_new_state(model):
    """Прогнозує дію для нового стану гри."""

    new_state = pd.DataFrame(
        [
            {
                "bird_y": 300,
                "bird_speed": 4.5,
                "distance_to_pipe": 120,
                "distance_to_gap": -70,
            }
        ]
    )

    predicted_action = model.predict(
        new_state
    )[0]

    print("\nНовий стан гри:")
    print(new_state)

    print(
        "\nПрогнозована дія:",
        predicted_action,
    )


def save_model(model):
    """Зберігає модель та список ознак."""

    model_data = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump(
        model_data,
        MODEL_FILE,
    )

    print(
        "\nМодель збережено у файл:",
        MODEL_FILE,
    )


def main():
    df = load_data()

    X, y = prepare_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(X, y)

    model = train_model(
        X_train,
        y_train,
    )

    evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    test_new_state(model)

    save_model(model)


if __name__ == "__main__":
    main()