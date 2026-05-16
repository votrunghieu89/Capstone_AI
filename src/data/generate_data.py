import numpy as np
import pandas as pd
import math

def generate_repair_dataset(n_samples=10000, outlier_ratio=0.03, random_seed=42):
    np.random.seed(random_seed)

    services = [
        "Sửa chữa và bảo dưỡng máy lạnh",
        "Sửa chữa và bảo dưỡng máy giặt",
        "Sửa chữa và bảo dưỡng tủ lạnh",
        "Sửa chữa và bảo dưỡng quạt điện",
        "Sửa chữa và bảo dưỡng lò vi sóng",
        "Sửa chữa và bảo dưỡng nồi cơm điện",
        "Sửa chữa và bảo dưỡng laptop",
        "Sửa chữa và bảo dưỡng máy tính",
        "Sửa chữa và bảo dưỡng điện thoại",
        "Sửa chữa và bảo dưỡng TV",
        "Sửa chữa và bảo dưỡng điện dân dụng",
        "Sửa chữa và bảo dưỡng hệ thống nước",
        "Sửa chữa và bảo dưỡng xe máy",
        "Sửa chữa và bảo dưỡng ô tô",
        "Sửa chữa và bảo dưỡng máy in"
    ]

    service_base = {
        "Sửa chữa và bảo dưỡng máy lạnh": 150,
        "Sửa chữa và bảo dưỡng máy giặt": 130,
        "Sửa chữa và bảo dưỡng tủ lạnh": 120,
        "Sửa chữa và bảo dưỡng quạt điện": 50,
        "Sửa chữa và bảo dưỡng lò vi sóng": 90,
        "Sửa chữa và bảo dưỡng nồi cơm điện": 60,
        "Sửa chữa và bảo dưỡng laptop": 100,
        "Sửa chữa và bảo dưỡng máy tính": 80,
        "Sửa chữa và bảo dưỡng điện thoại": 70,
        "Sửa chữa và bảo dưỡng TV": 110,
        "Sửa chữa và bảo dưỡng điện dân dụng": 100,
        "Sửa chữa và bảo dưỡng hệ thống nước": 120,
        "Sửa chữa và bảo dưỡng xe máy": 90,
        "Sửa chữa và bảo dưỡng ô tô": 200,
        "Sửa chữa và bảo dưỡng máy in": 85
    }

    # =========================
    # 🌧️ BALANCED RAIN (33% - 33% - 33%)
    # =========================
    rain_labels = (
        [0.0] * (n_samples // 3) +
        [0.5] * (n_samples // 3) +
        [1.0] * (n_samples - 2 * (n_samples // 3))
    )
    np.random.shuffle(rain_labels)

    # =========================
    # 👷 BALANCED EXPERIENCE (33% - 33% - 33%)
    # =========================
    exp_labels = (
        list(np.random.uniform(0, 3, n_samples // 3)) +
        list(np.random.uniform(3, 8, n_samples // 3)) +
        list(np.random.uniform(8, 30, n_samples - 2 * (n_samples // 3)))
    )
    np.random.shuffle(exp_labels)

    data = []

    for i in range(n_samples):

        service = np.random.choice(services)
        base_time = service_base[service]

        distance = np.random.choice(
            np.arange(1, 51),
            p=np.linspace(50, 1, 50) / np.linspace(50, 1, 50).sum()
        )

        experience = exp_labels[i]

        # =========================
        # EXPERIENCE EFFECT
        # =========================
        if experience < 1:
            exp_reduction = 5
        elif experience < 3:
            exp_reduction = 15
        elif experience < 8:
            exp_reduction = 25
        else:
            exp_reduction = 40

        hour = np.random.randint(0, 24)
        is_peak_hour = 1 if (7 <= hour <= 9 or 14 <= hour <= 19) else 0

        noise = np.random.normal(0, base_time * 0.08)

        # =========================
        # RAIN FEATURE
        # =========================
        rain_ratio = rain_labels[i]

        # =========================
        # TIME FORMULA
        # =========================
        time = (
            base_time
            + distance * 2.1
            - exp_reduction
            + is_peak_hour * 10
            + rain_ratio * 15
            + noise
        )

        # Outlier
        if np.random.rand() < outlier_ratio:
            if np.random.rand() < 0.5:
                time *= np.random.uniform(1.5, 2.5)
            else:
                time *= np.random.uniform(0.5, 0.8)

        # rounding
        if time >= 5:
            time = math.ceil(time)
        else:
            time = int(time)

        data.append([
            service,
            distance,
            experience,
            hour,
            is_peak_hour,
            rain_ratio,
            time
        ])

    df = pd.DataFrame(data, columns=[
        "service",
        "distance",
        "experience",
        "hour",
        "is_peak_hour",
        "rain_ratio",
        "completion_time"
    ])

    return df


# =========================
# GENERATE DATA
# =========================
df = generate_repair_dataset(n_samples=30000, outlier_ratio=0.01)

# SAVE
df.to_csv(r"E:\FixAI\data\output\datatest.csv", index=False)

print("Dataset generated:", df.shape)
print(df.head())

