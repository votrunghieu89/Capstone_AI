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

    data = []

    for _ in range(n_samples):
        
        service = np.random.choice(services)
        base_time = service_base[service]
        
        distance = np.random.choice(
                        np.arange(1, 51),
                        p=np.linspace(50, 1, 50) / np.linspace(50, 1, 50).sum()
                    )

        
        experience = int(np.random.choice(
            np.concatenate([
                np.random.uniform(2, 10, 60),
                np.random.uniform(0, 30, 40)
            ])
        ))

        # 3. Hour
        hour = np.random.randint(0, 24)
        is_peak_hour = 1 if (7 <= hour <= 9 or 14 <= hour <= 19) else 0
        
        # 4. Noise
        noise = np.random.normal(0, base_time * 0.08)

        # 5. Formula
        time = (
            base_time
            + distance * 2.1
            - experience * 0.6
            + is_peak_hour * 15
            + (1 / (experience + 1)) * 15
            + noise
        )

        # 6. Outlier
        if np.random.rand() < outlier_ratio:
            if np.random.rand() < 0.5:
                time *= np.random.uniform(1.5, 2.5)
            else:
                time *= np.random.uniform(0.5, 0.8)

        
        if time >= 5:
            time = math.ceil(time)   # làm tròn lên
        else:
            time = int(time)         # bỏ phần thập phân

        data.append([service, distance, experience, hour, is_peak_hour, time])

    df = pd.DataFrame(data, columns=[
        "service",
        "distance",
        "experience",
        "hour",
        "is_peak_hour",
        "completion_time"
    ])

    return df


# Generate data
df = generate_repair_dataset(n_samples=60000, outlier_ratio=0.01)

# Save CSV
df.to_csv(r"E:\FixAI\data\output\datatest.csv", index=False)

print("Dataset generated:", df.shape)
print(df)