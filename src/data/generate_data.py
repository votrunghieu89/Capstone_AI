import numpy as np
import pandas as pd
import math

def generate_repair_dataset(n_samples=10000, outlier_ratio=0.03, random_seed=42):
    np.random.seed(random_seed)
    services = [
        "Sửa chửa thiết bị gia dụng",
        "Sửa máy tính",
        "Sửa xe ô tô",
        "Sửa xe máy",
        "Sửa điện thoại",
        "Sửa điện nước dân dụng",
        "Sửa thiết bị văn phòng"
    ]
    
    service_base = {
        "Sửa chửa thiết bị gia dụng": 120,
        "Sửa máy tính": 60,
        "Sửa xe ô tô": 180,
        "Sửa xe máy": 120,
        "Sửa điện thoại": 40,
        "Sửa điện nước dân dụng": 80,
        "Sửa thiết bị văn phòng": 120
    }
    data = []

    for _ in range(n_samples):
        
        service = np.random.choice(services)
        base_time = service_base[service]
        
        distance = int(np.random.uniform(0, 150))

        
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
        noise = np.random.normal(0, 2.5)

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
df = generate_repair_dataset(n_samples=50000, outlier_ratio=0.01)

# Save CSV
df.to_csv(r"E:\FixAI\data\output\datatest.csv", index=False)

print("Dataset generated:", df.shape)
print(df)