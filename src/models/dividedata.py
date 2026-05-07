import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("E:/FixAI/data/output/datatest.csv")

df = df.drop(columns=["hour"])

train_df, test_df = train_test_split(
    df,
    test_size=0.3,      # 30% test
    random_state=42,
    shuffle=True
)


print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# 4. Save ra 2 file CSV
train_df.to_csv("E:/FixAI/data/processed/train.csv", index=False)
test_df.to_csv("E:/FixAI/data/processed/test.csv", index=False)

print("✅ Saved train.csv & test.csv")