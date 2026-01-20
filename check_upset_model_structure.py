"""UPSET分類器の構造を確認"""
import pickle
from pathlib import Path

model_file = Path("walk_forward_results_custom2/period_10/models/2025/upset_classifier_2015-2024.sav")

with open(model_file, 'rb') as f:
    data = pickle.load(f)

print(f"型: {type(data)}")
print(f"\n内容:")

if isinstance(data, dict):
    print(f"  キー: {list(data.keys())}")
    for key, value in data.items():
        print(f"\n  [{key}]")
        print(f"    型: {type(value)}")
        if isinstance(value, list):
            print(f"    要素数: {len(value)}")
            if len(value) > 0:
                print(f"    最初の要素: {type(value[0])}")
elif isinstance(data, list):
    print(f"  要素数: {len(data)}")
    if len(data) > 0:
        print(f"  最初の要素の型: {type(data[0])}")
        if hasattr(data[0], '__dict__'):
            print(f"  属性: {list(data[0].__dict__.keys())[:10]}")
else:
    print(f"  属性: {dir(data)[:20]}")
