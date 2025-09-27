from ultralytics import YOLO
import pandas as pd

# 학습된 모델 경로
model_path = r"C:\git\task3\cars_detection\runs\detect\train14\weights\best.pt"
model = YOLO(model_path)

# 모델 평가
metrics = model.val()  # DetMetrics 객체

# 클래스별 summary 가져오기
summary_list = metrics.summary()  # list of dicts

# DataFrame으로 변환
df = pd.DataFrame(summary_list)

# 표 출력
print(df)

import matplotlib.pyplot as plt
import numpy as np

# 클래스명
classes = df['Class'].tolist()

# 각 지표
precision = df['Box-P'].tolist()
recall = df['Box-R'].tolist()
map50 = df['mAP50'].tolist()
map5095 = df['mAP50-95'].tolist()

x = np.arange(len(classes))
width = 0.2  # 막대 너비

fig, ax = plt.subplots(figsize=(12,6))
ax.bar(x - 1.5*width, precision, width, label='Precision')
ax.bar(x - 0.5*width, recall, width, label='Recall')
ax.bar(x + 0.5*width, map50, width, label='mAP50')
ax.bar(x + 1.5*width, map5095, width, label='mAP50-95')

ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel('Score')
ax.set_ylim(0,1)
ax.set_title('Class-wise Detection Performance')
ax.legend()

# 값 표시
for i in range(len(classes)):
    ax.text(i - 1.5*width, precision[i]+0.02, f"{precision[i]:.2f}", ha='center')
    ax.text(i - 0.5*width, recall[i]+0.02, f"{recall[i]:.2f}", ha='center')
    ax.text(i + 0.5*width, map50[i]+0.02, f"{map50[i]:.2f}", ha='center')
    ax.text(i + 1.5*width, map5095[i]+0.02, f"{map5095[i]:.2f}", ha='center')

plt.show()
