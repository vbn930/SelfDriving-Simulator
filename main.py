import pandas as pd
import cv2
import os

# 1. 경로 설정 (수집된 세션 폴더 경로 입력)
data_path = r'D:\robotics\llm-driven-robotics\brain\data\TrainingData\Session_20260213_154559'
csv_path = os.path.join(data_path, 'log.csv')

# 2. CSV 로드
df = pd.read_csv(csv_path)

for i in range(len(df)):
    img_name = df.iloc[i]['Image']
    steering = df.iloc[i]['Steering']
    throttle = df.iloc[i]['Throttle']
    
    # 이미지 읽기
    img = cv2.imread(os.path.join(data_path, img_name))
    
    if img is None: continue

    # 이미지에 조향 값 표시 (텍스트)
    cv2.putText(img, f"Steer: {steering:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 조향 방향 시각화 (중앙 하단에 선 그리기)
    # -1(좌)이면 왼쪽으로, 1(우)이면 오른쪽으로 선이 기움
    start_point = (img.shape[1]//2, img.shape[0]-10)
    end_point = (int(img.shape[1]//2 + steering * 50), img.shape[0]-50)
    cv2.line(img, start_point, end_point, (0, 0, 255), 5)

    cv2.imshow('Data Verification', img)
    
    # 아무 키나 누르면 다음 이미지, 'q' 누르면 종료
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
