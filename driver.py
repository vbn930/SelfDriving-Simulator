import socket
import struct
import cv2
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. 모델 정의 (train.py와 동일해야 함)
# ==========================================

from train import Dave2

# ==========================================
# 2. 설정 및 모델 로드
# ==========================================
HOST = '127.0.0.1' # 로컬호스트
PORT = 9999        # 포트 번호
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Dave2().to(DEVICE)
model.load_state_dict(torch.load("models/best_model_v3.pth", map_location=DEVICE))
model.eval() # 평가 모드 (Dropout 끄기)
print("모델 로드 완료. 유니티 연결 대기 중...")

# ==========================================
# 3. 소켓 서버 실행
# ==========================================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

conn, addr = server_socket.accept()
print(f"유니티 연결됨: {addr}")

try:
    while True:
        # 1. 헤더 수신 (이미지 크기 - 4바이트)
        header = conn.recv(4)
        if not header: break
        img_len = struct.unpack('<i', header)[0] # Little-endian int

        # 2. 이미지 데이터 수신 (나눠서 올 수 있으므로 루프 처리)
        img_data = b''
        while len(img_data) < img_len:
            chunk = conn.recv(img_len - len(img_data))
            if not chunk: break
            img_data += chunk
            
        # 3. 속도 데이터 수신 (4바이트)
        speed_footer = conn.recv(4)
        if not speed_footer: break
        current_speed = struct.unpack('<f', speed_footer)[0] # Little-endian float

        # 4. 이미지 디코딩 및 전처리
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200, 66)) / 255.0
        img_tensor = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 속도 정규화 (학습 시와 동일하게 100으로 나눔)
        speed_tensor = torch.tensor([[current_speed / 100.0]], dtype=torch.float32).to(DEVICE)

        # 5. 추론 (Feature Fusion 입력)
        with torch.no_grad():
            prediction = model(img_tensor, speed_tensor).cpu().numpy()[0]
            steering, throttle = prediction[0], prediction[1]
            
        # 6. 결과 전송
        response = f"{steering:.4f},{throttle:.4f}\n".encode()
        conn.send(response)
        
        print(f"예측 결과 - Steering: {steering:.4f}, Throttle: {throttle:.4f}, Current Speed: {current_speed:.2f} km/h")

except Exception as e:
    print(f"에러 발생: {e}")
finally:
    conn.close()
    server_socket.close()