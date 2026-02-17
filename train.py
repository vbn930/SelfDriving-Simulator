import os
from matplotlib import image
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
BASE_PATH = r'D:\robotics\llm-driven-robotics\brain\data\TrainingData\Merged_DataSet'
CSV_FILE = 'log.csv'

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ==========================================
# 2. 데이터셋 정의 (Custom Dataset)
# ==========================================
class DriveDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.images = self.data['Image'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 데이터 로드
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        current_speed = self.data['Speed'].values[idx]
        steering = self.data['Steering'].values[idx]
        throttle = self.data['Throttle'].values[idx]

        # 2. 데이터 증강: 50% 확률로 좌우 반전
        if np.random.rand() < 0.5:
            # 이미지 좌우 반전 (1은 좌우, 0은 상하)
            image = cv2.flip(image, 1)
            # 조향값 반전
            steering = -steering

        # 3. 전처리 및 정규화
        image = cv2.resize(image, (200, 66)) 
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        speed_tensor = torch.tensor([current_speed / 100.0], dtype=torch.float32)
        label = torch.tensor([steering, throttle], dtype=torch.float32)
        
        return image, speed_tensor, label

# ==========================================
# 3. 모델 정의 (NVIDIA Dave2 Architecture)
# ==========================================
class Dave2(nn.Module):
    def __init__(self):
        super(Dave2, self).__init__()
        # 1. 이미지 처리를 위한 CNN 브랜치
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU()
        )
        
        # 2. 속도 데이터를 처리하기 위한 MLP 브랜치
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ELU()
        )

        # 3. 결합 후 최종 제어값을 내뱉는 FC 레이어
        # CNN 출력(1152) + 속도 특징(16) = 1168
        self.fc_layers = nn.Sequential(
            nn.Linear(1168, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 2) # [Steering, Throttle]
        )

    def forward(self, img, speed):
        # 이미지 특징 추출
        img_feat = self.conv_layers(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        
        # 속도 특징 추출
        speed_feat = self.speed_fc(speed)
        
        # 특징 결합 (Concatenation)
        combined = torch.cat((img_feat, speed_feat), dim=1)
        
        # 최종 예측
        output = self.fc_layers(combined)
        return output

# ==========================================
# 4. 학습 실행 (Training Loop)
# ==========================================
def main():
    # 데이터 로드
    full_dataset = DriveDataset(CSV_FILE, BASE_PATH)
    
    # 학습용/검증용 분리 (8:2)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 초기화
    model = Dave2().to(DEVICE)
    
    # 손실 함수 & 최적화 기법
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("학습 시작...")
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, speeds, labels in train_loader:
            images = images.to(DEVICE)
            speeds = speeds.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 3. 모델에 이미지와 속도를 함께 입력
            outputs = model(images, speeds) 
            
            # 4. 손실 계산
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, speeds, labels in val_loader:
                images = images.to(DEVICE)
                speeds = speeds.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images, speeds)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 모델 저장
    torch.save(model.state_dict(), "best_model.pth")
    print("모델 저장 완료: best_model.pth")

    # 손실 그래프
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()