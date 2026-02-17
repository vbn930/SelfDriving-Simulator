import os
import pandas as pd
import shutil
from tqdm import tqdm

# ==========================================
# 1. 경로 설정
# ==========================================
SOURCE_DIR = r'D:\robotics\llm-driven-robotics\brain\data\TrainingData' 
MERGED_DIR = r'D:\robotics\llm-driven-robotics\brain\data\TrainingData\Merged_DataSet'
LOG_FILENAME = 'log.csv' 

def merge_data():
    if not os.path.exists(MERGED_DIR):
        os.makedirs(MERGED_DIR)
        print(f"통합 폴더 생성 완료: {MERGED_DIR}")

    all_logs = []
    
    # SOURCE_DIR 내의 모든 세션 폴더 리스트업
    session_folders = [f for f in os.listdir(SOURCE_DIR) 
                       if os.path.isdir(os.path.join(SOURCE_DIR, f)) and f != 'Merged_DataSet']

    print(f"🚀 총 {len(session_folders)}개의 세션 폴더 통합 시작...")

    for session in session_folders:
        session_path = os.path.join(SOURCE_DIR, session)
        csv_path = os.path.join(session_path, LOG_FILENAME)

        if not os.path.exists(csv_path):
            continue

        # 로그 파일 로드
        df = pd.read_csv(csv_path)
        
        # tqdm으로 복사 진행 상황 표시
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Session: {session}"):
            img_name = row['Image']
            src_img_path = os.path.join(session_path, img_name)
            dst_img_path = os.path.join(MERGED_DIR, img_name)

            # 파일 이름이 겹치지 않으므로 바로 복사
            if os.path.exists(src_img_path):
                # [RTIS Tip] 성능을 위해 단순 copy2 사용
                shutil.copy2(src_img_path, dst_img_path) 
                all_logs.append(row)

    # 전체 로그 통합 및 저장
    merged_df = pd.DataFrame(all_logs)
    merged_df.to_csv(os.path.join(MERGED_DIR, 'log.csv'), index=False)
    
    print("\n데이터 통합 작업이 완료되었습니다.")
    print(f"최종 데이터셋 규모: {len(merged_df)} frames")

if __name__ == "__main__":
    merge_data()