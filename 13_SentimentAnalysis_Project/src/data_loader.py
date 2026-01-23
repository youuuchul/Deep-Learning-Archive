# ---------------------------------------------------------
# 3. 데이터 로드 및 경로 탐색 (최적화됨)
# ---------------------------------------------------------
from src.data_loader import load_json_files, parse_data, split_data_leakage_proof, create_dataset_dict
import unicodedata

def find_data_dir():
    # 1. 가장 확실한 경로(Fallback Path)부터 먼저 확인
    # (이미 통합 셋업에서 codeit 폴더로 이동했으므로 ./data/... 가 가장 유력)
    primary_target = Path("./data/쇼핑몰/02. 화장품")
    if primary_target.exists():
        logger.info(f"✅ 표준 데이터 경로 발견: {primary_target}")
        return primary_target

    # 2. 없으면 자동 탐색 시도
    candidates = [Path("./data"), Path("../data")]
    for cand in candidates:
        if cand.exists():
            try:
                # 쇼핑몰 폴더 찾기
                shopping_dir = next(cand.glob("*쇼핑몰*"))
                target_categories = ["화장품", "가전", "IT기기", "패션"]
                for category in target_categories:
                    for p in shopping_dir.glob("*"):
                        if category in unicodedata.normalize('NFC', p.name):
                            logger.info(f"🔍 자동 탐색된 카테고리: {p.name}")
                            return p
            except StopIteration:
                continue
    return None

# --- 실행부 ---
DATA_DIR = find_data_dir()

if DATA_DIR is None:
    raise FileNotFoundError("데이터 폴더를 찾을 수 없습니다. ./data 폴더 구조를 확인해주세요.")

logger.info(f"📂 최종 사용 경로: {DATA_DIR}")

# 샘플링 제한 (로컬일 때만)
SAMPLE_LIMIT = 2000 if is_mac_local else None

try:
    logger.info("🚀 데이터 로드 파이프라인 시작...")
    
    # (1) 파일 리스트
    files = load_json_files(DATA_DIR)
    
    # (2) 파싱 (NoneType 에러 수정된 버전)
    df = parse_data(files, sample_limit=SAMPLE_LIMIT)
    
    # (3) 분할
    train_df, test_df = split_data_leakage_proof(df)
    
    # (4) 데이터셋 생성
    dataset = create_dataset_dict(train_df, test_df)

    print("\n✅ 데이터 로드 성공!")
    print(dataset)

except Exception as e:
    logger.error(f"❌ 에러 발생: {e}")
    raise e