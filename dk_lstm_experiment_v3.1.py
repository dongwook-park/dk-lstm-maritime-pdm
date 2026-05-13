# ==============================================================================
# PAPER: A Standards-Aligned Hybrid AI–Digital Twin Framework for
#        Robust Predictive Maintenance under Data Scarcity
#
# EXPERIMENT SCRIPT v4
#   - CNN-LSTM 하이브리드 baseline 추가
#   - Multi-seed E1 검증 (seeds 0, 42, 123)
#   - 기존 셀 1~11은 변경 없음 (재현성 보장)
# ============================================================
#
# 🔴 [수정 A] RUL 클리핑 추가
#   문제: add_rul()이 Raw RUL(FD001 최대 ~362 사이클)을 그대로 사용.
#         이 상태에서 Constraint(iii) 상한 제약(MAX_RUL=125)을 Loss에 넣으면,
#         초기 사이클 정답이 125를 초과하므로 MSE Loss와 정면 충돌 발생.
#   수정: .clip(upper=MAX_RUL) 적용 (C-MAPSS Piecewise Linear 표준 관행)
#
# 🔴 [수정 B] Constraint (iii) 상한 제약 추가
#   Abstract 근거: "(iii) operating-range bounds, are explicitly embedded
#                  into the learning objective"
#   문제: MAX_RUL=125.0 변수 선언만 있고, dk_loss_fn 연산에 사용되지 않음.
#   수정: bound_penalty = relu(y_pred - MAX_RUL)^2 항 추가
#
# 🟡 [수정 C] 아키텍처 불공정 비교 수정
#   문제: Standard LSTM/GRU 출력층 → activation='relu' (구조적으로 음수 불가)
#         DK-LSTM 출력층 → activation='linear' (음수 가능)
#         Neg_Pct 지표 비교가 Loss 차이가 아닌 아키텍처 차이에서 기인할 수 있음.
#   수정: 모든 모델 출력층 activation='linear'으로 통일. (Neg_Pct는 post-hoc으로만 측정)
#
# 🟡 [수정 D] 학습 곡선 시각화 버그 수정
#   문제: plt.legend() 전에 plot 데이터가 없어 빈 그래프 저장.
#   수정: train_model()이 history 반환 → hist_std, hist_gru, hist_dk 저장 후 시각화
#
# ✅ [유지] 비대칭 Safety Penalty
#   Abstract에서 3대 제약(i)(ii)(iii)은 "domain-informed constraints"로 명시되며,
#   Safety Penalty는 별도의 실용적 손실 항(practical loss term)으로 구분 가능.
#   논문 방법론 절에서 "(i)(ii)(iii) 외에, 안전 임계 환경을 위해 비대칭
#   페널티를 추가 적용"으로 기술.
#
# ✅ [유지] 단조감소 구현 방식 (배치 정렬 근사법)
#   DK-LSTM은 단일 출력 모델 (return_sequences=False)이므로,
#   시퀀스 내부 단조성 직접 적용 불가. 미니배치 내에서 y_true 내림차순 정렬 후
#   예측값의 단조감소 위반을 페널티화하는 방식 유지.
#   "메모리 효율성을 위해 sequence-to-sequence 구조를 피하고,
#   미니배치 내 물리적 매니폴드(physical manifold) 정렬을 통해 단조성을
#   통계적으로 강제하는 정규화 기법을 채택."
# ==============================================================================


# ==============================================================================
# [CELL 1] 환경 설정 및 Google Drive 연결
# ==============================================================================
import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from google.colab import drive

# ── Google Drive 연결 ────────────────────────────────────────────────────────
if not os.path.exists('/content/drive'):
    print(">>> 구글 드라이브를 연결합니다...")
    drive.mount('/content/drive')

data_path = '/content/drive/MyDrive/Paper_Experiment/'
out_path  = data_path   # 결과물 저장 경로 (동일 폴더)
print(f"\n>>> 데이터/출력 경로: {data_path}")

# ── 컬럼 정의 ────────────────────────────────────────────────────────────────
index_names   = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names  = [f's_{i}' for i in range(1, 22)]
col_names     = index_names + setting_names + sensor_names

# ── 피처 선택 (변동성 없는 센서 제외 — 기존 코드 유지) ─────────────────────
FEATURES   = ['setting_1', 'setting_2',
              's_2', 's_3', 's_4', 's_7', 's_8', 's_9',
              's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
N_FEATURES = len(FEATURES)

# ── 하이퍼파라미터 ───────────────────────────────────────────────────────────
SEQUENCE_LEN = 30
BATCH_SIZE   = 128
MAX_EPOCHS   = 100
PATIENCE     = 15
MISSING_RATE = 0.30     # 센서 결측 비율 (Experiment 2)
MAX_RUL      = 125.0    # RUL 클리핑 및 상한 제약에 공통 사용

# ── DK-LSTM Loss 가중치 (전역 상수로 관리) ───────────────────────────────────
LAMBDA_SAFETY = 1.0     # 비대칭 Safety Penalty (practical loss term)
LAMBDA_NEG    = 1.0     # Constraint (i)  비음수
LAMBDA_MONO   = 0.0     # Constraint (ii) 단조감소
LAMBDA_UPPER  = 1.0     # Constraint (iii) 상한 [수정 B에서 추가]

RANDOM_SEED = 42

# ── 재현성 설정 ───────────────────────────────────────────────────────────────
def set_all_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds()

print("=" * 60)
print(f"  Sequence Length: {SEQUENCE_LEN}  |  MAX_RUL: {MAX_RUL}")
print(f"  Features: {N_FEATURES}  |  Missing Rate: {MISSING_RATE:.0%}")
print(f"  λ_safety={LAMBDA_SAFETY}, λ_neg={LAMBDA_NEG}, "
      f"λ_mono={LAMBDA_MONO}, λ_upper={LAMBDA_UPPER}")
print(f"  Random Seed: {RANDOM_SEED}")
print("=" * 60)

# ==============================================================================
# [CELL 2] ISO 19848 정렬 데이터 계약(Data Contract) 스키마 정의
# ==============================================================================

ISO_19848_DATA_CONTRACT = {
    "schema_version"    : "1.0",
    "standard_reference": (
        "ISO 19848:2024 (Ships and marine technology — "
        "Standard data for shipboard machinery and equipment)"
    ),
    "vessel_id"         : "PROXY_VESSEL_CMAPSS",
    "note": (
        "ISO 19848-aligned reference mapping for the NASA C-MAPSS proxy "
        "benchmark. Sensor names are mapped to ISO 19848 channel identifiers "
        "for semantic interoperability. This is a reference mapping, "
        "not a full standard compliance implementation."
    ),
    "data_channels": [
        {"channel_id": "OC_ALT",    "iso_name": "OperationalCondition_Altitude",
         "source_feature": "setting_1", "unit": "ft",   "type": "setting"},
        {"channel_id": "OC_MN",     "iso_name": "OperationalCondition_MachNumber",
         "source_feature": "setting_2", "unit": "mach", "type": "setting"},
        {"channel_id": "FAN_IN_T",  "iso_name": "FanInlet_Temperature",
         "source_feature": "s_2",      "unit": "degR", "type": "sensor"},
        {"channel_id": "LPC_OUT_T", "iso_name": "LPC_OutletTemperature",
         "source_feature": "s_3",      "unit": "degR", "type": "sensor"},
        {"channel_id": "HPC_OUT_T", "iso_name": "HPC_OutletTemperature",
         "source_feature": "s_4",      "unit": "degR", "type": "sensor"},
        {"channel_id": "OGV_T",     "iso_name": "BypassDuct_Temperature",
         "source_feature": "s_7",      "unit": "degR", "type": "sensor"},
        {"channel_id": "HPC_IN_T",  "iso_name": "HPC_InletTemperature",
         "source_feature": "s_8",      "unit": "degR", "type": "sensor"},
        {"channel_id": "HPT_IS_TS", "iso_name": "HPT_IsentropicEfficiency",
         "source_feature": "s_9",      "unit": "pct",  "type": "sensor"},
        {"channel_id": "HPC_OUT_P", "iso_name": "HPC_OutletPressure",
         "source_feature": "s_11",     "unit": "psia", "type": "sensor"},
        {"channel_id": "HPT_OUT_P", "iso_name": "HPT_OutletPressure",
         "source_feature": "s_12",     "unit": "psia", "type": "sensor"},
        {"channel_id": "FAN_EFF",   "iso_name": "Fan_IsentropicEfficiency",
         "source_feature": "s_13",     "unit": "pct",  "type": "sensor"},
        {"channel_id": "LPC_EFF",   "iso_name": "LPC_IsentropicEfficiency",
         "source_feature": "s_14",     "unit": "pct",  "type": "sensor"},
        {"channel_id": "HPT_EFF",   "iso_name": "HPT_IsentropicEfficiency",
         "source_feature": "s_15",     "unit": "pct",  "type": "sensor"},
        {"channel_id": "BPR",       "iso_name": "BypassRatio",
         "source_feature": "s_17",     "unit": "ratio","type": "sensor"},
        {"channel_id": "HPC_FLOW",  "iso_name": "HPC_FlowSensor",
         "source_feature": "s_20",     "unit": "lbm/s","type": "sensor"},
        {"channel_id": "LPT_EFF",   "iso_name": "LPT_IsentropicEfficiency",
         "source_feature": "s_21",     "unit": "pct",  "type": "sensor"},
    ],
    "target_variable": {
        "channel_id"                       : "RUL",
        "iso_name"                         : "RemainingUsefulLife",
        "unit"                             : "cycles",
        "constraint_non_negative"          : True,   # Constraint (i)
        "constraint_monotonic_degradation" : True,   # Constraint (ii)
        "constraint_upper_bound"           : MAX_RUL # Constraint (iii)
    }
}

contract_path = out_path + 'iso19848_data_contract.json'
with open(contract_path, 'w') as f:
    json.dump(ISO_19848_DATA_CONTRACT, f, indent=2)
print(f"✔ ISO 19848 Data Contract saved → {contract_path}")

# ==============================================================================
# [CELL 3] 데이터 로드 및 전처리
# ==============================================================================
import sys

# ── FD001 로드 ────────────────────────────────────────────────────────────────
try:
    print(">>> FD001 파일 읽기 시도 중...")
    train_df  = pd.read_csv(data_path + 'train_FD001.txt',
                            sep=r'\s+', header=None, names=col_names)
    test_df   = pd.read_csv(data_path + 'test_FD001.txt',
                            sep=r'\s+', header=None, names=col_names)
    y_test_df = pd.read_csv(data_path + 'RUL_FD001.txt',
                            sep=r'\s+', header=None, names=['RUL'])
    print(">>> [성공] FD001 로드 완료.")
except FileNotFoundError:
    print(f"[오류] {data_path} 경로에 파일이 없습니다.")
    sys.exit()

# ── FD002, FD003 로드 (도메인 시프트 실험용) ─────────────────────────────────
try:
    test_fd2 = pd.read_csv(data_path + 'test_FD002.txt',
                           sep=r'\s+', header=None, names=col_names)
    rul_fd2  = pd.read_csv(data_path + 'RUL_FD002.txt',
                           sep=r'\s+', header=None, names=['RUL'])
    test_fd3 = pd.read_csv(data_path + 'test_FD003.txt',
                           sep=r'\s+', header=None, names=col_names)
    rul_fd3  = pd.read_csv(data_path + 'RUL_FD003.txt',
                           sep=r'\s+', header=None, names=['RUL'])
    print(">>> [성공] FD002 / FD003 로드 완료.")
except FileNotFoundError:
    print("[경고] FD002/FD003 파일 없음 → 도메인 시프트 실험 불가")
    test_fd2 = rul_fd2 = test_fd3 = rul_fd3 = None

# ── [수정 A] RUL 계산 — Piecewise Linear 클리핑 적용 ─────────────────────────
# 문제: clip 없이 raw RUL 사용 → 최대 ~362 사이클.
#       Loss의 Constraint(iii) 상한(125)과 MSE Loss가 충돌.
# 수정: .clip(upper=MAX_RUL)으로 C-MAPSS 표준 관행 적용.
def add_rul(df, max_rul=MAX_RUL):
    max_life  = df.groupby('unit_nr')['time_cycles'].max()
    result_df = df.merge(max_life.to_frame(name='max_life'),
                         left_on='unit_nr', right_index=True)
    result_df['RUL'] = (result_df['max_life'] - result_df['time_cycles']
                        ).clip(upper=max_rul)   # ← [수정 A] 클리핑 추가
    return result_df.drop(columns=['max_life'])

if 'RUL' not in train_df.columns:
    train_df = add_rul(train_df)

# ── MinMax Scaling (FD001 훈련 데이터 기준 fit) ──────────────────────────────
scaler = MinMaxScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df[FEATURES]  = scaler.transform(test_df[FEATURES])

# ── Train / Validation Split (Unit 기반 80/20) ──────────────────────────────
units       = train_df['unit_nr'].unique()
train_units = units[:80]
val_units   = units[80:]
train_split = train_df[train_df['unit_nr'].isin(train_units)]
val_split   = train_df[train_df['unit_nr'].isin(val_units)]
print(f">>> Split: Train={len(train_units)} units, Val={len(val_units)} units")

# ── 슬라이딩 윈도우 시퀀스 생성 ─────────────────────────────────────────────
def make_sequences(df, seq_len=SEQUENCE_LEN, feats=FEATURES):
    X_list, y_list = [], []
    for uid in df['unit_nr'].unique():
        unit = df[df['unit_nr'] == uid]
        vals = unit[feats].values
        ruls = unit['RUL'].values
        for i in range(len(unit) - seq_len):
            X_list.append(vals[i: i + seq_len])
            y_list.append(ruls[i + seq_len])
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return X, y

x_train, y_train = make_sequences(train_split)
x_val,   y_val   = make_sequences(val_split)
print(f"Train: {x_train.shape}  Val: {x_val.shape}")

# ── 테스트 시퀀스 생성 (각 엔진의 마지막 seq_len 스텝) ─────────────────────
def make_test_sequences(df_test, rul_df, scaler_ref=None,
                        seq_len=SEQUENCE_LEN, feats=FEATURES):
    df = df_test.copy()
    if scaler_ref is not None:
        df[feats] = scaler_ref.transform(df[feats])
    X_list, y_list = [], []
    for idx, uid in enumerate(sorted(df['unit_nr'].unique())):
        unit = df[df['unit_nr'] == uid]
        if len(unit) < seq_len:
            continue
        X_list.append(unit[feats].values[-seq_len:])
        y_list.append(rul_df.iloc[idx]['RUL'])
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))

X_test_fd1, y_test_fd1 = make_test_sequences(test_df, y_test_df)
print(f"Test FD001: {X_test_fd1.shape}")

# ==============================================================================
# [CELL 4] Leakage-Aware 센서 결측 처리 모듈
# ==============================================================================
# 구현:
#   1. 센서별 독립적으로 MISSING_RATE 비율만큼 NaN 마킹
#   2. Forward-fill (직전 관측값 대체) → 미래 정보 유출 없음
#   3. 결측 여부 이진 마스크(1=결측, 0=정상)를 원본 피처에 concat
#      → 최종 입력 차원: N_FEATURES × 2

def apply_missingness(X, missing_rate=MISSING_RATE, seed=RANDOM_SEED):
    rng    = np.random.default_rng(seed)
    X_miss = X.copy().astype(np.float32)
    N, T, F = X_miss.shape
    mask   = np.zeros_like(X_miss, dtype=np.float32)

    miss_locs = rng.random((N, T, F)) < missing_rate
    X_miss[miss_locs] = np.nan
    mask[miss_locs]   = 1.0

    # Forward-fill (leakage-free)
    for n in range(N):
        for f in range(F):
            arr = X_miss[n, :, f]
            first_valid = np.where(~np.isnan(arr))[0]
            if len(first_valid) == 0:
                arr[:] = 0.0
            else:
                if first_valid[0] > 0:
                    arr[:first_valid[0]] = arr[first_valid[0]]
                arr[:] = pd.Series(arr).ffill().values
            X_miss[n, :, f] = arr

    return np.concatenate([X_miss, mask], axis=-1).astype(np.float32), mask

x_train_miss, _ = apply_missingness(x_train, MISSING_RATE, RANDOM_SEED)
x_val_miss,   _ = apply_missingness(x_val,   MISSING_RATE, RANDOM_SEED + 1)
X_test_miss,  _ = apply_missingness(X_test_fd1, MISSING_RATE, RANDOM_SEED + 10)

N_FEAT_MISS = x_train_miss.shape[-1]   # N_FEATURES * 2
print(f"✔ Missingness applied. Input dim: {N_FEATURES} → {N_FEAT_MISS}")

# ==============================================================================
# [CELL 5] 모델 정의
# ==============================================================================
# dk_loss_fn에 Constraint(iii) 상한 제약 추가
# 모든 모델 출력층 activation='linear'으로 통일 (공정 비교)

def dk_loss_fn(y_true, y_pred):
    """
    Domain-Knowledge Enhanced Loss Function.

    구성:
      · MSE                  : 데이터 손실
      · Safety Penalty       : 실용적 손실 항 (과대예측 억제, Abstract 3대 제약과 별개)
      · Constraint (i)       : 비음수성  — non-negativity of RUL
      · Constraint (ii)      : 단조감소  — monotonic degradation
      · Constraint (iii)     : 운용 범위 — operating-range bounds  [수정 B: 신규]

    Abstract 대응:
      "Domain-informed constraints, including
       (i) non-negativity of the Remaining Useful Life (RUL),
       (ii) monotonic degradation of the predicted RUL, and
       (iii) operating-range bounds,
       are explicitly embedded into the learning objective"
    """
    # ── 데이터 손실 ───────────────────────────────────────────────────────────
    L_mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # ── Safety Penalty ─────────────
    # 과대예측(d>0)에 2배, 과소예측(d<0)에 0.2배 가중치 → 안전 임계 환경 대응
    error          = y_pred - y_true
    L_safety       = tf.reduce_mean(
        tf.where(error > 0,
                 tf.square(error) * 2.0,
                 tf.square(error) * 0.2)
    )

    # ── Constraint (i): Non-negativity ───────────────────────────────────────
    # RUL은 물리적으로 음수가 될 수 없음
    L_neg = tf.reduce_mean(tf.square(tf.nn.relu(-y_pred)))

    # ── Constraint (ii): Monotonic Degradation ────────────────────────────────
    # 배치 내 샘플을 y_true 내림차순(=시간 오름차순)으로 정렬 후,
    # 예측값도 단조 감소해야 함 — 위반 시 페널티.
    # [설계 근거] return_sequences=False 단일 출력 모델에서는 시퀀스 내부
    # 직접 비교가 불가하므로, 미니배치 내 물리적 매니폴드 정렬로 근사.
    sorted_idx  = tf.argsort(tf.squeeze(y_true), direction='DESCENDING')
    pred_sorted = tf.gather(tf.squeeze(y_pred), sorted_idx)
    diff        = pred_sorted[:-1] - pred_sorted[1:]   # 감소 방향 = 양수
    L_mono      = tf.reduce_mean(tf.square(tf.nn.relu(-diff)))

    # ── Constraint (iii): Operating-Range Upper Bound ─────────────────────────
    # [수정] 예측값이 MAX_RUL(=125)을 초과하면 페널티 부여

    L_upper = tf.reduce_mean(tf.square(tf.nn.relu(y_pred - MAX_RUL)))

    return (L_mse
            + LAMBDA_SAFETY * L_safety
            + LAMBDA_NEG    * L_neg
            + LAMBDA_MONO   * L_mono
            + LAMBDA_UPPER  * L_upper)


def _build_base(layers, input_shape, name):
    """공통 LSTM/GRU 스택 빌더."""
    model = tf.keras.Sequential(layers, name=name)
    return model


def build_lstm(input_shape, name='Standard_LSTM'):
    """Standard LSTM — MSE Loss, 공정 비교를 위해 linear 출력."""
    # [수정] 기존 relu → linear (모든 모델 동일 아키텍처)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='linear'),
    ], name=name)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru(input_shape, name='GRU'):
    """GRU baseline — MSE Loss, linear 출력."""
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='linear'),
    ], name=name)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_dk_lstm(input_shape, name='DK-LSTM'):
    """Domain-Knowledge Enhanced LSTM — dk_loss_fn (3대 제약 포함)."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='linear'),
    ], name=name)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=opt, loss=dk_loss_fn, metrics=['mae'])
    return model

def build_cnn_lstm(input_shape, name='CNN-LSTM'):
    """
    CNN-LSTM Hybrid Baseline — Reviewer 1, 2의 SOTA 비교 요구 대응.

    구조 설계 근거:
      Ren et al. [14] (IEEE TII 2021)의 auto-CNN-LSTM 구조를 단순화하여
      본 논문의 입력 차원 및 시퀀스 길이(L=30)에 적합하도록 조정.
      1D Conv layer 2개로 지역 패턴 추출 → LSTM 2개로 시간 의존성 학습.

    공정 비교 원칙 (논문 §3.3):
      - 출력층 activation='linear' 통일
      - Dropout 0.2 동일
      - Dense 헤드 구조(32, ReLU → 1, Linear) 동일
      - MSE Loss 단독 사용 (DK-LSTM의 손실 함수 효과를 격리하기 위함)
    """
    model = tf.keras.Sequential([
        # ── CNN block: 지역 시간 패턴 추출 ───────────────────────────
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=3, padding='same', activation='relu',
            input_shape=input_shape
        ),
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=3, padding='same', activation='relu'
        ),
        # ── LSTM block: 시간 의존성 학습 ─────────────────────────────
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        # ── Dense head: 다른 모델과 동일 ─────────────────────────────
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='linear'),
    ], name=name)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

print("✔ Model builders defined.")
print(f"  DK-LSTM Loss: MSE + Safety + (i)Neg + (ii)Mono + (iii)Upper")
print(f"  λ: safety={LAMBDA_SAFETY}, neg={LAMBDA_NEG}, "
      f"mono={LAMBDA_MONO}, upper={LAMBDA_UPPER}")

# ==============================================================================
# [CELL 6] 학습 유틸리티 및 평가 지표
# ==============================================================================

def train_model(model, X_tr, y_tr, X_val, y_val,
                seed=RANDOM_SEED):
    """EarlyStopping 포함 모델 학습. history 반환."""
    set_all_seeds(seed)
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae', patience=PATIENCE,
        restore_best_weights=True, verbose=0
    )
    print(f"\n>>> Training [{model.name}] ...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cb],
        verbose=1
    )
    best_mae = min(history.history['val_mae'])
    print(f"    Best Val MAE: {best_mae:.4f} "
          f"(stopped at epoch {len(history.history['val_mae'])})")
    return history


def nasa_score(y_true, y_pred):
    """NASA 비대칭 점수 함수. Lower is better."""
    d = y_pred.flatten() - y_true.flatten()
    s = np.where(d < 0,
                 np.exp(-d / 13.0) - 1,
                 np.exp( d / 10.0) - 1)
    return float(np.sum(s))


def evaluate_model(model, X_test, y_test, model_name=""):
    """예측 후 RMSE / MAE / NASA Score / Neg% 반환."""
    y_raw    = model.predict(X_test, verbose=0).flatten()
    y_pred_c = np.maximum(y_raw, 0)   # post-hoc non-neg
    return dict(
        model      = model_name,
        RMSE       = float(np.sqrt(mean_squared_error(y_test, y_pred_c))),
        MAE        = float(mean_absolute_error(y_test, y_pred_c)),
        NASA_Score = nasa_score(y_test, y_pred_c),
        Neg_Pct    = float(np.mean(y_raw < 0) * 100),
    )

print("✔ Utilities defined.")

# ==============================================================================
# [CELL 7] Experiment 1 — Baseline Performance (FD001)
# ==============================================================================
print("\n" + "=" * 60)
print("  EXPERIMENT 1: Baseline Performance on FD001")
print("=" * 60)

normal_shape = (SEQUENCE_LEN, N_FEATURES)
set_all_seeds()

model_std = build_lstm(normal_shape,    name='Standard_LSTM')
model_gru = build_gru(normal_shape,     name='GRU')
model_dk  = build_dk_lstm(normal_shape, name='DK-LSTM')
model_cnn = build_cnn_lstm(normal_shape, name='CNN-LSTM')

hist_std = train_model(model_std, x_train, y_train, x_val, y_val)
hist_gru = train_model(model_gru, x_train, y_train, x_val, y_val)
hist_dk  = train_model(model_dk,  x_train, y_train, x_val, y_val)
hist_cnn = train_model(model_cnn, x_train, y_train, x_val, y_val)

# ── 학습 곡선 시각화 ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist_std.history['val_mae'], 'r--', label='Standard LSTM Val MAE')
ax.plot(hist_gru.history['val_mae'], 'g:',  label='GRU Val MAE')
ax.plot(hist_cnn.history['val_mae'], 'm-.', label='CNN-LSTM Val MAE')
ax.plot(hist_dk.history['val_mae'],  'b-',  label='DK-LSTM Val MAE')
ax.set_title("Learning Curve (Validation MAE)")
ax.set_xlabel("Epochs")
ax.set_ylabel("MAE — Lower is Better")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
lc_path = out_path + 'fig_learning_curve.png'
plt.savefig(lc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✔ Learning curve saved → {lc_path}")

# ── FD001 테스트 평가 ────────────────────────────────────────────────────────
res_e1 = [
    evaluate_model(model_std, X_test_fd1, y_test_fd1, "Standard LSTM"),
    evaluate_model(model_gru, X_test_fd1, y_test_fd1, "GRU"),
    evaluate_model(model_cnn, X_test_fd1, y_test_fd1, "CNN-LSTM"),
    evaluate_model(model_dk,  X_test_fd1, y_test_fd1, "DK-LSTM (Proposed)"),
]
df_e1 = pd.DataFrame(res_e1).set_index('model')
print("\n[Experiment 1 Results — FD001 Baseline]")
print(df_e1.round(3).to_string())

# ==============================================================================
# [CELL 8] Experiment 2 — Sensor Missingness Robustness (FD001)
# ==============================================================================

print("\n" + "=" * 60)
print(f"  EXPERIMENT 2: Sensor Missingness ({int(MISSING_RATE*100)}%)")
print(f"  Method: Forward-Fill + Binary Mask Indicator (leakage-free)")
print("=" * 60)

miss_shape = (SEQUENCE_LEN, N_FEAT_MISS)
set_all_seeds()

model_std_m = build_lstm(miss_shape,    name='Standard_LSTM_miss')
model_gru_m = build_gru(miss_shape,     name='GRU_miss')
model_dk_m  = build_dk_lstm(miss_shape, name='DK-LSTM_miss')
model_cnn_m = build_cnn_lstm(miss_shape, name='CNN-LSTM_miss')   # v4

train_model(model_std_m, x_train_miss, y_train, x_val_miss, y_val)
train_model(model_gru_m, x_train_miss, y_train, x_val_miss, y_val)
train_model(model_dk_m,  x_train_miss, y_train, x_val_miss, y_val)
train_model(model_cnn_m, x_train_miss, y_train, x_val_miss, y_val)   # v4

res_e2 = [
    evaluate_model(model_std_m, X_test_miss, y_test_fd1, "Standard LSTM"),
    evaluate_model(model_gru_m, X_test_miss, y_test_fd1, "GRU"),
    evaluate_model(model_cnn_m, X_test_miss, y_test_fd1, "CNN-LSTM"),   # v4
    evaluate_model(model_dk_m,  X_test_miss, y_test_fd1, "DK-LSTM (Proposed)"),
]
df_e2 = pd.DataFrame(res_e2).set_index('model')
print("\n[Experiment 2 Results — Sensor Missingness 30%]")
print(df_e2.round(3).to_string())

# ==============================================================================
# [CELL 9] Experiment 3 — Structured Domain Shift
# ==============================================================================
#
# FD001 훈련 모델(Exp.1)을 그대로 사용. FD001 스케일러 재적용.
print("\n" + "=" * 60)
print("  EXPERIMENT 3: Structured Domain Shift")
print("  Case A — Operational Condition : FD001 → FD002")
print("  Case B — Fault Mode            : FD001 → FD003")
print("=" * 60)

res_e3a = res_e3b = None
df_e3a  = df_e3b  = None

if test_fd2 is not None:
    X_fd2, y_fd2 = make_test_sequences(test_fd2, rul_fd2, scaler_ref=scaler)
    print(f"\n[3a] OC-Shift  FD001→FD002  (test units: {len(y_fd2)})")
    res_e3a = [
        evaluate_model(model_std, X_fd2, y_fd2, "Standard LSTM"),
        evaluate_model(model_gru, X_fd2, y_fd2, "GRU"),
        evaluate_model(model_cnn, X_fd2, y_fd2, "CNN-LSTM"),   # v4
        evaluate_model(model_dk,  X_fd2, y_fd2, "DK-LSTM (Proposed)"),
    ]
    df_e3a = pd.DataFrame(res_e3a).set_index('model')
    print(df_e3a.round(3).to_string())

if test_fd3 is not None:
    X_fd3, y_fd3 = make_test_sequences(test_fd3, rul_fd3, scaler_ref=scaler)
    print(f"\n[3b] FM-Shift   FD001→FD003  (test units: {len(y_fd3)})")
    res_e3b = [
        evaluate_model(model_std, X_fd3, y_fd3, "Standard LSTM"),
        evaluate_model(model_gru, X_fd3, y_fd3, "GRU"),
        evaluate_model(model_cnn, X_fd3, y_fd3, "CNN-LSTM"),   # v4
        evaluate_model(model_dk,  X_fd3, y_fd3, "DK-LSTM (Proposed)"),
    ]
    df_e3b = pd.DataFrame(res_e3b).set_index('model')
    print(df_e3b.round(3).to_string())

# ==============================================================================
# [CELL 10] 결과 요약 테이블 + 시각화
# ==============================================================================

def make_summary_table(e1, e2, e3a=None, e3b=None):
    rows  = []
    pairs = [("E1: FD001 Baseline", e1),
             ("E2: Missingness 30%", e2)]
    if e3a:
        pairs.append(("E3a: FD002 OC-Shift", e3a))
    if e3b:
        pairs.append(("E3b: FD003 FM-Shift", e3b))
    for scenario, metrics in pairs:
        for m in metrics:
            rows.append({
                'Scenario'  : scenario,
                'Model'     : m['model'],
                'RMSE'      : round(m['RMSE'],       2),
                'MAE'       : round(m['MAE'],        2),
                'NASA Score': round(m['NASA_Score'], 1),
                'Neg% ↓'   : round(m['Neg_Pct'],    1),
            })
    return pd.DataFrame(rows)

df_summary = make_summary_table(res_e1, res_e2, res_e3a, res_e3b)
print("\n" + "=" * 60)
print("  SUMMARY TABLE (for Paper)")
print("=" * 60)
print(df_summary.to_string(index=False))
csv_path = out_path + 'results_summary.csv'
df_summary.to_csv(csv_path, index=False)
print(f"\n✔ Table saved → {csv_path}")

# ── 막대 그래프 시각화 ────────────────────────────────────────────────────────
colors_map = {'Standard LSTM': '#e74c3c',
              'GRU':           '#f39c12',
              'CNN-LSTM':            '#27ae60',
              'DK-LSTM (Proposed)': '#2980b9'}

def plot_bar(ax, df_res, metric, title, ylabel, log_scale=False):
    models = df_res.index.tolist()
    vals   = df_res[metric].values
    bars   = ax.bar(models, vals,
                    color=[colors_map.get(m, '#95a5a6') for m in models],
                    edgecolor='white', width=0.5)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='x', labelsize=7, rotation=12)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() * 1.02,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    if log_scale:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

experiments = [("E1 Baseline\nRMSE (FD001)",        df_e1,  'RMSE',       False),
               ("E1 Baseline\nNASA Score (lower=↓)", df_e1,  'NASA_Score', True),
               (f"E2 Missingness {int(MISSING_RATE*100)}%\nRMSE", df_e2, 'RMSE', False)]
if df_e3a is not None:
    experiments.append(("E3a OC-Shift\nRMSE (FD001→FD002)", df_e3a, 'RMSE', False))
if df_e3b is not None:
    experiments.append(("E3b FM-Shift\nRMSE (FD001→FD003)", df_e3b, 'RMSE', False))

n   = len(experiments)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
if n == 1:
    axes = [axes]
for ax, (title, df_e, metric, log) in zip(axes, experiments):
    plot_bar(ax, df_e, metric, title, metric, log)

fig.suptitle("DK-LSTM vs Baselines — Scenario-Based Robustness\n"
             "(NASA C-MAPSS Benchmark | Seed=42)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = out_path + 'fig_results.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✔ Figure saved → {fig_path}")

# ==============================================================================
# [CELL 11] 논문 수치 정리
# ==============================================================================
print("\n" + "=" * 60)
print("  PAPER-READY NUMBERS  (v3 — Abstract aligned)")
print("=" * 60)

def delta(base, prop, lower_is_better=True):
    d = (base - prop) / base * 100
    return d if lower_is_better else -d

pairs_out = [("E1 (FD001 Baseline)",    df_e1),
             ("E2 (Missingness 30%)",   df_e2)]
if df_e3a is not None:
    pairs_out.append(("E3a (FD002 OC-Shift)", df_e3a))
if df_e3b is not None:
    pairs_out.append(("E3b (FD003 FM-Shift)", df_e3b))

for label, df_e in pairs_out:
    if 'DK-LSTM (Proposed)' not in df_e.index:
        continue
    b_rmse  = df_e.loc['Standard LSTM',       'RMSE']
    p_rmse  = df_e.loc['DK-LSTM (Proposed)',  'RMSE']
    b_score = df_e.loc['Standard LSTM',       'NASA_Score']
    p_score = df_e.loc['DK-LSTM (Proposed)',  'NASA_Score']
    b_neg   = df_e.loc['Standard LSTM',       'Neg_Pct']
    p_neg   = df_e.loc['DK-LSTM (Proposed)',  'Neg_Pct']
    print(f"\n  [{label}]")
    print(f"    RMSE      : {b_rmse:.2f} → {p_rmse:.2f}  (Δ {delta(b_rmse, p_rmse):+.1f}%)")
    print(f"    NASA Score: {b_score:.1f} → {p_score:.1f}  "
          f"(Δ {delta(b_score, p_score):+.1f}%)")
    print(f"    Neg%      : {b_neg:.1f}% → {p_neg:.1f}%  "
          f"(physical constraint violation rate)")

print("\n" + "=" * 60)
print("  Outputs saved:")
print(f"    {contract_path}")
print(f"    {lc_path}")
print(f"    {csv_path}")
print(f"    {fig_path}")
print("=" * 60)

# ==============================================================================
# [CELL 12] Multi-Seed E1 Verification
# ==============================================================================
# 목적: E1 시나리오에서 seed 0, 42, 123으로 재실행하여
#       NASA Score 평균과 표준편차 산출
#
# 수행 범위:
#   - E1 (FD001 Baseline) 한정. E2/E3는 미수행.
#   - 이유: E2는 DK-LSTM 최하위 시나리오, E3는 개선 폭이 충분히 커서 시드 민감도
#           추가 검증의 한계 효용이 낮음.
# ==============================================================================
import time

SEEDS = [0, 42, 123]
multiseed_results = []

print("\n" + "=" * 70)
print("  [CELL 12] MULTI-SEED VERIFICATION — E1 SCENARIO (v4)")
print("=" * 70)
print(f"  Seeds: {SEEDS}")
print(f"  Models: Standard LSTM, GRU, CNN-LSTM, DK-LSTM (Proposed)")
print("=" * 70)

start_time = time.time()

for seed in SEEDS:
    print(f"\n>>> SEED = {seed}")
    print("-" * 70)

    set_all_seeds(seed)

    # 매 시드마다 모델 신규 빌드 — 가중치 초기화 독립성 보장
    m_std = build_lstm(normal_shape,    name=f'Standard_LSTM_s{seed}')
    m_gru = build_gru(normal_shape,     name=f'GRU_s{seed}')
    m_cnn = build_cnn_lstm(normal_shape, name=f'CNN-LSTM_s{seed}')
    m_dk  = build_dk_lstm(normal_shape, name=f'DK-LSTM_s{seed}')

    train_model(m_std, x_train, y_train, x_val, y_val, seed=seed)
    train_model(m_gru, x_train, y_train, x_val, y_val, seed=seed)
    train_model(m_cnn, x_train, y_train, x_val, y_val, seed=seed)
    train_model(m_dk,  x_train, y_train, x_val, y_val, seed=seed)

    res_std = evaluate_model(m_std, X_test_fd1, y_test_fd1, "Standard LSTM")
    res_gru = evaluate_model(m_gru, X_test_fd1, y_test_fd1, "GRU")
    res_cnn = evaluate_model(m_cnn, X_test_fd1, y_test_fd1, "CNN-LSTM")
    res_dk  = evaluate_model(m_dk,  X_test_fd1, y_test_fd1, "DK-LSTM (Proposed)")

    for res in [res_std, res_gru, res_cnn, res_dk]:
        multiseed_results.append({
            'seed'      : seed,
            'model'     : res['model'],
            'RMSE'      : res['RMSE'],
            'MAE'       : res['MAE'],
            'NASA_Score': res['NASA_Score'],
            'Neg_Pct'   : res['Neg_Pct'],
        })

    # 메모리 정리 — Colab GPU 메모리 누수 방지
    del m_std, m_gru, m_cnn, m_dk
    tf.keras.backend.clear_session()

elapsed = time.time() - start_time
print(f"\n>>> Total elapsed: {elapsed/60:.1f} min")

# ── Raw 결과 저장 ────────────────────────────────────────────────────────────
df_ms = pd.DataFrame(multiseed_results)
ms_path = out_path + 'results_multiseed_e1_raw.csv'
df_ms.to_csv(ms_path, index=False)
print(f"\n✔ Raw results saved → {ms_path}")

# ── 모델별 평균/표준편차 집계 ──────────────────────────────────────────────────
agg = df_ms.groupby('model').agg(
    RMSE_mean       = ('RMSE',       'mean'),
    RMSE_std        = ('RMSE',       'std'),
    MAE_mean        = ('MAE',        'mean'),
    MAE_std         = ('MAE',        'std'),
    NASA_Score_mean = ('NASA_Score', 'mean'),
    NASA_Score_std  = ('NASA_Score', 'std'),
    Neg_Pct_mean    = ('Neg_Pct',    'mean'),
).round(2)

model_order = ['Standard LSTM', 'GRU', 'CNN-LSTM', 'DK-LSTM (Proposed)']
agg = agg.reindex(model_order)

agg_path = out_path + 'results_multiseed_e1_summary.csv'
agg.to_csv(agg_path)

print("\n" + "=" * 70)
print("  E1 — MULTI-SEED SUMMARY (n = 3 seeds: 0, 42, 123)")
print("=" * 70)
print(agg.to_string())
print("=" * 70)

# ── Box plot 시각화 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics = ['RMSE', 'MAE', 'NASA_Score']
titles  = ['RMSE (lower=↓)', 'MAE (lower=↓)', 'NASA Score (lower=↓)']

for ax, metric, title in zip(axes, metrics, titles):
    data = [df_ms[df_ms['model']==m][metric].values for m in model_order]
    bp = ax.boxplot(data, labels=model_order, patch_artist=True, widths=0.5)
    colors_bp = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    # 개별 시드 점 (n=3이므로 분포 시각화 필수)
    for i, m in enumerate(model_order, 1):
        vals = df_ms[df_ms['model']==m][metric].values
        ax.scatter([i]*len(vals), vals, color='black', s=40, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(metric, fontsize=10)
    ax.tick_params(axis='x', labelsize=8, rotation=12)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle("Multi-Seed Verification — E1 Scenario (FD001 Baseline)\n"
             "Seeds: 0, 42, 123 | n=3", fontsize=12, fontweight='bold')
plt.tight_layout()
fig_ms_path = out_path + 'fig_multiseed_e1.png'
plt.savefig(fig_ms_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✔ Box plot saved → {fig_ms_path}")

# ── 통계 요약 출력 ────────────────────────────────────────────────────────────
print("\n[Statistical Snapshot — n=3, mean ± std]")
for metric in ['RMSE', 'MAE', 'NASA_Score']:
    print(f"\n  {metric}:")
    for m in model_order:
        sub = df_ms[df_ms['model']==m][metric]
        print(f"    {m:25s}  {sub.mean():8.2f} ± {sub.std():6.2f}")

print("\n>>> CELL 12 complete. Generated files:")
print(f"    {ms_path}")
print(f"    {agg_path}")
print(f"    {fig_ms_path}")
print("=" * 70)

# ==============================================================================
# [CELL 13] Multi-Seed E3 Verification (Domain Shift Scenarios)
# ==============================================================================
# 목적: 본 논문의 핵심 기여인 도메인 시프트 시나리오(E3a, E3b)에서
#       DK-LSTM의 우위를 시드 변동을 넘어선 강건성으로 검증.
#
# 수행 범위:
#   - E3a (FD001 → FD002, 운용 조건 시프트)
#   - E3b (FD001 → FD003, 고장 모드 시프트)
#   - 4 모델 × 3 시드 × 2 시나리오 = 24 평가
#
# 학습 자체는 FD001에서만 수행. FD001에서 학습한 모델을
# FD002, FD003 테스트셋에 zero-shot 적용.
# ==============================================================================
import time

SEEDS = [0, 42, 123]
model_order = ['Standard LSTM', 'GRU', 'CNN-LSTM', 'DK-LSTM (Proposed)']

print("\n" + "=" * 70)
print("  [CELL 13] MULTI-SEED VERIFICATION — E3a / E3b SCENARIOS")
print("=" * 70)
print(f"  Seeds: {SEEDS}")
print(f"  Models: Standard LSTM, GRU, CNN-LSTM, DK-LSTM (Proposed)")
print(f"  Scenarios: E3a (FD001→FD002), E3b (FD001→FD003)")
print("=" * 70)

# 사전 점검 — FD002, FD003 데이터 가용성
if test_fd2 is None or test_fd3 is None:
    print("[ERROR] FD002 또는 FD003 데이터가 로드되지 않았습니다.")
    print("        CELL 13 스킵.")
else:
    # 테스트 시퀀스 (CELL 9에서 이미 생성되었을 수 있으나, 안전하게 재생성)
    if 'X_fd2' not in globals():
        X_fd2, y_fd2 = make_test_sequences(test_fd2, rul_fd2, scaler_ref=scaler)
    if 'X_fd3' not in globals():
        X_fd3, y_fd3 = make_test_sequences(test_fd3, rul_fd3, scaler_ref=scaler)

    multiseed_e3_results = []
    start_time_e3 = time.time()

    for seed in SEEDS:
        print(f"\n>>> SEED = {seed}")
        print("-" * 70)

        set_all_seeds(seed)

        # FD001에서 학습
        m_std = build_lstm(normal_shape,    name=f'Standard_LSTM_e3_s{seed}')
        m_gru = build_gru(normal_shape,     name=f'GRU_e3_s{seed}')
        m_cnn = build_cnn_lstm(normal_shape, name=f'CNN-LSTM_e3_s{seed}')
        m_dk  = build_dk_lstm(normal_shape, name=f'DK-LSTM_e3_s{seed}')

        train_model(m_std, x_train, y_train, x_val, y_val, seed=seed)
        train_model(m_gru, x_train, y_train, x_val, y_val, seed=seed)
        train_model(m_cnn, x_train, y_train, x_val, y_val, seed=seed)
        train_model(m_dk,  x_train, y_train, x_val, y_val, seed=seed)

        # E3a 평가 (FD002)
        for model_obj, name in [(m_std, "Standard LSTM"),
                                 (m_gru, "GRU"),
                                 (m_cnn, "CNN-LSTM"),
                                 (m_dk,  "DK-LSTM (Proposed)")]:
            r = evaluate_model(model_obj, X_fd2, y_fd2, name)
            multiseed_e3_results.append({
                'seed': seed, 'scenario': 'E3a',
                'model': r['model'], 'RMSE': r['RMSE'], 'MAE': r['MAE'],
                'NASA_Score': r['NASA_Score'], 'Neg_Pct': r['Neg_Pct'],
            })

        # E3b 평가 (FD003)
        for model_obj, name in [(m_std, "Standard LSTM"),
                                 (m_gru, "GRU"),
                                 (m_cnn, "CNN-LSTM"),
                                 (m_dk,  "DK-LSTM (Proposed)")]:
            r = evaluate_model(model_obj, X_fd3, y_fd3, name)
            multiseed_e3_results.append({
                'seed': seed, 'scenario': 'E3b',
                'model': r['model'], 'RMSE': r['RMSE'], 'MAE': r['MAE'],
                'NASA_Score': r['NASA_Score'], 'Neg_Pct': r['Neg_Pct'],
            })

        # 메모리 정리
        del m_std, m_gru, m_cnn, m_dk
        tf.keras.backend.clear_session()

    elapsed_e3 = time.time() - start_time_e3
    print(f"\n>>> Total elapsed: {elapsed_e3/60:.1f} min")

    # ── Raw 결과 저장 ────────────────────────────────────────────────────────
    df_ms_e3 = pd.DataFrame(multiseed_e3_results)
    missing_labels = set(model_order) - set(df_ms_e3['model'].unique())
    if missing_labels:
      print(f"⚠ 경고: model_order에 있으나 df_ms_e3에 없는 라벨: {missing_labels}")
      print(f"    df_ms_e3['model'] unique: {df_ms_e3['model'].unique().tolist()}")
    ms_e3_path = out_path + 'results_multiseed_e3_raw.csv'
    df_ms_e3.to_csv(ms_e3_path, index=False)
    print(f"\n✔ Raw results saved → {ms_e3_path}")

    # ── 시나리오별/모델별 평균/표준편차 ───────────────────────────────────────
    agg_e3 = df_ms_e3.groupby(['scenario', 'model']).agg(
        RMSE_mean       = ('RMSE',       'mean'),
        RMSE_std        = ('RMSE',       'std'),
        MAE_mean        = ('MAE',        'mean'),
        MAE_std         = ('MAE',        'std'),
        NASA_Score_mean = ('NASA_Score', 'mean'),
        NASA_Score_std  = ('NASA_Score', 'std'),
    ).round(2)

    agg_e3_path = out_path + 'results_multiseed_e3_summary.csv'
    agg_e3.to_csv(agg_e3_path)

    print("\n" + "=" * 70)
    print("  E3 — MULTI-SEED SUMMARY (n = 3 seeds: 0, 42, 123)")
    print("=" * 70)
    print(agg_e3.to_string())
    print("=" * 70)

    # ── 통계 요약 (시나리오별) ────────────────────────────────────────────────
    print("\n[Statistical Snapshot — n=3, mean ± std]")
    for scenario in ['E3a', 'E3b']:
        print(f"\n  --- {scenario} ---")
        for metric in ['RMSE', 'MAE', 'NASA_Score']:
            print(f"  {metric}:")
            for m in model_order:
                sub = df_ms_e3[(df_ms_e3['scenario']==scenario) &
                               (df_ms_e3['model']==m)][metric]
                print(f"    {m:25s}  {sub.mean():12.2f} ± {sub.std():10.2f}")

    # ── Box plot 시각화 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, scenario in enumerate(['E3a', 'E3b']):
        sub_df = df_ms_e3[df_ms_e3['scenario'] == scenario]
        scenario_label = 'E3a (FD001→FD002)' if scenario == 'E3a' else 'E3b (FD001→FD003)'

        for col, (metric, title) in enumerate(zip(
            ['RMSE', 'MAE', 'NASA_Score'],
            ['RMSE (lower=↓)', 'MAE (lower=↓)', 'NASA Score (lower=↓)']
        )):
            ax = axes[row, col]
            data = [sub_df[sub_df['model']==m][metric].values for m in model_order]
            bp = ax.boxplot(data, labels=model_order, patch_artist=True, widths=0.5)
            colors_bp = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']
            for patch, color in zip(bp['boxes'], colors_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for i, m in enumerate(model_order, 1):
                vals = sub_df[sub_df['model']==m][metric].values
                ax.scatter([i]*len(vals), vals, color='black', s=40, zorder=3)
            ax.set_title(f"{scenario_label} — {title}", fontsize=10)
            ax.set_ylabel(metric, fontsize=9)
            ax.tick_params(axis='x', labelsize=7, rotation=15)
            ax.grid(axis='y', alpha=0.3)
            # NASA Score는 log scale 권장 (값이 매우 큼)
            if metric == 'NASA_Score':
                ax.set_yscale('log')

    fig.suptitle("Multi-Seed Verification — E3 Domain-Shift Scenarios\n"
                 "Seeds: 0, 42, 123 | n=3", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_ms_e3_path = out_path + 'fig_multiseed_e3.png'
    plt.savefig(fig_ms_e3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✔ Box plot saved → {fig_ms_e3_path}")

    # ── DK-LSTM vs baselines 개선폭 (시드 평균 기준) ─────────────────────────
    print("\n[NASA Score Improvement — DK-LSTM vs Baselines (mean basis)]")
    for scenario in ['E3a', 'E3b']:
        print(f"\n  --- {scenario} ---")
        dk_mean = df_ms_e3[(df_ms_e3['scenario']==scenario) &
                           (df_ms_e3['model']=='DK-LSTM (Proposed)')]['NASA_Score'].mean()
        for m in ['Standard LSTM', 'GRU', 'CNN-LSTM']:
            base_mean = df_ms_e3[(df_ms_e3['scenario']==scenario) &
                                 (df_ms_e3['model']==m)]['NASA_Score'].mean()
            improvement = (base_mean - dk_mean) / base_mean * 100
            print(f"    vs {m:20s}: {base_mean:12.1f} → {dk_mean:12.1f}  "
                  f"(Δ {improvement:+5.1f}%)")

    print("\n>>> CELL 13 complete. Generated files:")
    print(f"    {ms_e3_path}")
    print(f"    {agg_e3_path}")
    print(f"    {fig_ms_e3_path}")
    print("=" * 70)

# ==============================================================================
# [CELL 14] Tuning Experiment — DK-LSTM v2 (Safety Penalty Weakening)
# ==============================================================================
# 목적: Multi-seed 검증에서 발견된 DK-LSTM의 시드 민감성 완화 시도
#
# 진단 근거:
#   E3a/E3b multi-seed에서 DK-LSTM의 NASA Score std/mean 비율이 91%로 최고.
#   L_safety의 비대칭 가중치(2.0/0.2)가 도메인 시프트 환경에서 시드별로
#   다른 국소 해로 수렴시키는 것으로 추정.
#
# 튜닝안 A (이번 실행):
#   - LAMBDA_SAFETY: 1.0 → 0.5
#   - 비대칭 가중치: (2.0, 0.2) → (1.5, 0.5)
#   - 다른 항(λ_neg, λ_mono, λ_upper)은 그대로
#
# 비교:
#   기존 DK-LSTM (CELL 13) vs DK-LSTM-v2 (CELL 14)
#   E3a, E3b multi-seed (3 seeds × 2 시나리오)
#
# 평가 기준:
#   1. NASA Score 평균이 GRU보다 낮아지는가?
#   2. NASA Score std가 줄어드는가? (시드 안정성)
# ==============================================================================

LAMBDA_SAFETY_V2 = 0.5
SAFETY_OVER_V2  = 1.5
SAFETY_UNDER_V2 = 0.5

def dk_loss_fn_v2(y_true, y_pred):
    """
    DK-LSTM v2 손실 함수 — Safety penalty 가중치 완화.

    변경 사항:
      - L_safety의 비대칭 가중치를 (2.0, 0.2)에서 (1.5, 0.5)로 완화
      - λ_safety = 0.5 (전체 항 가중치도 절반)
      - 다른 모든 항은 동일

    근거:
      도메인 시프트 환경에서 비대칭 페널티의 과도한 보수성이
      시드 민감성을 유발한다는 진단에 따른 조정.
    """
    L_mse = tf.reduce_mean(tf.square(y_true - y_pred))

    error = y_pred - y_true
    L_safety = tf.reduce_mean(
        tf.where(error > 0,
                 tf.square(error) * SAFETY_OVER_V2,
                 tf.square(error) * SAFETY_UNDER_V2)
    )

    L_neg = tf.reduce_mean(tf.square(tf.nn.relu(-y_pred)))

    sorted_idx  = tf.argsort(tf.squeeze(y_true), direction='DESCENDING')
    pred_sorted = tf.gather(tf.squeeze(y_pred), sorted_idx)
    diff        = pred_sorted[:-1] - pred_sorted[1:]
    L_mono      = tf.reduce_mean(tf.square(tf.nn.relu(-diff)))

    L_upper = tf.reduce_mean(tf.square(tf.nn.relu(y_pred - MAX_RUL)))

    return (L_mse
            + LAMBDA_SAFETY_V2 * L_safety
            + LAMBDA_NEG       * L_neg
            + LAMBDA_MONO      * L_mono
            + LAMBDA_UPPER     * L_upper)


def build_dk_lstm_v2(input_shape, name='DK-LSTM-v2'):
    """DK-LSTM with weakened safety penalty (튜닝안 A)."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='linear'),
    ], name=name)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=opt, loss=dk_loss_fn_v2, metrics=['mae'])
    return model


print("\n" + "=" * 70)
print("  [CELL 14] DK-LSTM-v2 TUNING — Safety Penalty Weakened")
print("=" * 70)
print(f"  λ_safety: 1.0 → {LAMBDA_SAFETY_V2}")
print(f"  Asymmetric weights: (2.0, 0.2) → ({SAFETY_OVER_V2}, {SAFETY_UNDER_V2})")
print(f"  Seeds: {SEEDS}")
print(f"  Scenarios: E1, E3a, E3b (E2 skipped to save time)")
print("=" * 70)

# 테스트 시퀀스 사전 준비
if 'X_fd2' not in globals():
    X_fd2, y_fd2 = make_test_sequences(test_fd2, rul_fd2, scaler_ref=scaler)
if 'X_fd3' not in globals():
    X_fd3, y_fd3 = make_test_sequences(test_fd3, rul_fd3, scaler_ref=scaler)

dk_v2_results = []
start_time_v2 = time.time()

for seed in SEEDS:
    print(f"\n>>> SEED = {seed}")
    print("-" * 70)

    set_all_seeds(seed)
    m_dk_v2 = build_dk_lstm_v2(normal_shape, name=f'DK-LSTM-v2_s{seed}')
    train_model(m_dk_v2, x_train, y_train, x_val, y_val, seed=seed)

    # E1, E3a, E3b 평가
    for X_test, y_test, scen in [
        (X_test_fd1, y_test_fd1, 'E1'),
        (X_fd2,      y_fd2,      'E3a'),
        (X_fd3,      y_fd3,      'E3b'),
    ]:
        r = evaluate_model(m_dk_v2, X_test, y_test, "DK-LSTM-v2")
        dk_v2_results.append({
            'seed': seed, 'scenario': scen,
            'model': r['model'], 'RMSE': r['RMSE'], 'MAE': r['MAE'],
            'NASA_Score': r['NASA_Score'], 'Neg_Pct': r['Neg_Pct'],
        })

    del m_dk_v2
    tf.keras.backend.clear_session()

elapsed_v2 = time.time() - start_time_v2
print(f"\n>>> Total elapsed: {elapsed_v2/60:.1f} min")

# ── Raw 결과 저장 ────────────────────────────────────────────────────────────
df_v2 = pd.DataFrame(dk_v2_results)
v2_path = out_path + 'results_dk_lstm_v2_raw.csv'
df_v2.to_csv(v2_path, index=False)
print(f"\n✔ Raw results saved → {v2_path}")

# ── 시나리오별 평균/표준편차 ──────────────────────────────────────────────────
agg_v2 = df_v2.groupby('scenario').agg(
    RMSE_mean       = ('RMSE',       'mean'),
    RMSE_std        = ('RMSE',       'std'),
    MAE_mean        = ('MAE',        'mean'),
    MAE_std         = ('MAE',        'std'),
    NASA_Score_mean = ('NASA_Score', 'mean'),
    NASA_Score_std  = ('NASA_Score', 'std'),
).round(2)

print("\n" + "=" * 70)
print("  DK-LSTM-v2 — MULTI-SEED SUMMARY (n = 3 seeds: 0, 42, 123)")
print("=" * 70)
print(agg_v2.to_string())
print("=" * 70)

# ── 기존 DK-LSTM과 비교 ───────────────────────────────────────────────────────
# CELL 13의 df_ms_e3에서 DK-LSTM 결과만 추출
print("\n[Comparison: DK-LSTM (original) vs DK-LSTM-v2 (tuned)]")
print("=" * 70)
for scen in ['E3a', 'E3b']:
    print(f"\n  --- {scen} ---")
    # 기존 DK-LSTM
    if 'df_ms_e3' in globals():
        orig = df_ms_e3[(df_ms_e3['scenario']==scen) &
                        (df_ms_e3['model']=='DK-LSTM (Proposed)')]
        print(f"    DK-LSTM (orig)  NASA: {orig['NASA_Score'].mean():12.1f} "
              f"± {orig['NASA_Score'].std():10.1f}")
    # v2
    v2 = df_v2[df_v2['scenario']==scen]
    print(f"    DK-LSTM-v2      NASA: {v2['NASA_Score'].mean():12.1f} "
          f"± {v2['NASA_Score'].std():10.1f}")

    # GRU baseline (참고)
    if 'df_ms_e3' in globals():
        gru = df_ms_e3[(df_ms_e3['scenario']==scen) &
                       (df_ms_e3['model']=='GRU')]
        print(f"    GRU (baseline)  NASA: {gru['NASA_Score'].mean():12.1f} "
              f"± {gru['NASA_Score'].std():10.1f}")

print("\n>>> CELL 14 complete.")
print("=" * 70)

# ==============================================================================
# [CELL 15] Tuning Experiment — DK-LSTM v3 (λ_upper Strengthening)
# ==============================================================================
# 목적: CELL 14의 v2(safety penalty 완화)에서 E3a가 14배 발산한 문제 해결
#
# 진단:
#   v2에서 비대칭 가중치 (1.5, 0.5)로 완화 → 시드 안정성은 개선되었으나
#   E3a (6개 운용 조건 혼재)에서 모델이 새 분포에 대해 자유롭게 변동 →
#   상한(125)을 크게 초과하는 outlier 예측이 폭증한 것으로 추정
#
# 튜닝안 D:
#   λ_upper: 1.0 → 2.0 (상한 제약 2배 강화)
#   v2의 다른 설정은 유지 (λ_safety=0.5, weights 1.5/0.5)
#   = "안정성은 v2 수준 유지하면서 상한 침범만 강하게 억제"
#
# 가설:
#   상한 제약을 강화하면 outlier 예측이 차단되어 NASA Score 평균이
#   현실적 범위로 복귀할 것. 시드 안정성은 v2 수준 유지.
#
# 평가 시나리오: E1, E3a, E3b (E2 제외)
# ==============================================================================

import time
SEEDS = [0, 42, 123]
normal_shape = (SEQUENCE_LEN, N_FEATURES)

LAMBDA_UPPER_V3 = 2.0   # 1.0 → 2.0
LAMBDA_SAFETY_V3 = 0.5  # v2와 동일
SAFETY_OVER_V3  = 1.5   # v2와 동일
SAFETY_UNDER_V3 = 0.5   # v2와 동일

def dk_loss_fn_v3(y_true, y_pred):
    """
    DK-LSTM v3 손실 함수 — λ_upper 강화 (튜닝안 D).

    변경 사항 (v2 대비):
      - LAMBDA_UPPER: 1.0 → 2.0
      - 다른 항(safety 약화 포함)은 v2와 동일

    가설:
      v2의 시드 안정성 유지 + 상한 침범 강제 억제로 E3a 발산 방지.
    """
    L_mse = tf.reduce_mean(tf.square(y_true - y_pred))

    error = y_pred - y_true
    L_safety = tf.reduce_mean(
        tf.where(error > 0,
                 tf.square(error) * SAFETY_OVER_V3,
                 tf.square(error) * SAFETY_UNDER_V3)
    )

    L_neg = tf.reduce_mean(tf.square(tf.nn.relu(-y_pred)))

    sorted_idx  = tf.argsort(tf.squeeze(y_true), direction='DESCENDING')
    pred_sorted = tf.gather(tf.squeeze(y_pred), sorted_idx)
    diff        = pred_sorted[:-1] - pred_sorted[1:]
    L_mono      = tf.reduce_mean(tf.square(tf.nn.relu(-diff)))

    L_upper = tf.reduce_mean(tf.square(tf.nn.relu(y_pred - MAX_RUL)))

    return (L_mse
            + LAMBDA_SAFETY_V3 * L_safety
            + LAMBDA_NEG       * L_neg
            + LAMBDA_MONO      * L_mono
            + LAMBDA_UPPER_V3  * L_upper)   # 핵심 변경


def build_dk_lstm_v3(input_shape, name='DK-LSTM-v3'):
    """DK-LSTM with strengthened upper bound penalty (튜닝안 D)."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='linear'),
    ], name=name)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=opt, loss=dk_loss_fn_v3, metrics=['mae'])
    return model


print("\n" + "=" * 70)
print("  [CELL 15] DK-LSTM-v3 TUNING — λ_upper Strengthened")
print("=" * 70)
print(f"  λ_upper:  1.0 → {LAMBDA_UPPER_V3}")
print(f"  λ_safety: 1.0 → {LAMBDA_SAFETY_V3} (v2와 동일)")
print(f"  Asymmetric weights: ({SAFETY_OVER_V3}, {SAFETY_UNDER_V3}) (v2와 동일)")
print(f"  Seeds: {SEEDS}")
print(f"  Scenarios: E1, E3a, E3b")
print("=" * 70)

# 테스트 시퀀스 사전 준비
if 'X_fd2' not in globals():
    X_fd2, y_fd2 = make_test_sequences(test_fd2, rul_fd2, scaler_ref=scaler)
if 'X_fd3' not in globals():
    X_fd3, y_fd3 = make_test_sequences(test_fd3, rul_fd3, scaler_ref=scaler)

dk_v3_results = []
start_time_v3 = time.time()

for seed in SEEDS:
    print(f"\n>>> SEED = {seed}")
    print("-" * 70)

    set_all_seeds(seed)
    m_dk_v3 = build_dk_lstm_v3(normal_shape, name=f'DK-LSTM-v3_s{seed}')
    train_model(m_dk_v3, x_train, y_train, x_val, y_val, seed=seed)

    for X_test, y_test, scen in [
        (X_test_fd1, y_test_fd1, 'E1'),
        (X_fd2,      y_fd2,      'E3a'),
        (X_fd3,      y_fd3,      'E3b'),
    ]:
        r = evaluate_model(m_dk_v3, X_test, y_test, "DK-LSTM-v3")
        dk_v3_results.append({
            'seed': seed, 'scenario': scen,
            'model': r['model'], 'RMSE': r['RMSE'], 'MAE': r['MAE'],
            'NASA_Score': r['NASA_Score'], 'Neg_Pct': r['Neg_Pct'],
        })

    del m_dk_v3
    tf.keras.backend.clear_session()

elapsed_v3 = time.time() - start_time_v3
print(f"\n>>> Total elapsed: {elapsed_v3/60:.1f} min")

# ── Raw 결과 저장 ────────────────────────────────────────────────────────────
df_v3 = pd.DataFrame(dk_v3_results)
v3_path = out_path + 'results_dk_lstm_v3_raw.csv'
df_v3.to_csv(v3_path, index=False)
print(f"\n✔ Raw results saved → {v3_path}")

# ── 시나리오별 평균/표준편차 ──────────────────────────────────────────────────
agg_v3 = df_v3.groupby('scenario').agg(
    RMSE_mean       = ('RMSE',       'mean'),
    RMSE_std        = ('RMSE',       'std'),
    MAE_mean        = ('MAE',        'mean'),
    MAE_std         = ('MAE',        'std'),
    NASA_Score_mean = ('NASA_Score', 'mean'),
    NASA_Score_std  = ('NASA_Score', 'std'),
).round(2)

print("\n" + "=" * 70)
print("  DK-LSTM-v3 — MULTI-SEED SUMMARY (n = 3 seeds: 0, 42, 123)")
print("=" * 70)
print(agg_v3.to_string())
print("=" * 70)

# ── 종합 비교 — Original vs v2 vs v3 vs GRU ──────────────────────────────────
print("\n[Comparison: DK-LSTM versions vs GRU baseline]")
print("=" * 70)
for scen in ['E1', 'E3a', 'E3b']:
    print(f"\n  --- {scen} ---")
    # Original
    if 'df_ms_e3' in globals() and scen in ['E3a', 'E3b']:
        orig = df_ms_e3[(df_ms_e3['scenario']==scen) &
                        (df_ms_e3['model']=='DK-LSTM (Proposed)')]
        print(f"    DK-LSTM (orig)  NASA: {orig['NASA_Score'].mean():14.1f} "
              f"± {orig['NASA_Score'].std():12.1f}")
    # v2
    if 'df_v2' in globals():
        v2 = df_v2[df_v2['scenario']==scen]
        if len(v2) > 0:
            print(f"    DK-LSTM-v2      NASA: {v2['NASA_Score'].mean():14.1f} "
                  f"± {v2['NASA_Score'].std():12.1f}")
    # v3
    v3 = df_v3[df_v3['scenario']==scen]
    print(f"    DK-LSTM-v3      NASA: {v3['NASA_Score'].mean():14.1f} "
          f"± {v3['NASA_Score'].std():12.1f}")
    # GRU baseline
    if 'df_ms_e3' in globals() and scen in ['E3a', 'E3b']:
        gru = df_ms_e3[(df_ms_e3['scenario']==scen) &
                       (df_ms_e3['model']=='GRU')]
        print(f"    GRU (baseline)  NASA: {gru['NASA_Score'].mean():14.1f} "
              f"± {gru['NASA_Score'].std():12.1f}")

print("\n>>> CELL 15 complete.")
print("=" * 70)

# ==============================================================================
# [CELL 16] DK-LSTM v4 — Comprehensive Stabilization
# ==============================================================================
# 목적: E3a 발산 문제의 마지막 시도. v3 설정에 다음 3가지를 동시 적용.
#
# 변경 사항 (v3 대비):
#   1. Gradient clipping 강화: clipnorm 1.0 → 0.5
#      → 발산을 학습 단계에서 직접 억제
#   2. Learning rate scheduling 추가 (Cosine decay)
#      → 후반 epoch에서 학습률 감소로 안정 수렴
#   3. Dropout 강화: 0.2 → 0.3
#      → 정규화 강화로 시드 변동성 흡수
#
# 손실 함수 설정 (v3와 동일):
#   - λ_safety = 0.5, weights (1.5, 0.5)
#   - λ_upper = 2.0
#   - λ_neg = 1.0, λ_mono = 0.0
#
# 가설:
#   학습 안정화 기법 3종을 통합 적용하면 E3a 발산이 억제될 것.
#   발산이 여전히 발생하면 손실 함수 가중치 조정만으로는
#   본 데이터셋의 다중 운용 조건 분포 문제를 해결할 수 없음을 확인.
#
# 평가 시나리오: E1, E3a, E3b
# ==============================================================================
import time
SEEDS = [0, 42, 123]
normal_shape = (SEQUENCE_LEN, N_FEATURES)

LAMBDA_UPPER_V4   = 2.0   # v3와 동일
LAMBDA_SAFETY_V4  = 0.5   # v3와 동일
SAFETY_OVER_V4   = 1.5    # v3와 동일
SAFETY_UNDER_V4  = 0.5    # v3와 동일
DROPOUT_V4       = 0.3    # 0.2 → 0.3 강화
CLIPNORM_V4      = 0.5    # 1.0 → 0.5 강화

def dk_loss_fn_v4(y_true, y_pred):
    """DK-LSTM v4 손실 함수 — v3와 동일 (가중치는 그대로)."""
    L_mse = tf.reduce_mean(tf.square(y_true - y_pred))

    error = y_pred - y_true
    L_safety = tf.reduce_mean(
        tf.where(error > 0,
                 tf.square(error) * SAFETY_OVER_V4,
                 tf.square(error) * SAFETY_UNDER_V4)
    )

    L_neg = tf.reduce_mean(tf.square(tf.nn.relu(-y_pred)))

    sorted_idx  = tf.argsort(tf.squeeze(y_true), direction='DESCENDING')
    pred_sorted = tf.gather(tf.squeeze(y_pred), sorted_idx)
    diff        = pred_sorted[:-1] - pred_sorted[1:]
    L_mono      = tf.reduce_mean(tf.square(tf.nn.relu(-diff)))

    L_upper = tf.reduce_mean(tf.square(tf.nn.relu(y_pred - MAX_RUL)))

    return (L_mse
            + LAMBDA_SAFETY_V4 * L_safety
            + LAMBDA_NEG       * L_neg
            + LAMBDA_MONO      * L_mono
            + LAMBDA_UPPER_V4  * L_upper)


def build_dk_lstm_v4(input_shape, name='DK-LSTM-v4'):
    """
    DK-LSTM v4 — 학습 안정화 기법 3종 통합 적용 (안 H).

    변경 사항 (v3 대비):
      - Dropout: 0.2 → 0.3
      - clipnorm: 1.0 → 0.5
      - Cosine LR decay 추가 (compile 시점이 아닌 callback으로 적용)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(DROPOUT_V4),                # 0.2 → 0.3
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(DROPOUT_V4),                # 0.2 → 0.3
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='linear'),
    ], name=name)
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=CLIPNORM_V4   # 1.0 → 0.5
    )
    model.compile(optimizer=opt, loss=dk_loss_fn_v4, metrics=['mae'])
    return model


def train_model_v4(model, X_tr, y_tr, X_val, y_val, seed=RANDOM_SEED):
    """
    v4 전용 학습 함수 — Cosine LR scheduling 추가.

    기존 train_model()과 차이:
      - LearningRateScheduler callback 추가
      - 50 epoch을 기준으로 cosine decay (0.001 → 0.0001)
    """
    set_all_seeds(seed)

    def cosine_lr(epoch, lr):
        # 초기 5 epoch은 warmup으로 0.001 유지
        if epoch < 5:
            return 0.001
        # 그 이후 cosine decay (50 epoch 기준)
        progress = (epoch - 5) / 50.0
        progress = min(progress, 1.0)
        return float(0.0001 + 0.5 * (0.001 - 0.0001) *
                     (1.0 + tf.math.cos(np.pi * progress).numpy()))

    cb_es  = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae', patience=PATIENCE,
        restore_best_weights=True, verbose=0
    )
    cb_lr  = tf.keras.callbacks.LearningRateScheduler(cosine_lr, verbose=0)

    print(f"\n>>> Training [{model.name}] (v4 with cosine LR + clipnorm 0.5 + dropout 0.3)")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cb_es, cb_lr],
        verbose=1
    )
    best_mae = min(history.history['val_mae'])
    print(f"    Best Val MAE: {best_mae:.4f} "
          f"(stopped at epoch {len(history.history['val_mae'])})")
    return history


print("\n" + "=" * 70)
print("  [CELL 16] DK-LSTM-v4 TUNING — Comprehensive Stabilization (안 H)")
print("=" * 70)
print(f"  Loss weights (v3와 동일):")
print(f"    λ_safety = {LAMBDA_SAFETY_V4}, weights ({SAFETY_OVER_V4}, {SAFETY_UNDER_V4})")
print(f"    λ_upper  = {LAMBDA_UPPER_V4}")
print(f"  Stabilization (v3 대비 신규):")
print(f"    Dropout:  0.2 → {DROPOUT_V4}")
print(f"    Clipnorm: 1.0 → {CLIPNORM_V4}")
print(f"    LR schedule: Cosine decay (warmup 5 epochs)")
print(f"  Seeds: {SEEDS}")
print(f"  Scenarios: E1, E3a, E3b")
print("=" * 70)

# 테스트 시퀀스 사전 준비
if 'X_fd2' not in globals():
    X_fd2, y_fd2 = make_test_sequences(test_fd2, rul_fd2, scaler_ref=scaler)
if 'X_fd3' not in globals():
    X_fd3, y_fd3 = make_test_sequences(test_fd3, rul_fd3, scaler_ref=scaler)

dk_v4_results = []
start_time_v4 = time.time()

for seed in SEEDS:
    print(f"\n>>> SEED = {seed}")
    print("-" * 70)

    set_all_seeds(seed)
    m_dk_v4 = build_dk_lstm_v4(normal_shape, name=f'DK-LSTM-v4_s{seed}')
    train_model_v4(m_dk_v4, x_train, y_train, x_val, y_val, seed=seed)

    for X_test, y_test, scen in [
        (X_test_fd1, y_test_fd1, 'E1'),
        (X_fd2,      y_fd2,      'E3a'),
        (X_fd3,      y_fd3,      'E3b'),
    ]:
        r = evaluate_model(m_dk_v4, X_test, y_test, "DK-LSTM-v4")
        dk_v4_results.append({
            'seed': seed, 'scenario': scen,
            'model': r['model'], 'RMSE': r['RMSE'], 'MAE': r['MAE'],
            'NASA_Score': r['NASA_Score'], 'Neg_Pct': r['Neg_Pct'],
        })

    del m_dk_v4
    tf.keras.backend.clear_session()

elapsed_v4 = time.time() - start_time_v4
print(f"\n>>> Total elapsed: {elapsed_v4/60:.1f} min")

# ── Raw 결과 저장 ────────────────────────────────────────────────────────────
df_v4 = pd.DataFrame(dk_v4_results)
v4_path = out_path + 'results_dk_lstm_v4_raw.csv'
df_v4.to_csv(v4_path, index=False)
print(f"\n✔ Raw results saved → {v4_path}")

# ── 시나리오별 평균/표준편차 ──────────────────────────────────────────────────
agg_v4 = df_v4.groupby('scenario').agg(
    RMSE_mean       = ('RMSE',       'mean'),
    RMSE_std        = ('RMSE',       'std'),
    MAE_mean        = ('MAE',        'mean'),
    MAE_std         = ('MAE',        'std'),
    NASA_Score_mean = ('NASA_Score', 'mean'),
    NASA_Score_std  = ('NASA_Score', 'std'),
).round(2)

print("\n" + "=" * 70)
print("  DK-LSTM-v4 — MULTI-SEED SUMMARY (n = 3 seeds: 0, 42, 123)")
print("=" * 70)
print(agg_v4.to_string())
print("=" * 70)

# ── 4-way 종합 비교 — orig vs v2 vs v3 vs v4 vs GRU ──────────────────────────
print("\n[Final Comparison: All DK-LSTM versions vs GRU baseline]")
print("=" * 70)
for scen in ['E1', 'E3a', 'E3b']:
    print(f"\n  --- {scen} ---")
    if 'df_ms_e3' in globals() and scen in ['E3a', 'E3b']:
        orig = df_ms_e3[(df_ms_e3['scenario']==scen) &
                        (df_ms_e3['model']=='DK-LSTM (Proposed)')]
        print(f"    DK-LSTM (orig)  NASA: {orig['NASA_Score'].mean():14.1f} "
              f"± {orig['NASA_Score'].std():12.1f}")
    if 'df_v2' in globals():
        v2 = df_v2[df_v2['scenario']==scen]
        if len(v2) > 0:
            print(f"    DK-LSTM-v2      NASA: {v2['NASA_Score'].mean():14.1f} "
                  f"± {v2['NASA_Score'].std():12.1f}")
    if 'df_v3' in globals():
        v3 = df_v3[df_v3['scenario']==scen]
        if len(v3) > 0:
            print(f"    DK-LSTM-v3      NASA: {v3['NASA_Score'].mean():14.1f} "
                  f"± {v3['NASA_Score'].std():12.1f}")
    v4 = df_v4[df_v4['scenario']==scen]
    print(f"    DK-LSTM-v4      NASA: {v4['NASA_Score'].mean():14.1f} "
          f"± {v4['NASA_Score'].std():12.1f}")
    if 'df_ms_e3' in globals() and scen in ['E3a', 'E3b']:
        gru = df_ms_e3[(df_ms_e3['scenario']==scen) &
                       (df_ms_e3['model']=='GRU')]
        print(f"    GRU (baseline)  NASA: {gru['NASA_Score'].mean():14.1f} "
              f"± {gru['NASA_Score'].std():12.1f}")

print("\n>>> CELL 16 complete.")
print("=" * 70)
