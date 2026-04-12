# ==============================================================================
# PAPER: A Standards-Aligned Hybrid AI–Digital Twin Framework for
#        Robust Predictive Maintenance under Data Scarcity
#
# EXPERIMENT SCRIPT v3.1 — Google Colab Ready
# 셀 단위로 구분: [CELL N] 주석 기준으로 Colab에서 분리 실행 가능
#
# ============================================================
# 주요 설계 결정 사항
# ============================================================
#
# [RUL 클리핑]
#   C-MAPSS Piecewise Linear 표준 관행에 따라 .clip(upper=MAX_RUL) 적용.
#   Raw RUL(FD001 최대 ~362 사이클)을 그대로 사용할 경우, Constraint(iii)
#   상한 제약(MAX_RUL=125)과 MSE Loss가 초기 사이클 구간에서 충돌하므로
#   클리핑을 전처리 단계에서 명시적으로 수행.
#
# [Constraint (iii) 상한 제약]
#   Abstract: "(iii) operating-range bounds, are explicitly embedded
#              into the learning objective"
#   bound_penalty = relu(y_pred - MAX_RUL)^2 항으로 구현.
#
# [공정 비교를 위한 출력층 통일]
#   Standard LSTM, GRU, DK-LSTM 모두 출력층 activation='linear'으로 통일.
#   구조적 음수 차단(relu)이 아닌, 손실 함수에 내장된 페널티 항만이
#   물리적 제약을 학습하도록 하여 손실 함수의 기여를 유일한 독립변수로 격리.
#   Neg_Pct는 사후(post-hoc) 지표로만 측정.
#
# [비대칭 Safety Penalty]
#   3대 도메인 제약(i)(ii)(iii)과 별개의 실용적 손실 항(practical loss term).
#   과대예측이 유지보전 지연 및 기기 고장 위험으로 직결되는 해양 안전 임계
#   환경의 특성상, 과대예측(d>0)에 2배, 과소예측(d<0)에 0.2배 비대칭 가중치 적용.
#
# [단조감소 구현 방식 — 배치 정렬 근사법]
#   DK-LSTM은 단일 출력 모델(return_sequences=False)이므로 시퀀스 내부
#   단조성 직접 강제가 불가. 미니배치 내에서 y_true 내림차순 정렬 후
#   예측값의 단조감소 위반을 페널티화하는 배치 정렬 근사법을 채택.
#   다중 운용 조건이 혼재하는 도메인 이동 환경(FD002)에서 배치 정렬 근사법이
#   시계열의 물리적 시간 순서를 왜곡하여 최적화를 불안정하게 만드는 현상이
#   예비 실험에서 확인되어, 현재 실험에서는 lambda_mono=0.0으로 비활성화.
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
SEQUENCE_LEN = 30       # 기존 코드 유지
BATCH_SIZE   = 128
MAX_EPOCHS   = 100
PATIENCE     = 15
MISSING_RATE = 0.30     # 센서 결측 비율 (Experiment 2)
MAX_RUL      = 125.0    # RUL 클리핑 상한 및 Constraint(iii) 공통 사용

# ── DK-LSTM Loss 가중치 (전역 상수로 관리) ───────────────────────────────────
LAMBDA_SAFETY = 1.0     # 비대칭 Safety Penalty (practical loss term)
LAMBDA_NEG    = 1.0     # Constraint (i)  비음수
LAMBDA_MONO   = 0.0     # Constraint (ii) 단조감소
LAMBDA_UPPER  = 1.0     # Constraint (iii) 상한

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

# ── RUL 계산 — Piecewise Linear 클리핑 적용 ──────────────────────────────────
# C-MAPSS 표준 관행에 따라 MAX_RUL로 클리핑.
# Constraint(iii) 상한 제약과 MSE Loss의 충돌을 방지하기 위해
# 전처리 단계에서 명시적으로 수행.
def add_rul(df, max_rul=MAX_RUL):
    max_life  = df.groupby('unit_nr')['time_cycles'].max()
    result_df = df.merge(max_life.to_frame(name='max_life'),
                         left_on='unit_nr', right_index=True)
    result_df['RUL'] = (result_df['max_life'] - result_df['time_cycles']
                        ).clip(upper=max_rul)   # Piecewise Linear 클리핑
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
#
# 구현
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
# 모든 모델 출력층 activation='linear'으로 통일 (공정 비교)
# dk_loss_fn에 Constraint(iii) 상한 제약 포함

def dk_loss_fn(y_true, y_pred):
    """
    Domain-Knowledge Enhanced Loss Function.

    구성:
      · MSE                  : 데이터 손실
      · Safety Penalty       : 실용적 손실 항 (과대예측 억제, 3대 도메인 제약과 별개)
      · Constraint (i)       : 비음수성  — non-negativity of RUL
      · Constraint (ii)      : 단조감소  — monotonic degradation
      · Constraint (iii)     : 운용 범위 — operating-range bounds

    Abstract 대응:
      "Domain-informed constraints, including
       (i) non-negativity of the Remaining Useful Life (RUL),
       (ii) monotonic degradation of the predicted RUL, and
       (iii) operating-range bounds,
       are explicitly embedded into the learning objective"
    """
    # ── 데이터 손실 ───────────────────────────────────────────────────────────
    L_mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # ── Safety Penalty (practical loss, Abstract 3대 제약과 별개) ─────────────
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
    # 예측값이 MAX_RUL(=125)을 초과하면 페널티 부여.
    # Abstract: "(iii) operating-range bounds, are explicitly embedded
    #            into the learning objective"
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
    """Standard LSTM — MSE Loss. 출력층 activation='linear' (공정 비교)."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='linear'),  # 손실 함수 기여 격리
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
        tf.keras.layers.Dense(1,  activation='linear'),  # 손실 함수 기여 격리
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

hist_std = train_model(model_std, x_train, y_train, x_val, y_val)
hist_gru = train_model(model_gru, x_train, y_train, x_val, y_val)
hist_dk  = train_model(model_dk,  x_train, y_train, x_val, y_val)

# ── 학습 곡선 시각화 ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist_std.history['val_mae'], 'r--', label='Standard LSTM Val MAE')
ax.plot(hist_gru.history['val_mae'], 'g:',  label='GRU Val MAE')
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
    evaluate_model(model_dk,  X_test_fd1, y_test_fd1, "DK-LSTM (Proposed)"),
]
df_e1 = pd.DataFrame(res_e1).set_index('model')
print("\n[Experiment 1 Results — FD001 Baseline]")
print(df_e1.round(3).to_string())


# ==============================================================================
# [CELL 8] Experiment 2 — Sensor Missingness Robustness (FD001)
# ==============================================================================
# Abstract: "sensor missingness, addressed through masking indicators
#            and leakage-aware imputation"
print("\n" + "=" * 60)
print(f"  EXPERIMENT 2: Sensor Missingness ({int(MISSING_RATE*100)}%)")
print(f"  Method: Forward-Fill + Binary Mask Indicator (leakage-free)")
print("=" * 60)

miss_shape = (SEQUENCE_LEN, N_FEAT_MISS)
set_all_seeds()

model_std_m = build_lstm(miss_shape,    name='Standard_LSTM_miss')
model_gru_m = build_gru(miss_shape,     name='GRU_miss')
model_dk_m  = build_dk_lstm(miss_shape, name='DK-LSTM_miss')

train_model(model_std_m, x_train_miss, y_train, x_val_miss, y_val)
train_model(model_gru_m, x_train_miss, y_train, x_val_miss, y_val)
train_model(model_dk_m,  x_train_miss, y_train, x_val_miss, y_val)

res_e2 = [
    evaluate_model(model_std_m, X_test_miss, y_test_fd1, "Standard LSTM"),
    evaluate_model(model_gru_m, X_test_miss, y_test_fd1, "GRU"),
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
             "(NASA C-MAPSS Benchmark | Seed=42 | v3)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = out_path + 'fig_results_v3.png'
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
