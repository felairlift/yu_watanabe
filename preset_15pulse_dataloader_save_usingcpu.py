import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import glob
import os
import numpy as np
import pandas as pd
import re
import gc
import psutil


def log_memory_usage(prefix=""):
    """
    現在のメモリ使用状況をログ出力する関数
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()

    print(f"{prefix} Memory Usage:")
    print(f"  Process Memory: {memory_info.rss / (1024**2):.2f} MB (RSS), {memory_info.vms / (1024**2):.2f} MB (VMS)")
    print(f"  System Memory: {system_memory.used / (1024**2):.2f} MB used, {system_memory.available / (1024**2):.2f} MB available")
    print("-" * 50)

# Filtered_USB_Data_shape:(14994, 4012)
# Filtered_USB_Data_shape:(2499, 4012) (6ずつ増やしたとき)

input_camera_files = r"/mnt/sdb/ywatanabe/csv_files/camera_data/volume_fraction_20241117/*"
input_ultrasonic_files = r"/mnt/sdb/ywatanabe/csv_files/2023_2dim_rawcsv"
path_in='/mnt/sdb/ywatanabe/CNN_dataset/reflectedwave_rawcsv'

TDX = "_tdx1"
pattern = re.compile(r"s[0-9]+[-_]g0+[-_]l[0-9]+[-_]t[0-9]+")  # 固液二相流のみに行う。

camera_files = [f for f in glob.glob(input_camera_files) if pattern.search(f)]
print(len(camera_files))

camera_files = sorted(camera_files, key=lambda x: int(re.search(r"s([0-9]+)", x).group(1)), reverse=False)

# 修正：空のテンソルで初期化
X_train = torch.empty((0, 1, 15, 4011), dtype=torch.float32)
X_valid = torch.empty((0, 1, 15, 4011), dtype=torch.float32)
X_test = torch.empty((0, 1, 15, 4011), dtype=torch.float32)
y_train = torch.empty((0,), dtype=torch.float32)
y_valid = torch.empty((0,), dtype=torch.float32)
y_test = torch.empty((0,), dtype=torch.float32)
batch_size = 100  # 一度に処理するデータの数（適宜調整してください）
for camera_folder in camera_files:
    gc.collect()  # メモリ解放
    data = []
    print("basename:" + os.path.basename(camera_folder))
    basename = os.path.basename(camera_folder)
    basename = basename.replace(".csv", "")  # s1_g0_l1_t1
    filepath_ultrasonic = os.path.join(input_ultrasonic_files + TDX, basename + TDX + ".csv")    
    
    try:
        USB_data = pd.read_csv(filepath_ultrasonic, header=None, dtype=np.float32, usecols=range(1, 4012))
        Solid_fraction_data = pd.read_csv(camera_folder, header=None, dtype=np.float32)
    except Exception as e:
        print(f"Error loading file: {basename}, {e}")
        continue  # エラーがあればスキップ

    print("USB_data.shape:" + str(USB_data.shape))
    
    if USB_data.shape[0] < 14986:
        print(f"Skipping insufficient data file: {basename}")
        continue

    for i in range(0, 2497, batch_size):
        batch_data = []
        for j in range(batch_size):
            if i + j >= 2497:  # 範囲外を防ぐ
                break
            try:
                Solid_15pulse = Solid_fraction_data.iloc[i + j + 1, 0]
                USB_data_15pulse = USB_data.iloc[(i + j) * 6 + 2:(i + j) * 6 + 17]
                batch_data.append((Solid_15pulse, USB_data_15pulse))
            except Exception as e:
                print(f"Error processing batch at index {i+j}: {e}")
                continue  # エラーがあればスキップ
        
        if not batch_data:
            continue  # バッチデータが空なら次へ
        
        batch_data = np.array(batch_data, dtype='object')
        X = np.stack(batch_data[:, 1]).astype(np.float32)
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])  # [バッチサイズ, チャネル数, 高さ, 幅]
        y = batch_data[:, 0].astype(np.float32)

        X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(X, y, test_size=0.2, random_state=10)
        X_train_part, X_valid_part, y_train_part, y_valid_part = train_test_split(X_train_part, y_train_part, test_size=0.2, random_state=10)

        X_train_part = torch.from_numpy(X_train_part).to(torch.float32)
        X_valid_part = torch.from_numpy(X_valid_part).to(torch.float32)
        X_test_part = torch.from_numpy(X_test_part).to(torch.float32)
        y_train_part = torch.from_numpy(y_train_part).to(torch.float32)
        y_valid_part = torch.from_numpy(y_valid_part).to(torch.float32)
        y_test_part = torch.from_numpy(y_test_part).to(torch.float32)

        # メモリ節約のためデータを即座に結合
        X_train = torch.cat((X_train, X_train_part), dim=0)
        X_valid = torch.cat((X_valid, X_valid_part), dim=0)
        X_test = torch.cat((X_test, X_test_part), dim=0)
        y_train = torch.cat((y_train, y_train_part), dim=0)
        y_valid = torch.cat((y_valid, y_valid_part), dim=0)
        y_test = torch.cat((y_test, y_test_part), dim=0)

        # 一時変数を解放
        del X_train_part, X_test_part, X_valid_part, y_train_part, y_test_part, y_valid_part, batch_data
    gc.collect()
    log_memory_usage()


# データローダーを作成
dataset_train = TensorDataset(X_train, y_train)
dataset_valid = TensorDataset(X_valid, y_valid)
dataset_test = TensorDataset(X_test, y_test)


# データセットのテンソルを保存
torch.save(dataset_train.tensors, os.path.join(path_in,'15pulse/dataset_train_1121.pt'))
torch.save(dataset_valid.tensors,  os.path.join(path_in,'15pulse/dataset_valid_1121.pt'))
torch.save(dataset_test.tensors,  os.path.join(path_in,'15pulse/dataset_test_1121.pt'))
