import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import pandas as pd
import re
import gc
import pynvml


#Filtered_USB_Data_shape:(14994, 4012)
#Filtered_USB_Data_shape:(2499, 4012) (6ずつ増やしたとき)
r"""
def memoricount():

    torch.cuda.empty_cache()
    gc.collect()  # ガベージコレクタを明示的に呼び出す
    # NVMLの初期化
    pynvml.nvmlInit()
    # 各GPUのメモリ情報を取得して表示
    for i in range(1):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total / 1024**3  # GB単位
        used_memory = memory_info.used / 1024**3    # GB単位
        free_memory = memory_info.free / 1024**3    # GB単位
        #print(f"GPU {i}:")
        #print(f"  Total Memory: {total_memory:.2f} GB")
        print(f"  Used Memory: {used_memory:.2f} GB")
        #print(f"  Free Memory: {free_memory:.2f} GB")
    # NVMLの終了
    pynvml.nvmlShutdown()

def memoricount():
    if torch.cuda.is_available():
        # 現在使用しているGPUのインデックスを取得
        #device = torch.cuda.current_device()
        
        # 現在のGPUのメモリ使用量を取得（MB単位）
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        reserved_memory = torch.cuda.memory_reserved(device) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(device) / 1024**3

        # 結果を表示
        print(f"GPU {device}:")
        print(f"  Allocated Memory: {allocated_memory:.2f} GB")
        print(f"  Reserved Memory: {reserved_memory:.2f} GB")
        print(f"  Max Allocated Memory: {max_memory:.2f} GB")
    else:

        print("CUDA is not available.")
"""


input_camera_files = r"/mnt/sdb/ywatanabe/csv_files/camera_data/volume_fraction_210250_40_8_25_30_7_22/*"
input_ultrasonic_files = r"/mnt/sdb/ywatanabe/csv_files/2023_2dim_hilbert"
TDX = "_tdx1_trans"
pattern = re.compile(r"s[0-9]+[-_]g0+[-_]l[0-9]+[-_]t[0-9]+")  # 固液二相流のみに行う。
#pattern = re.compile(r"s1[-_]g0+[-_]l1+[-_]t[1]")  # 固液二相流のみに行う。

camera_files = [f for f in glob.glob(input_camera_files) if pattern.search(f)]
print(len(camera_files))

camera_files = sorted(camera_files, key=lambda x: int(re.search(r"s([0-9]+)", x).group(1)), reverse=False)

# 修正：空のテンソルで初期化
X_train = torch.empty((0, 1, 1, 3112), dtype=torch.float32)         # [バッチサイズ, チャネル数, 高さ, 幅] に
X_valid = torch.empty((0, 1, 1, 3112), dtype=torch.float32)
X_test = torch.empty((0, 1, 1, 3112), dtype=torch.float32)
y_train = torch.empty((0,), dtype=torch.float32)
y_valid = torch.empty((0,), dtype=torch.float32)
y_test = torch.empty((0,), dtype=torch.float32)

for camera_folder in camera_files:
    data = []
    print("basename:"+os.path.basename(camera_folder))
    basename=os.path.basename(camera_folder)
    basename=basename.replace(".csv","")#s1_g0_l1_t1
    filepath_ultrasonic = os.path.join(input_ultrasonic_files+TDX,basename+TDX+".csv")    
    USB_data = pd.read_csv(filepath_ultrasonic,header=None,usecols=range(900, 4012))
    Solid_fraction_data = pd.read_csv(camera_folder,header=None)
    print("datalength: "+str(USB_data.shape))

    if USB_data.shape[0]<14986:
        print("datalength: "+str(USB_data.shape[0]))
        print("not satisfied file: basename"+basename)
        continue

    for i in range(2497):
        Solid_15pulse = Solid_fraction_data.iloc[i+1, 0]
        USB_data_15pulse=USB_data.iloc[i*6+9]
        data.append((Solid_15pulse,USB_data_15pulse))   

    data = np.array(data,dtype='object')
    X = np.stack(data[:, 1]).astype(np.float32) 
    X = X.reshape(X.shape[0], 1, 1, X.shape[1])  # 次元を追加して [バッチサイズ, チャネル数, 高さ, 幅] に

    y = data[:, 0].astype(np.float32) # CrossEntropyLoss には int64 が必要

    X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train_part, X_valid_part, y_train_part, y_valid_part = train_test_split(X_train_part, y_train_part, test_size=0.2, random_state=0)
    X_train_part = torch.from_numpy(X_train_part).to(torch.float32)
    X_valid_part = torch.from_numpy(X_valid_part).to(torch.float32)
    X_test_part = torch.from_numpy(X_test_part).to(torch.float32)
    y_train_part = torch.from_numpy(y_train_part).to(torch.float32)
    y_valid_part = torch.from_numpy(y_valid_part).to(torch.float32)
    y_test_part = torch.from_numpy(y_test_part).to(torch.float32)
    
    X_train =  torch.cat((X_train, X_train_part), dim=0)
    X_valid =  torch.cat((X_valid, X_valid_part), dim=0)
    X_test  =  torch.cat((X_test , X_test_part), dim=0)
    y_train =  torch.cat((y_train, y_train_part), dim=0)
    y_valid =  torch.cat((y_valid, y_valid_part), dim=0)
    y_test  =  torch.cat((y_test , y_test_part), dim=0)
    del X_train_part,X_test_part,X_valid_part,y_train_part,y_test_part,y_valid_part

#データローダーを作成
dataset_train = TensorDataset(X_train, y_train)
dataset_valid = TensorDataset(X_valid,y_valid)
dataset_test = TensorDataset(X_test,y_test)

# データセットのテンソルを保存
save_path='/mnt/sdb/ywatanabe/CNN_dataset/transwave/1pulse'
torch.save(dataset_train.tensors, os.path.join(save_path,'dataset_train_210250.pt'))
torch.save(dataset_valid.tensors,os.path.join(save_path,'dataset_valid_210250.pt'))
torch.save(dataset_test.tensors, os.path.join(save_path,'dataset_test_210250.pt'))
