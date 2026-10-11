# preprocessing/automate_Maya.py
"""
Script otomatis untuk preprocessing data heart disease.
Mengubah raw data menjadi data siap training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

def load_data(file_path):
    """
    Load dataset dari file CSV
    
    Parameters:
    file_path (str): Path ke file CSV
    
    Returns:
    DataFrame: Dataset yang sudah diload
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data berhasil diload: {df.shape[0]} baris, {df.shape[1]} kolom")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def explore_data(df):
    """
    Melakukan EDA sederhana (opsional, untuk logging)
    
    Parameters:
    df (DataFrame): Dataset
    """
    print("\nEDA Summary:")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")

def preprocess_data(df):
    """
    Melakukan preprocessing:
    1. Menghapus duplikat
    2. Memisahkan fitur dan target
    3. Standardisasi fitur numerik
    
    Parameters:
    df (DataFrame): Raw dataset
    
    Returns:
    DataFrame: Data siap training
    dict: Informasi preprocessing
    """
    print("\nStarting Preprocessing...")
    
    # 1. Hapus duplikat
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = initial_shape - df.shape[0]
    print(f"Menghapus {duplicates_removed} baris duplikat")
    
    # 2. Pisahkan fitur dan target
    X = df.drop("target", axis=1)
    y = df["target"]
    print(f"Fitur: {list(X.columns)}")
    print(f"Target: target")
    
    # 3. Standardisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Standardisasi selesai (mean=0, std=1)")
    
    # 4. Gabungkan kembali
    df_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    df_preprocessed["target"] = y.values
    
    print(f"Preprocessing selesai! Shape akhir: {df_preprocessed.shape}")
    
    preprocessing_info = {
        "duplicates_removed": duplicates_removed,
        "features": list(X.columns),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }
    
    return df_preprocessed, preprocessing_info

def save_data(df, output_path):
    """
    Menyimpan data hasil preprocessing
    
    Parameters:
    df (DataFrame): Data siap training
    output_path (str): Path penyimpanan
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"\nData tersimpan di: {output_path}")
        print(f"   Ukuran: {df.shape[0]} baris x {df.shape[1]} kolom")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)

def main():
    """
    Main function untuk menjalankan preprocessing otomatis
    """
    print("=" * 50)
    print("AUTOMATIC PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Tentukan path 
    input_path = "heart.csv"  # Raw data di root folder
    output_path = "heart_preprocessing.csv"  # Output di root folder
    
    # Cek apakah file input ada
    if not os.path.exists(input_path):
        print(f"File tidak ditemukan: {input_path}")
        print("Pastikan heart.csv berada di folder utama (satu level di atas preprocessing/)")
        sys.exit(1)
    
    # Eksekusi pipeline
    df_raw = load_data(input_path)
    explore_data(df_raw)
    df_clean, preprocess_info = preprocess_data(df_raw)
    save_data(df_clean, output_path)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    return df_clean, preprocess_info

if __name__ == "__main__":
    df_final, info = main()