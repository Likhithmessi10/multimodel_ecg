import os
import ast
import numpy as np
import pandas as pd
from scipy import signal
import wfdb
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

class PTBXLZeroLeakageLoader:
    def __init__(self, data_dir="./data", fs=100, cache_dir="./data/cache"):
        self.data_dir = data_dir
        self.fs = fs
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        self.metadata_cols = ['age', 'sex', 'height', 'weight']
        
        self.db_csv_path = os.path.join(data_dir, "ptbxl_database.csv")
        self.scp_csv_path = os.path.join(data_dir, "scp_statements.csv")
        
        # FIR Bandpass Filter coefficients (0.5 to 45 Hz for 100 Hz sampling rate)
        self.b_fir = self._design_fir_filter()
        
    def _design_fir_filter(self):
        # Nyquist frequency is Fs / 2 = 50 Hz.
        # Passband: 0.5 Hz to 45 Hz
        nyq = 0.5 * self.fs
        numtaps = 101 # Odd number for Type I linear phase FIR
        # 45 Hz is 0.9 * Nyquist, 0.5 Hz is 0.01 * Nyquist
        b = signal.firwin(numtaps, [0.5, 45.0], pass_zero='bandpass', fs=self.fs)
        return b
        
    def _apply_filter(self, data):
        # data has shape (time, channels)
        # Apply zero-phase filtering along time axis (axis=0)
        return signal.filtfilt(self.b_fir, [1.0], data, axis=0)

    def load_raw_metadata(self):
        if not os.path.exists(self.db_csv_path) or not os.path.exists(self.scp_csv_path):
            raise FileNotFoundError(
                f"PTB-XL database files not found in {self.data_dir}. "
                "Please run download_data.py first."
            )
            
        df = pd.read_csv(self.db_csv_path, index_col='ecg_id')
        df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
        
        scp_statements = pd.read_csv(self.scp_csv_path, index_col=0)
        # Map codes to superclasses
        diag_map = scp_statements[scp_statements.diagnostic_class.notnull()]['diagnostic_class'].to_dict()
        
        def aggregate_diagnostic(scp_dict):
            classes = set()
            for key in scp_dict.keys():
                if key in diag_map:
                    c = diag_map[key]
                    if c in self.superclasses:
                        classes.add(c)
            return list(classes)
            
        df['diagnostic_superclass'] = df['scp_codes'].apply(aggregate_diagnostic)
        
        # Prepare multi-label targets
        for cls in self.superclasses:
            df[cls] = df['diagnostic_superclass'].apply(lambda x: 1 if cls in x else 0)
            
        return df

    def preprocess_metadata(self, df):
        # We need age, sex, weight, height.
        # Check sex column format. In PTB-XL, it is 0 (male) or 1 (female), or strings.
        # Let's make sure it is numeric.
        if df['sex'].dtype == object:
            df['sex'] = df['sex'].map({'Male': 0, 'Female': 1}).fillna(0).astype(int)
        else:
            df['sex'] = df['sex'].fillna(0).astype(int)
            
        # Stratified split: Train (1-8), Val (9), Test (10)
        train_mask = df['strat_fold'].isin(range(1, 9))
        val_mask = df['strat_fold'] == 9
        test_mask = df['strat_fold'] == 10
        
        # Extract features
        X_meta = df[self.metadata_cols].copy()
        
        # Imputation values calculated ONLY on train split to prevent leakage
        impute_values = {}
        for col in ['age', 'height', 'weight']:
            median_val = X_meta.loc[train_mask, col].median()
            # If median is nan (should not happen if there is data), fallback to 0 or mean
            if pd.isna(median_val):
                median_val = 0.0
            impute_values[col] = median_val
            
        # Impute missing values
        for col in ['age', 'height', 'weight']:
            X_meta[col] = X_meta[col].fillna(impute_values[col])
            
        # Fit and transform ONLY on training folds to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_meta.loc[train_mask])
        
        # Apply strict transform to Validation (Fold 9) and Testing (Fold 10)
        X_val_scaled = scaler.transform(X_meta.loc[val_mask])
        X_test_scaled = scaler.transform(X_meta.loc[test_mask])
        
        # Save scaler and impute values for inference/app use
        joblib.dump(scaler, os.path.join(self.cache_dir, "metadata_scaler.pkl"))
        joblib.dump(impute_values, os.path.join(self.cache_dir, "metadata_impute_values.pkl"))
        
        # Re-assemble scaled metadata dataframe
        X_meta_scaled = pd.DataFrame(index=X_meta.index, columns=self.metadata_cols)
        X_meta_scaled.loc[train_mask] = X_train_scaled
        X_meta_scaled.loc[val_mask] = X_val_scaled
        X_meta_scaled.loc[test_mask] = X_test_scaled
        
        # Combine back
        df_processed = df.copy()
        for col in self.metadata_cols:
            df_processed[col] = X_meta_scaled[col].astype(np.float32)
            
        return df_processed, scaler, impute_values

    def load_and_preprocess_waveforms(self, df, subset_size=None):
        """
        Loads, filters, and returns waveforms for the provided dataframe.
        Saves filtered waveforms to a binary cache for fast loading.
        """
        # Filter metadata for files that actually exist on disk
        existing_mask = []
        for ecg_id, row in df.iterrows():
            file_path = os.path.join(self.data_dir, row['filename_lr'] + ".dat")
            existing_mask.append(os.path.exists(file_path))
        df = df[existing_mask]
        
        if len(df) == 0:
            raise FileNotFoundError(f"No waveform files found in {self.data_dir}. Please run download_data.py first.")
            
        if subset_size is not None and subset_size < len(df):
            # Maintain stratified ratio across folds if subsetting
            df_subset = []
            for fold in range(1, 11):
                fold_df = df[df['strat_fold'] == fold]
                n_samples = max(1, int(subset_size * len(fold_df) / len(df)))
                df_subset.append(fold_df.head(n_samples))
            df = pd.concat(df_subset)
            
        cache_file = os.path.join(self.cache_dir, f"filtered_waveforms_fs{self.fs}_{len(df)}.npy")
        cache_meta_file = os.path.join(self.cache_dir, f"filtered_meta_fs{self.fs}_{len(df)}.csv")
        
        if os.path.exists(cache_file) and os.path.exists(cache_meta_file):
            print(f"Loading filtered waveforms from cache: {cache_file}")
            waveforms = np.load(cache_file)
            meta_df = pd.read_csv(cache_meta_file, index_col='ecg_id')
            return waveforms, meta_df
            
        print("Preprocessing waveforms. This might take a few minutes for the first run...")
        waveforms_list = []
        valid_indices = []
        
        for ecg_id, row in tqdm(df.iterrows(), total=len(df), desc="Loading and Filtering ECGs"):
            # Load waveform using wfdb.
            # filename_lr points to the 100Hz path, e.g., 'records100/00000/00001_lr'
            file_path = os.path.join(self.data_dir, row['filename_lr'])
            
            try:
                # Read signal
                # wfdb.rdsamp returns a tuple: (signal_data, fields_dict)
                # signal_data has shape (1000, 12) for 10s at 100Hz
                signal_data, fields = wfdb.rdsamp(file_path)
                
                # Check for nan values in signal and impute
                if np.isnan(signal_data).any():
                    signal_data = np.nan_to_num(signal_data, nan=0.0)
                
                # Apply bandpass FIR filter (zero-phase)
                filtered_signal = self._apply_filter(signal_data)
                
                # Reshape to (12, 1000) for PyTorch 1D CNN (channels, length)
                filtered_signal = filtered_signal.T
                
                waveforms_list.append(filtered_signal)
                valid_indices.append(ecg_id)
            except Exception as e:
                print(f"Error loading ecg_id {ecg_id} at {file_path}: {e}")
                continue
                
        waveforms = np.stack(waveforms_list, axis=0) # Shape: (N, 12, 1000)
        meta_df = df.loc[valid_indices].copy()
        
        # Save to cache
        np.save(cache_file, waveforms)
        meta_df.to_csv(cache_meta_file)
        print("Caching completed.")
        
        return waveforms, meta_df

    def get_data_splits(self, subset_size=None):
        """
        Main function to get train, validation, and test sets.
        Returns:
            X_train_wave, X_train_meta, y_train
            X_val_wave, X_val_meta, y_val
            X_test_wave, X_test_meta, y_test
        """
        df_raw = self.load_raw_metadata()
        df_processed, scaler, impute_values = self.preprocess_metadata(df_raw)
        
        waveforms, df_final = self.load_and_preprocess_waveforms(df_processed, subset_size=subset_size)
        
        train_mask = df_final['strat_fold'].isin(range(1, 9))
        val_mask = df_final['strat_fold'] == 9
        test_mask = df_final['strat_fold'] == 10
        
        # Target labels
        y = df_final[self.superclasses].values.astype(np.float32)
        
        # Meta features
        X_meta = df_final[self.metadata_cols].values.astype(np.float32)
        
        # Waveforms splits
        X_train_wave = waveforms[train_mask]
        X_val_wave = waveforms[val_mask]
        X_test_wave = waveforms[test_mask]
        
        # Meta splits
        X_train_meta = X_meta[train_mask]
        X_val_meta = X_meta[val_mask]
        X_test_meta = X_meta[test_mask]
        
        # Target splits
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        
        return (
            (X_train_wave, X_train_meta, y_train),
            (X_val_wave, X_val_meta, y_val),
            (X_test_wave, X_test_meta, y_test)
        )
