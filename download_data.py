import os
import argparse
import urllib.request
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path, desc=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if desc is None:
        desc = os.path.basename(output_path)
    try:
        # Use simple retrieve for CSVs with progress bar
        urllib.request.urlretrieve(url, filename=output_path)
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False

def download_full_zip(data_dir):
    import zipfile
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    zip_filename = "ptb_xl.zip"
    zip_filepath = os.path.join(data_dir, zip_filename)
    
    print(f"Downloading full PTB-XL dataset (1.84 GB) from {url}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="ptb_xl.zip") as t:
            urllib.request.urlretrieve(url, filename=zip_filepath, reporthook=t.update_to)
        
        print("Download complete. Extracting 100Hz records and metadata...")
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            target_files = []
            for file in all_files:
                parts = file.split('/')
                if len(parts) > 1:
                    subpath = "/".join(parts[1:])
                    if subpath in ["ptbxl_database.csv", "scp_statements.csv"] or subpath.startswith("records100/"):
                        target_files.append(file)
            
            print(f"Found {len(target_files)} relevant files to extract out of {len(all_files)} total.")
            for file in tqdm(target_files, desc="Extracting"):
                parts = file.split('/')
                subpath = "/".join(parts[1:])
                if not subpath:
                    continue
                dest_path = os.path.join(data_dir, subpath)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                if not file.endswith('/'):
                    with zip_ref.open(file) as source, open(dest_path, "wb") as target:
                        target.write(source.read())
        print("Extraction complete. Cleaning up zip file...")
        try:
            os.remove(zip_filepath)
        except Exception as e:
            print(f"Error removing zip: {e}")
        print("Dataset setup successful.")
    except Exception as e:
        print(f"Full download failed: {e}")

def download_single_file(url_path_pair):
    url, path = url_path_pair
    if os.path.exists(path):
        return True
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, filename=path)
        return True
    except Exception as e:
        # Retry once
        try:
            urllib.request.urlretrieve(url, filename=path)
            return True
        except:
            return False

def download_light_dataset(data_dir, num_records, max_workers=20):
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    
    # 1. Download metadata CSVs
    db_csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    scp_csv_path = os.path.join(data_dir, "scp_statements.csv")
    
    print("Downloading metadata CSVs...")
    if not os.path.exists(db_csv_path):
        urllib.request.urlretrieve(base_url + "ptbxl_database.csv", db_csv_path)
    else:
        print("ptbxl_database.csv already exists.")
        
    if not os.path.exists(scp_csv_path):
        urllib.request.urlretrieve(base_url + "scp_statements.csv", scp_csv_path)
    else:
        print("scp_statements.csv already exists.")
        
    # Read database CSV
    df = pd.read_csv(db_csv_path)
    print(f"Total database contains {len(df)} records.")
    
    # Select records stratified by fold to get an even split
    df_sorted = df.sort_values(by=['strat_fold', 'ecg_id'])
    
    records_per_fold = num_records // 10
    subset_dfs = []
    for fold in range(1, 11):
        fold_df = df_sorted[df_sorted['strat_fold'] == fold]
        subset_dfs.append(fold_df.head(records_per_fold))
    
    subset_df = pd.concat(subset_dfs)
    print(f"Selected {len(subset_df)} records stratified across Folds 1-10.")
    
    # Prepare download tasks
    tasks = []
    for _, row in subset_df.iterrows():
        rel_path = row['filename_lr']
        tasks.append((base_url + rel_path + ".dat", os.path.join(data_dir, rel_path + ".dat")))
        tasks.append((base_url + rel_path + ".hea", os.path.join(data_dir, rel_path + ".hea")))
        
    print(f"Downloading {len(tasks)} files in parallel using {max_workers} threads...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(download_single_file, task): task for task in tasks}
        
        # Monitor progress
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Downloading Waveforms"):
            if future.result():
                success_count += 1
                
    print(f"\nSuccessfully downloaded {success_count} / {len(tasks)} files.")
    print("Dataset setup in light mode complete.")

def main():
    parser = argparse.ArgumentParser(description="PTB-XL ECG Dataset Downloader")
    parser.add_argument("--full", action="store_true", help="Download the full 1.84 GB dataset (takes longer)")
    parser.add_argument("--num_records", type=int, default=1000, 
                        help="Number of records to download in light mode (default: 1000)")
    parser.add_argument("--max_workers", type=int, default=25, help="Number of concurrent download threads")
    parser.add_argument("--data_dir", type=str, default="./data", help="Target directory for dataset")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.full:
        download_full_zip(args.data_dir)
    else:
        print(f"Running in LIGHT mode (downloading metadata + {args.num_records} stratified records).")
        print("To download the full 1.84 GB dataset, run: python download_data.py --full")
        download_light_dataset(args.data_dir, args.num_records, max_workers=args.max_workers)

if __name__ == "__main__":
    main()
