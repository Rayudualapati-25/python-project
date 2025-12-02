"""
Download NYC Taxi Dataset from Kaggle

This script helps you download the required dataset files.
You'll need a Kaggle account and API credentials.

Setup Kaggle API:
1. Create account at https://www.kaggle.com
2. Go to Account settings → API → Create New API Token
3. This downloads kaggle.json to your computer
4. Move it to: ~/.kaggle/kaggle.json (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\kaggle.json (Windows)
5. Run this script: python download_data.py
"""

import os
import sys

def check_kaggle_setup():
    """Check if Kaggle API is configured"""
    try:
        import kaggle
        print("✓ Kaggle library found")
        return True
    except ImportError:
        print("✗ Kaggle library not found")
        print("\nInstalling kaggle...")
        os.system(f"{sys.executable} -m pip install kaggle")
        try:
            import kaggle
            return True
        except ImportError:
            print("✗ Failed to install kaggle library")
            return False


def download_dataset():
    """Download NYC taxi fare dataset from Kaggle"""
    
    print("\n" + "="*60)
    print("NYC TAXI FARE DATASET DOWNLOADER")
    print("="*60 + "\n")
    
    if not check_kaggle_setup():
        return False
    
    # Check for kaggle.json
    kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_config):
        print("\n✗ Kaggle API credentials not found!")
        print("\nPlease set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Move the downloaded kaggle.json to:")
        print(f"   {kaggle_config}")
        print("5. Run this script again")
        return False
    
    print("✓ Kaggle API credentials found\n")
    
    # Create data directory
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    print(f"✓ Created directory: {data_dir}\n")
    
    # Download dataset
    print("Downloading dataset from Kaggle...")
    print("Competition: new-york-city-taxi-fare-prediction")
    print("\nThis may take several minutes (dataset is ~5GB)...\n")
    
    try:
        import kaggle
        kaggle.api.competition_download_files(
            'new-york-city-taxi-fare-prediction',
            path=data_dir
        )
        print("\n✓ Dataset downloaded successfully!")
        
        # Check if files need to be extracted
        zip_file = os.path.join(data_dir, "new-york-city-taxi-fare-prediction.zip")
        if os.path.exists(zip_file):
            print("\nExtracting files...")
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("✓ Files extracted")
            
            # Clean up zip file
            os.remove(zip_file)
            print("✓ Cleaned up zip file")
        
        # Verify files
        train_file = os.path.join(data_dir, "train.csv")
        test_file = os.path.join(data_dir, "test.csv")
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            train_size = os.path.getsize(train_file) / (1024**3)  # GB
            test_size = os.path.getsize(test_file) / (1024**3)    # GB
            
            print("\n" + "="*60)
            print("DOWNLOAD COMPLETE!")
            print("="*60)
            print(f"\n✓ train.csv ({train_size:.2f} GB)")
            print(f"✓ test.csv ({test_size:.2f} GB)")
            print(f"\nFiles saved to: {data_dir}/")
            print("\nNext steps:")
            print("  1. Run: python train.py")
            print("  2. Run: python predict.py")
            print("  3. Run: streamlit run app.py")
            return True
        else:
            print("\n✗ Expected files not found after extraction")
            return False
            
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nAlternative: Manual download")
        print("1. Visit: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data")
        print("2. Download train.csv and test.csv manually")
        print(f"3. Place them in: {data_dir}/")
        return False


if __name__ == "__main__":
    success = download_dataset()
    
    if not success:
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print("\nIf automatic download fails, follow these steps:")
        print("\n1. Visit Kaggle competition page:")
        print("   https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data")
        print("\n2. Accept competition rules (if prompted)")
        print("\n3. Download these files:")
        print("   - train.csv")
        print("   - test.csv")
        print("\n4. Move them to this directory:")
        print(f"   {os.path.abspath('data/raw')}/")
        print("\n5. Verify files are in place:")
        print("   ls data/raw/")
        print("\n6. Then run:")
        print("   python train.py")
    
    sys.exit(0 if success else 1)
