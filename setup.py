#!/usr/bin/env python3
"""
TestPhotos Setup Script

Quick setup for TestPhotos historical photo analysis tool.
Run this after installing Python 3.8+ and pip.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False


def check_python():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "sample_photos",
        "main_app_outputs",
        "main_app_outputs/results",
        "main_app_outputs/colorized",
        "main_app_outputs/data_search_dashord"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True


def install_dependencies():
    """Install Python dependencies."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )


def create_env_file():
    """Create .env file template if it doesn't exist."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    env_template = """# TestPhotos Environment Configuration
# Optional: Add your Groq API key for LLM analysis
# GROQ_API_KEY=your_groq_api_key_here

# Optional: Custom paths (usually auto-detected)
# TESSERACT_CMD=/usr/bin/tesseract
"""
    
    try:
        env_file.write_text(env_template)
        print("✅ Created .env file template")
        return True
    except Exception as e:
        print(f"❌ Could not create .env file: {e}")
        return False


def test_installation():
    """Test that the installation works."""
    print("🧪 Testing installation...")
    
    # Test import
    try:
        sys.path.insert(0, str(Path(__file__).parent / "main_app"))
        from main_pipeline import PhotoSelector
        print("✅ Main pipeline imports successfully")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    # Test run.py
    if not Path("run.py").exists():
        print("❌ run.py not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, "run.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if "TestPhotos" in result.stdout or result.returncode == 0:
            print("✅ run.py executes successfully")
            return True
    except Exception:
        pass
    
    print("❌ run.py test failed")
    return False


def main():
    """Main setup process."""
    print("🖼️  TestPhotos - Setup & Installation")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    # Install dependencies
    if not install_dependencies():
        success = False
        print("💡 Try running: pip install --upgrade pip")
        print("💡 Or try: pip install torch torchvision ultralytics opencv-python")
    
    # Create .env file
    if not create_env_file():
        success = False
    
    # Test installation
    if success and not test_installation():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📖 Next steps:")
        print("1. Add photos to sample_photos/ folder")
        print("2. Run: python run.py process")
        print("3. Run: python run.py dashboard")
        print("\n💡 For help: python run.py")
        
        # Show sample photo count
        sample_dir = Path("sample_photos")
        if sample_dir.exists():
            photo_count = len([f for f in sample_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            if photo_count > 0:
                print(f"\n📸 Found {photo_count} photos ready to analyze!")
            else:
                print(f"\n📸 Add photos to {sample_dir}/ to get started")
    else:
        print("❌ Setup encountered errors")
        print("\n🔧 Troubleshooting:")
        print("• Make sure you have Python 3.8+")
        print("• Run: pip install --upgrade pip")
        print("• Check internet connection for downloading packages")
        print("• Try: python -m pip install torch torchvision ultralytics")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())