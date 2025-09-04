# 🪟 Windows Setup Guide for TestPhotos
## Complete Guide for Non-Technical Users with AI Assistant

This guide will help you set up TestPhotos on Windows even if you've never worked with code before. We'll use AI assistants (like ChatGPT, Claude, or Cursor) to help you every step of the way.

---

## 📋 What You'll Need

- **Windows 10 or 11** (any edition)
- **Internet connection** (for downloads)
- **At least 8GB RAM** (4GB minimum)
- **5GB free disk space**
- **Your historical photos** (JPG, PNG format)

---

## 🤖 Method 1: Using Cursor IDE (Recommended for Beginners)

### Step 1: Download and Install Cursor
1. **Go to**: https://cursor.sh
2. **Click**: "Download for Windows"
3. **Run the installer** when download completes
4. **Follow the installation wizard** (click Next → Next → Install)

### Step 2: Install Git (Required)
1. **Go to**: https://git-scm.com/download/win
2. **Click**: "64-bit Git for Windows Setup"
3. **Run the installer** with default settings (keep clicking Next)
4. **Important**: When asked about "Adjusting your PATH environment", choose "Git from the command line and also from 3rd-party software"

### Step 3: Install Python
1. **Go to**: https://www.python.org/downloads/
2. **Click**: "Download Python 3.11.x" (latest version)
3. **Run the installer**
4. **⚠️ IMPORTANT**: Check "Add Python to PATH" before clicking Install
5. **Click**: "Install Now"

### Step 4: Get TestPhotos with Cursor
1. **Open Cursor**
2. **Press**: `Ctrl + Shift + P`
3. **Type**: "Git: Clone"
4. **Paste this URL**: `https://github.com/[your-username]/TestPhotos.git`
5. **Choose a folder** where you want TestPhotos (like Desktop or Documents)
6. **Click**: "Open" when it finishes downloading

### Step 5: Let AI Set Everything Up
1. **In Cursor, press**: `Ctrl + L` (opens AI chat)
2. **Copy and paste this message**:

```
I just downloaded TestPhotos on Windows. I'm not technical. Please help me:

1. Install all the Python requirements 
2. Set up the project properly
3. Show me how to add my photos
4. Explain how to run the analysis
5. Walk me through using the search dashboard

Be very detailed and assume I've never used command line before. Give me exact commands to copy and paste.
```

3. **Follow the AI's instructions step by step**
4. **Ask follow-up questions** if anything is unclear

---

## 🛠️ Method 2: Manual Installation (Step-by-Step)

If you prefer to do it manually or Cursor isn't working:

### Step 1: Install Prerequisites

#### Install Git:
1. Download from: https://git-scm.com/download/win
2. Run installer with default settings
3. Open Command Prompt (Press `Windows key + R`, type `cmd`, press Enter)
4. Test by typing: `git --version`

#### Install Python:
1. Download from: https://www.python.org/downloads/
2. **CRITICAL**: Check "Add Python to PATH" during installation
3. Test by opening Command Prompt and typing: `python --version`

### Step 2: Download TestPhotos
1. **Open Command Prompt**
2. **Navigate to where you want TestPhotos**:
   ```cmd
   cd Desktop
   ```
3. **Download TestPhotos**:
   ```cmd
   git clone https://github.com/[your-username]/TestPhotos.git
   ```
4. **Enter the folder**:
   ```cmd
   cd TestPhotos
   ```

### Step 3: Install Python Packages
```cmd
pip install -r requirements.txt
```
*This will take 5-10 minutes and download about 2GB of AI models*

### Step 4: Test Installation
```cmd
python run.py --help
```
*Should show help information if everything is working*

---

## 📸 Adding Your Photos

### Method 1: Drag and Drop
1. **Open File Explorer**
2. **Navigate to your TestPhotos folder**
3. **Open the `sample_photos` folder**
4. **Drag your photos** from wherever they are into this folder
5. **Supported formats**: JPG, JPEG, PNG, GIF

### Method 2: Copy with Command
1. **Open Command Prompt in TestPhotos folder**
2. **Copy photos from another folder**:
   ```cmd
   copy "C:\Users\YourName\Pictures\OldPhotos\*.*" sample_photos\
   ```

---

## 🚀 Running TestPhotos

### Simple Method (Interactive Menu):
1. **Open Command Prompt in TestPhotos folder**
2. **Run**:
   ```cmd
   python run.py
   ```
3. **Choose option 1** to process photos
4. **Choose option 2** to search results

### Direct Commands:
```cmd
# Process all photos with AI analysis
python run.py photo_processing

# Open search dashboard
python run.py dashboard
```

---

## 🔍 Using the Search Dashboard

After processing your photos, you can search them in two ways:

### 1. Category Search
Find photos with specific characteristics:
- Photos with Nazi symbols
- Photos with Jewish symbols  
- Photos with Hebrew or German text
- Photos showing violence
- Indoor vs outdoor photos

### 2. Semantic Search
Search using natural language:
- "family photos with children"
- "military uniforms and soldiers"
- "religious ceremonies"
- "documents with German text"
- "people wearing traditional clothing"

---

## 🆘 Getting Help from AI

If you get stuck, use these prompts with any AI assistant:

### For Setup Problems:
```
I'm trying to set up TestPhotos on Windows but getting this error:
[paste the exact error message]

I'm not technical. Please give me step-by-step instructions to fix this.
```

### For Usage Questions:
```
I have TestPhotos working on Windows. How do I:
- Add more photos to analyze
- Search for specific types of photos
- Understand the analysis results
- Export or save search results

Please explain like I'm a beginner.
```

### For Technical Issues:
```
TestPhotos on Windows is giving me this problem:
[describe what's happening]

My setup:
- Windows [10 or 11]
- Python version: [run: python --version]
- Error message: [paste exact error]

Please help me debug this step by step.
```

---

## 📁 Understanding the Folder Structure

After setup, your TestPhotos folder will look like this:

```
TestPhotos/
├── sample_photos/          ← Put your photos here
├── main_app_outputs/       ← Results appear here
│   ├── colorized/          ← Colorized versions of B&W photos
│   ├── results/            ← Detailed analysis data
│   └── data_search_dashord/ ← Search results
├── run.py                  ← Main program to run
└── README.md               ← General instructions
```

---

## ⚠️ Common Windows Issues and Solutions

### "Python is not recognized"
**Problem**: You didn't add Python to PATH during installation
**Solution**: 
1. Uninstall Python
2. Reinstall and check "Add Python to PATH"
3. Restart Command Prompt

### "git is not recognized"  
**Problem**: Git not installed or not in PATH
**Solution**:
1. Install Git from https://git-scm.com/download/win
2. Choose "Git from command line" during installation
3. Restart Command Prompt

### "Permission denied" errors
**Problem**: Windows blocking file access
**Solution**:
1. Right-click Command Prompt → "Run as administrator"
2. Or move TestPhotos folder to a location you own (like Desktop)

### "Out of memory" errors
**Problem**: Not enough RAM
**Solution**:
1. Close other programs
2. Process fewer photos at once
3. Use the interactive menu to process "1 random photo" first

### Antivirus blocking downloads
**Problem**: Antivirus thinks AI models are suspicious
**Solution**:
1. Temporarily disable antivirus during installation
2. Add TestPhotos folder to antivirus exclusions
3. Re-enable antivirus after setup

---

## 🎯 Quick Start Checklist

- [ ] Install Cursor IDE
- [ ] Install Git with PATH option
- [ ] Install Python with PATH option  
- [ ] Download TestPhotos using Cursor or git clone
- [ ] Install Python packages with `pip install -r requirements.txt`
- [ ] Test with `python run.py --help`
- [ ] Add photos to `sample_photos/` folder
- [ ] Run `python run.py` and choose processing
- [ ] Use search dashboard to explore results

---

## 🤝 Getting More Help

1. **Use the AI chat in Cursor** - Press `Ctrl + L` and ask questions
2. **Ask ChatGPT/Claude** - Copy the error prompts from this guide
3. **Check the main README.md** - For general usage information
4. **Ask specific questions** like:
   - "How do I add more photos to TestPhotos?"
   - "What do these analysis results mean?"
   - "How do I search for specific types of photos?"

Remember: **There's no such thing as a stupid question when learning!** AI assistants are there to help you succeed.

---

**Good luck with your historical photo analysis! 📸🔍**