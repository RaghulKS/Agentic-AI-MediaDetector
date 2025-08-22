#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    if sys.version_info < (3, 8):
        print("Python 3.8+ required")
        print(f"Current: {sys.version.split()[0]}")
        return False
    print(f"Python {sys.version.split()[0]}")
    return True

def setup_virtual_environment():
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Already in venv")
        return True
    
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Created venv")
        print("Activate with:")
        if os.name == 'nt':
            print("  venv\\Scripts\\activate")
        else:
            print("  source venv/bin/activate")
        print("Then run setup again.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Venv creation failed: {e}")
        return False

def install_dependencies():
    print("Installing packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Packages installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Install failed: {e}")
        return False

def setup_environment_file():
    """Set up environment configuration file."""
    env_file = ".env"
    template_file = "env_template.txt"
    
    if os.path.exists(env_file):
        print(".env file already exists")
        return True
    
    if os.path.exists(template_file):
        try:
            shutil.copy(template_file, env_file)
            print("Created .env file from template")
            print("IMPORTANT: Edit .env file and add your OPENAI_API_KEY")
            return True
        except Exception as e:
            print(f"Failed to create .env file: {e}")
            return False
    else:
        print("Environment template file not found")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["data", "reports", ".cache"]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Failed to create directory {directory}: {e}")
            return False
    
    return True

def check_api_key():
    """Check if OpenAI API key is configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            print("OpenAI API key is configured")
            return True
        else:
            print("OpenAI API key not configured")
            print("Please edit .env file and add your API key")
            return False
    except ImportError:
        print("Cannot check API key (python-dotenv not installed yet)")
        return False

def run_system_test():
    """Run basic system test."""
    print("\nRunning system test...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("System test passed!")
            return True
        else:
            print("System test failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("System test timed out")
        return False
    except Exception as e:
        print(f"Failed to run system test: {e}")
        return False

def main():
    """Run the complete setup process."""
    print("=" * 60)
    print("AGENTIC AI IMAGE AUTHENTICITY DETECTOR - SETUP")
    print("=" * 60)
    
    setup_steps = [
        ("Checking Python version", check_python_version),
        ("Setting up virtual environment", setup_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment file", setup_environment_file),
        ("Creating directories", create_directories),
    ]
    
    failed_steps = []
    
    for step_name, step_function in setup_steps:
        print(f"\n{step_name}...")
        if not step_function():
            failed_steps.append(step_name)
            if step_name == "Setting up virtual environment":
                # Special case - need to exit for venv setup
                return 1
    
    # Optional steps
    print("\nOptional checks...")
    check_api_key()
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if failed_steps:
        print(f"Setup completed with issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease resolve the issues above before using the system.")
        return 1
    else:
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate virtual environment (if not already active)")
        print("2. Edit .env file and add your OPENAI_API_KEY")
        print("3. Run system test: python test_system.py")
        print("4. Analyze an image: python main.py your_image.jpg")
        print("\nFor help, see README.md")
        return 0

if __name__ == "__main__":
    exit(main())
