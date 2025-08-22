#!/usr/bin/env python3

import os
import sys

def check_file_exists(filepath):
    if os.path.exists(filepath):
        print(f"{filepath}")
        return True
    else:
        print(f"{filepath} - MISSING")
        return False

def check_directory_exists(dirpath):
    if os.path.isdir(dirpath):
        print(f"{dirpath}/")
        return True
    else:
        print(f"{dirpath}/ - MISSING")
        return False

def validate_structure():
    print("Validating project structure...")
    
    required_files = [
        "main.py",
        "requirements.txt", 
        "README.md",
        "config.json",
        "env_template.txt",
        ".gitignore"
    ]
    
    required_dirs = [
        "agents",
        "tools", 
        "data",
        "reports",
        "prompts"
    ]
    
    agent_files = [
        "agents/planner_agent.py",
        "agents/classifier_agent.py",
        "agents/forensics_agent.py",
        "agents/retriever_agent.py",
        "agents/counterfactual_agent.py",
        "agents/explainer_agent.py",
        "agents/reporter_agent.py"
    ]
    
    tool_files = [
        "tools/vision_tools.py",
        "tools/metadata_tools.py", 
        "tools/search_tools.py",
        "tools/report_tools.py"
    ]
    
    prompt_files = [
        "prompts/planner_prompt.md",
        "prompts/explainer_prompt.md",
        "prompts/reporter_prompt.md"
    ]
    
    all_valid = True
    
    print("\nCore files:")
    for f in required_files:
        if not check_file_exists(f):
            all_valid = False
    
    print("\nDirectories:")
    for d in required_dirs:
        if not check_directory_exists(d):
            all_valid = False
    
    print("\nAgent modules:")
    for f in agent_files:
        if not check_file_exists(f):
            all_valid = False
    
    print("\nTool modules:")
    for f in tool_files:
        if not check_file_exists(f):
            all_valid = False
    
    print("\nPrompt templates:")
    for f in prompt_files:
        if not check_file_exists(f):
            all_valid = False
    
    return all_valid

def check_python_syntax():
    print("\nChecking Python syntax...")
    
    python_files = []
    
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), py_file, 'exec')
            print(f"{py_file}")
        except SyntaxError as e:
            print(f"{py_file} - Syntax Error: {e}")
            syntax_errors.append(py_file)
        except Exception as e:
            print(f"{py_file} - Warning: {e}")
    
    return len(syntax_errors) == 0

def main():
    print("=" * 60)
    print("PROJECT STRUCTURE VALIDATION")
    print("=" * 60)
    
    structure_valid = validate_structure()
    syntax_valid = check_python_syntax()
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    if structure_valid and syntax_valid:
        print("Project structure is complete and valid!")
        print("\nNext steps:")
        print("1. Set up virtual environment: python -m venv venv")
        print("2. Activate: venv\\Scripts\\activate (Windows) or source venv/bin/activate")
        print("3. Install dependencies: pip install -r requirements.txt") 
        print("4. Configure .env file with your OpenAI API key")
        print("5. Run: python main.py your_image.jpg")
        return 0
    else:
        if not structure_valid:
            print("Project structure incomplete")
        if not syntax_valid:
            print("Python syntax errors found")
        return 1

if __name__ == "__main__":
    exit(main())
