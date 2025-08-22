#!/usr/bin/env python3
"""
Test script for the Agentic AI Image Authenticity Detector
Demonstrates basic functionality and validates system setup.
"""

import os
import sys
import json
from dotenv import load_dotenv
from main import ImageAuthenticityDetector

def test_system_initialization():
    """Test that the system can be initialized properly."""
    print("Testing system initialization...")
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not found in environment variables")
            print("Please copy env_template.txt to .env and add your API key")
            return False
        
        # Initialize detector
        config = {
            'log_level': 'WARNING',  # Reduce noise during testing
            'agent_settings': {
                'counterfactual': {'enabled': False},  # Skip for faster testing
                'retriever': {'enabled': False}        # Skip for faster testing
            }
        }
        
        detector = ImageAuthenticityDetector(config)
        print("System initialized successfully")
        return True
        
    except Exception as e:
        print(f"System initialization failed: {str(e)}")
        return False

def test_agent_creation():
    """Test that all agents can be created."""
    print("\nTesting agent creation...")
    
    try:
        from agents.planner_agent import PlannerAgent
        from agents.classifier_agent import ClassifierAgent
        from agents.forensics_agent import ForensicsAgent
        from agents.explainer_agent import ExplainerAgent
        from agents.reporter_agent import ReporterAgent
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Test each agent
        agents = {
            'Planner': PlannerAgent,
            'Classifier': ClassifierAgent, 
            'Forensics': ForensicsAgent,
            'Explainer': ExplainerAgent,
            'Reporter': ReporterAgent
        }
        
        for name, agent_class in agents.items():
            try:
                agent = agent_class(api_key)
                print(f"{name}Agent created successfully")
            except Exception as e:
                print(f"{name}Agent creation failed: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Agent creation test failed: {str(e)}")
        return False

def test_tools_import():
    """Test that all tool modules can be imported."""
    print("\nTesting tool imports...")
    
    try:
        from tools.vision_tools import VisionAuthenticityDetector, ImagePreprocessor
        from tools.metadata_tools import MetadataForensicsAnalyzer
        from tools.search_tools import ImageSearchEngine
        from tools.report_tools import AuthenticityReportGenerator
        
        print("Vision tools imported successfully")
        print("Metadata tools imported successfully")
        print("Search tools imported successfully")
        print("Report tools imported successfully")
        
        return True
        
    except Exception as e:
        print(f"Tool import failed: {str(e)}")
        return False

def create_test_image():
    """Create a simple test image for analysis."""
    print("\nCreating test image...")
    
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a simple test image
        width, height = 400, 300
        image = Image.new('RGB', (width, height), color='lightblue')
        
        # Add some simple content
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 350, 250], outline='darkblue', width=3)
        draw.text((100, 150), "TEST IMAGE", fill='darkblue')
        
        # Save test image
        test_image_path = os.path.join('data', 'test_image.png')
        os.makedirs('data', exist_ok=True)
        image.save(test_image_path)
        
        print(f"Test image created: {test_image_path}")
        return test_image_path
        
    except Exception as e:
        print(f"Test image creation failed: {str(e)}")
        return None

def test_basic_analysis(image_path):
    """Test basic analysis functionality."""
    print(f"\nTesting basic analysis with: {image_path}")
    
    try:
        # Initialize detector with minimal config for testing
        config = {
            'log_level': 'ERROR',  # Minimize output
            'agent_settings': {
                'counterfactual': {'enabled': False},
                'retriever': {'enabled': False}
            }
        }
        
        detector = ImageAuthenticityDetector(config)
        
        print("Running analysis (this may take a few minutes)...")
        
        # Run analysis
        results = detector.analyze_image(image_path)
        
        if results.get('error'):
            print(f"Analysis failed: {results['error']}")
            return False
        
        # Check results
        if results.get('analysis_completed'):
            print("Analysis completed successfully")
            
            # Get summary
            summary = detector.get_analysis_summary()
            print(f"   Overall Score: {summary.get('overall_authenticity_score', 0):.2%}")
            print(f"   Assessment: {summary.get('authenticity_assessment', 'Unknown')}")
            print(f"   Duration: {summary.get('analysis_duration', 0):.1f} seconds")
            
            if summary.get('report_path'):
                print(f"   Report: {summary['report_path']}")
            
            return True
        else:
            print("Analysis did not complete properly")
            return False
        
    except Exception as e:
        print(f"Analysis test failed: {str(e)}")
        return False

def check_dependencies():
    """Check that required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'openai', 'crewai', 'langchain', 'torch', 'transformers', 
        'pillow', 'opencv-python', 'reportlab', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"{package}")
        except ImportError:
            print(f"{package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("All required packages found")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("AGENTIC AI IMAGE AUTHENTICITY DETECTOR - SYSTEM TEST")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Dependencies
    total_tests += 1
    if check_dependencies():
        tests_passed += 1
    
    # Test 2: System initialization
    total_tests += 1
    if test_system_initialization():
        tests_passed += 1
    else:
        print("\nCannot continue without proper initialization")
        return 1
    
    # Test 3: Agent creation
    total_tests += 1
    if test_agent_creation():
        tests_passed += 1
    
    # Test 4: Tool imports
    total_tests += 1
    if test_tools_import():
        tests_passed += 1
    
    # Test 5: Create test image
    test_image_path = create_test_image()
    if test_image_path:
        # Test 6: Basic analysis
        total_tests += 1
        if test_basic_analysis(test_image_path):
            tests_passed += 1
    else:
        total_tests += 1  # Count as a failed test
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("ALL TESTS PASSED - System is ready!")
        print("\nNext steps:")
        print("1. Add a real image to analyze: python main.py your_image.jpg")
        print("2. Check the generated report in the reports/ directory")
        print("3. Review logs in data/authenticity_analysis.log")
        return 0
    else:
        print("Some tests failed - please check the errors above")
        print("\nCommon fixes:")
        print("1. Ensure OPENAI_API_KEY is set in .env file")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Check Python version (3.8+ required)")
        return 1

if __name__ == "__main__":
    exit(main())
