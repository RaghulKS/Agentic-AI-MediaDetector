#!/usr/bin/env python3

import os
import sys
import tempfile
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

def create_test_image():
    width, height = 512, 384
    image = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(image)
    
    draw.rectangle([50, 50, 462, 334], outline='darkblue', width=2)
    draw.text((150, 180), "TEST IMAGE", fill='darkblue')
    draw.ellipse([200, 120, 312, 232], outline='red', width=3)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        image.save(f.name)
        return f.name

def test_basic_functionality():
    print("Testing basic system functionality...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("OPENAI_API_KEY not configured")
        return False
    
    test_image = create_test_image()
    print(f"Created test image: {test_image}")
    
    try:
        from main import ImageAuthenticityDetector
        
        detector = ImageAuthenticityDetector({
            'log_level': 'ERROR',
            'agent_settings': {
                'counterfactual': {'enabled': False},
                'retriever': {'enabled': False}
            }
        })
        
        print("Running analysis...")
        results = detector.analyze_image(test_image)
        
        if results.get('error'):
            print(f"Analysis failed: {results['error']}")
            return False
        
        if results.get('analysis_completed'):
            score = results.get('overall_authenticity_score', 0)
            print(f"Analysis completed - Score: {score:.2%}")
            
            summary = detector.get_analysis_summary()
            print(f"Assessment: {summary.get('authenticity_assessment', 'Unknown')}")
            
            if summary.get('report_path') and os.path.exists(summary['report_path']):
                print(f"Report generated: {summary['report_path']}")
            
            return True
        else:
            print("Analysis incomplete")
            return False
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False
        
    finally:
        if os.path.exists(test_image):
            os.unlink(test_image)

def test_components():
    print("\nTesting individual components...")
    
    try:
        from tools.vision_tools import VisionAuthenticityDetector
        print("Vision tools import successful")
    except Exception as e:
        print(f"Vision tools: {e}")
        return False
    
    try:
        from tools.metadata_tools import MetadataForensicsAnalyzer
        print("Metadata tools import successful")
    except Exception as e:
        print(f"Metadata tools: {e}")
        return False
    
    try:
        from tools.search_tools import ImageSearchEngine
        print("Search tools import successful")
    except Exception as e:
        print(f"Search tools: {e}")
        return False
    
    try:
        from tools.report_tools import AuthenticityReportGenerator
        print("Report tools import successful")
    except Exception as e:
        print(f"Report tools: {e}")
        return False
    
    return True

def main():
    print("=" * 50)
    print("IMAGE AUTHENTICITY DETECTOR - FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Component imports", test_components),
        ("Basic functionality", test_basic_functionality),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\nRunning {name}...")
        if test_func():
            passed += 1
        else:
            print(f"{name} failed")
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All tests passed! System is functional.")
        return 0
    else:
        print("Some tests failed. Check configuration.")
        return 1

if __name__ == "__main__":
    exit(main())
