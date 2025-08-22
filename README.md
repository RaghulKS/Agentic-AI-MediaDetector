Image Authenticity Detector

Multi-agent system for detecting AI-generated and manipulated images using computer vision, metadata analysis, and contextual verification.

Features

- Computer vision models (CLIP, ViT) for visual authenticity assessment
- EXIF metadata extraction and forensics analysis  
- Reverse image search and source verification
- Consistency testing through alternative scenario analysis
- Detailed PDF reports with confidence scores and visualizations
- Modular agent-based architecture for extensibility

Quick Start

Requirements
- Python 3.8+
- OpenAI API key
- 4GB+ RAM
- GPU optional

Installation

1. Clone repository:
```bash
git clone <repository-url>
cd image-authenticity-detector
```

2. Set up environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure API key:
```bash
cp env_template.txt .env
# Edit .env file and add: OPENAI_API_KEY=your_key_here
```

Usage

Basic analysis:
```bash
python main.py image.jpg
```

With configuration:
```bash
python main.py image.jpg --config config.json --log-level INFO
```

How It Works

The system uses multiple specialized agents that work together:

1. **Planner** - Creates analysis workflow based on image characteristics
2. **Classifier** - Runs computer vision models to detect AI-generated content  
3. **Forensics** - Analyzes EXIF metadata and file properties
4. **Retriever** - Performs reverse image search and source verification
5. **Counterfactual** - Tests consistency through alternative explanations
6. **Explainer** - Generates human-readable analysis results
7. **Reporter** - Creates detailed PDF reports with visualizations

Output

The system provides:

- **Authenticity Score**: 0-100% confidence that image is authentic
- **Analysis Breakdown**: Scores from each analysis component
- **Key Findings**: Most significant evidence found
- **Risk Assessment**: Usage recommendations based on findings
- **PDF Report**: Professional document with detailed analysis

Example output:
```
Overall Authenticity Score: 23.4%
Assessment: Likely AI-generated or manipulated
Confidence Level: high

Key Findings:
1. Computer vision models strongly suggest AI generation
2. Missing essential camera metadata
3. Visual artifacts consistent with diffusion model output

Report: reports/authenticity_report_20240822_140523.pdf
```

Configuration

Key settings in `config.json`:

```json
{
  "analysis_settings": {
    "confidence_threshold": 0.7,
    "enable_caching": true
  },
  "agent_settings": {
    "classifier": {
      "anomaly_detection_threshold": 0.6
    },
    "retriever": {
      "max_search_results": 10
    }
  }
}
```

Environment variables in `.env`:
```
OPENAI_API_KEY=your_key_here
LOG_LEVEL=INFO
CONFIDENCE_THRESHOLD=0.7
```


Advanced Usage

Programmatic API

```python
from main import ImageAuthenticityDetector

detector = ImageAuthenticityDetector({
    'log_level': 'INFO',
    'confidence_threshold': 0.8
})

results = detector.analyze_image('image.jpg')
print(f"Score: {results['overall_authenticity_score']:.2%}")

summary = detector.get_analysis_summary()
print(f"Assessment: {summary['authenticity_assessment']}")
```

Batch Processing

```python
import os
from main import ImageAuthenticityDetector

detector = ImageAuthenticityDetector()

for filename in os.listdir('images/'):
    if filename.lower().endswith(('.jpg', '.png')):
        results = detector.analyze_image(f'images/{filename}')
        score = results.get('overall_authenticity_score', 0)
        print(f"{filename}: {score:.1%}")
```

Limitations

- AI models may miss new generation techniques
- Metadata can be stripped or falsified
- Requires internet connection for reverse search
- Processing time varies (30 seconds to several minutes)
- Not suitable as sole evidence for legal proceedings

Testing

Run validation tests:
```bash
python validate_structure.py  # Check project structure
python test_functionality.py  # Test with sample image
```

