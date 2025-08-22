"""
Classifier Agent - Performs computer vision-based image authenticity classification.
Uses CLIP, ViT, and other vision models to determine if an image is authentic or AI-generated.
"""

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional
import os
import sys

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

from vision_tools import VisionAuthenticityDetector, ImagePreprocessor


class ClassifierAgent:
    """
    Specialized agent for computer vision-based image authenticity classification.
    Uses multiple deep learning models to analyze visual features.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the classifier agent."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=self.api_key
        )
        
        # Initialize vision tools
        try:
            self.vision_detector = VisionAuthenticityDetector()
            self.image_preprocessor = ImagePreprocessor()
        except Exception as e:
            print(f"Warning: Could not initialize vision models: {e}")
            self.vision_detector = None
            self.image_preprocessor = None
        
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the classifier agent with CrewAI."""
        return Agent(
            role="Computer Vision Authenticity Classifier",
            goal="Analyze images using advanced computer vision models to determine authenticity and detect AI generation",
            backstory="""You are an expert computer vision specialist focused on image authenticity detection. 
            You use state-of-the-art deep learning models including CLIP and Vision Transformers to analyze 
            visual features, patterns, and anomalies that indicate whether an image is authentic, AI-generated, 
            or digitally manipulated. Your expertise lies in interpreting model outputs, understanding visual 
            artifacts, and providing confidence-calibrated assessments of image authenticity.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def perform_vision_analysis(self, image_path: str, analysis_config: Dict = None) -> Dict:
        """
        Perform comprehensive computer vision analysis for authenticity detection.
        
        Args:
            image_path: Path to the image file
            analysis_config: Optional configuration for analysis parameters
            
        Returns:
            Dict containing vision analysis results
        """
        try:
            # Check if image exists and is readable
            if not os.path.exists(image_path):
                return {
                    'error': f"Image file not found: {image_path}",
                    'overall_authenticity_score': 0.5,
                    'analysis_completed': False
                }
            
            results = {
                'analysis_type': 'computer_vision_classification',
                'image_path': image_path,
                'analysis_completed': False,
                'overall_authenticity_score': 0.5,
                'model_results': {},
                'visual_features': {},
                'anomaly_detection': {},
                'confidence_assessment': {},
                'recommendations': []
            }
            
            # Perform vision model analysis
            if self.vision_detector:
                vision_results = self.vision_detector.analyze_image_authenticity(image_path)
                results.update(vision_results)
                results['analysis_completed'] = True
            else:
                results['error'] = "Vision models not available"
                return results
            
            # Extract basic image features
            if self.image_preprocessor:
                features = self.image_preprocessor.extract_image_features(image_path)
                results['visual_features'] = features
            
            # Use AI agent for interpretation and reasoning
            interpretation_results = self._interpret_vision_results(results, analysis_config)
            results.update(interpretation_results)
            
            return results
            
        except Exception as e:
            return {
                'error': f"Vision analysis failed: {str(e)}",
                'overall_authenticity_score': 0.5,
                'analysis_completed': False,
                'image_path': image_path
            }
    
    def _interpret_vision_results(self, vision_results: Dict, config: Dict = None) -> Dict:
        """Use AI agent to interpret vision model results and provide reasoning."""
        try:
            # Create interpretation task
            interpretation_task = Task(
                description=f"""
                Analyze and interpret the following computer vision results for image authenticity detection:
                
                Vision Analysis Results:
                {self._format_results_for_prompt(vision_results)}
                
                Analysis Configuration: {config or 'Standard analysis'}
                
                Please provide:
                1. Interpretation of model confidence scores and predictions
                2. Analysis of visual anomalies and their significance
                3. Assessment of consistency between different models
                4. Identification of key authenticity indicators or red flags
                5. Overall confidence level in the authenticity assessment
                6. Specific recommendations for further analysis if needed
                7. Explanation of reasoning chain for the conclusions
                
                Consider:
                - Model reliability and known limitations
                - Significance of detected anomalies
                - Consistency across multiple vision models
                - Visual artifacts that indicate AI generation or manipulation
                """,
                agent=self.agent,
                expected_output="Comprehensive interpretation with reasoning, confidence assessment, and recommendations"
            )
            
            # Execute interpretation
            crew = Crew(
                agents=[self.agent],
                tasks=[interpretation_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse interpretation result
            interpretation = self._parse_interpretation_result(str(result))
            
            return {
                'ai_interpretation': interpretation,
                'reasoning_chain': interpretation.get('reasoning', []),
                'confidence_level': interpretation.get('confidence_level', 'medium'),
                'key_findings': interpretation.get('key_findings', []),
                'recommendations': interpretation.get('recommendations', [])
            }
            
        except Exception as e:
            return {
                'interpretation_error': f"AI interpretation failed: {str(e)}",
                'reasoning_chain': ["Error in AI interpretation"],
                'confidence_level': 'low'
            }
    
    def _format_results_for_prompt(self, results: Dict) -> str:
        """Format vision results for AI prompt."""
        formatted = []
        
        # Overall score
        if 'overall_authenticity_score' in results:
            formatted.append(f"Overall Authenticity Score: {results['overall_authenticity_score']:.3f}")
        
        # Model scores
        if 'authenticity_scores' in results:
            formatted.append("\nModel Scores:")
            for model, score in results['authenticity_scores'].items():
                formatted.append(f"  {model.upper()}: {score:.3f}")
        
        # Model predictions
        if 'model_predictions' in results:
            formatted.append("\nModel Predictions:")
            for model, prediction in results['model_predictions'].items():
                formatted.append(f"  {model.upper()}: {prediction}")
        
        # Visual anomalies
        if 'visual_anomalies' in results:
            formatted.append("\nVisual Anomalies Detected:")
            for anomaly_type, data in results['visual_anomalies'].items():
                if isinstance(data, dict):
                    score = data.get('score', 0)
                    suspicious = data.get('suspicious', False)
                    status = "SUSPICIOUS" if suspicious else "NORMAL"
                    formatted.append(f"  {anomaly_type}: Score={score:.3f}, Status={status}")
        
        # Confidence scores
        if 'confidence_scores' in results:
            formatted.append("\nModel Confidence Scores:")
            for model, confidence in results['confidence_scores'].items():
                formatted.append(f"  {model.upper()}: {confidence:.3f}")
        
        return "\n".join(formatted)
    
    def _parse_interpretation_result(self, result_text: str) -> Dict:
        """Parse AI interpretation result into structured format."""
        # In production, this would use more sophisticated parsing
        # For now, we'll create a structured response based on the text
        
        interpretation = {
            'summary': result_text[:500] + "..." if len(result_text) > 500 else result_text,
            'reasoning': [],
            'confidence_level': 'medium',
            'key_findings': [],
            'recommendations': []
        }
        
        # Simple parsing logic (in production, use more advanced NLP)
        text_lower = result_text.lower()
        
        # Determine confidence level
        if 'high confidence' in text_lower or 'very confident' in text_lower:
            interpretation['confidence_level'] = 'high'
        elif 'low confidence' in text_lower or 'uncertain' in text_lower:
            interpretation['confidence_level'] = 'low'
        
        # Extract key findings (simplified)
        if 'ai-generated' in text_lower or 'artificial' in text_lower:
            interpretation['key_findings'].append("Indicators of AI generation detected")
        if 'authentic' in text_lower and 'likely' in text_lower:
            interpretation['key_findings'].append("Strong indicators of authenticity")
        if 'anomaly' in text_lower or 'artifact' in text_lower:
            interpretation['key_findings'].append("Visual anomalies detected")
        
        # Extract recommendations (simplified)
        if 'further analysis' in text_lower:
            interpretation['recommendations'].append("Conduct additional analysis")
        if 'metadata' in text_lower:
            interpretation['recommendations'].append("Examine metadata for additional clues")
        
        # Create reasoning chain (simplified)
        interpretation['reasoning'] = [
            "Computer vision models analyzed image features",
            "Multiple models provided authenticity scores",
            "Visual anomalies were detected and evaluated",
            "Results were synthesized for final assessment"
        ]
        
        return interpretation
    
    def analyze_specific_regions(self, image_path: str, regions: List[Dict]) -> Dict:
        """
        Analyze specific regions of the image for detailed authenticity assessment.
        
        Args:
            image_path: Path to the image file
            regions: List of region specifications with coordinates
            
        Returns:
            Dict containing region-specific analysis results
        """
        try:
            # This is a placeholder for advanced regional analysis
            # In production, this would crop regions and analyze them separately
            
            analysis_task = Task(
                description=f"""
                Perform regional authenticity analysis for specific areas of the image at {image_path}.
                
                Target regions: {regions}
                
                For each region, assess:
                1. Local authenticity indicators
                2. Consistency with surrounding areas  
                3. Artifacts specific to that region
                4. Likelihood of manipulation or generation
                
                Provide detailed per-region assessment with confidence scores.
                """,
                agent=self.agent,
                expected_output="Regional analysis with per-region authenticity scores and findings"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[analysis_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'regional_analysis': str(result),
                'analyzed_regions': regions,
                'analysis_type': 'regional_authenticity'
            }
            
        except Exception as e:
            return {
                'error': f"Regional analysis failed: {str(e)}",
                'analyzed_regions': regions
            }
    
    def compare_with_reference_images(self, image_path: str, reference_paths: List[str]) -> Dict:
        """
        Compare the target image with reference images for consistency analysis.
        
        Args:
            image_path: Path to the target image
            reference_paths: List of paths to reference images
            
        Returns:
            Dict containing comparison analysis results
        """
        try:
            comparison_task = Task(
                description=f"""
                Compare the target image at {image_path} with reference images: {reference_paths}
                
                Analyze:
                1. Visual similarity and consistency
                2. Style and technique differences
                3. Metadata consistency
                4. Authenticity indicators comparison
                
                Determine if the target image is consistent with the reference set or shows
                signs of being from a different source (potentially AI-generated).
                """,
                agent=self.agent,
                expected_output="Comparison analysis with similarity scores and authenticity assessment"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[analysis_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'comparison_analysis': str(result),
                'target_image': image_path,
                'reference_images': reference_paths,
                'analysis_type': 'reference_comparison'
            }
            
        except Exception as e:
            return {
                'error': f"Comparison analysis failed: {str(e)}",
                'target_image': image_path,
                'reference_images': reference_paths
            }
    
    def get_authenticity_confidence(self, analysis_results: Dict) -> Dict:
        """
        Calculate and explain confidence levels for authenticity assessment.
        
        Args:
            analysis_results: Results from vision analysis
            
        Returns:
            Dict containing confidence assessment
        """
        try:
            overall_score = analysis_results.get('overall_authenticity_score', 0.5)
            model_scores = analysis_results.get('authenticity_scores', {})
            confidence_scores = analysis_results.get('confidence_scores', {})
            
            # Calculate confidence factors
            confidence_factors = {
                'model_agreement': self._calculate_model_agreement(model_scores),
                'individual_confidence': self._calculate_individual_confidence(confidence_scores),
                'score_distribution': self._analyze_score_distribution(model_scores),
                'anomaly_consistency': self._analyze_anomaly_consistency(analysis_results)
            }
            
            # Overall confidence calculation
            overall_confidence = sum(confidence_factors.values()) / len(confidence_factors)
            
            confidence_level = 'high' if overall_confidence > 0.7 else 'medium' if overall_confidence > 0.4 else 'low'
            
            return {
                'overall_confidence': overall_confidence,
                'confidence_level': confidence_level,
                'confidence_factors': confidence_factors,
                'reliability_assessment': {
                    'model_reliability': 'high' if len(model_scores) > 1 else 'medium',
                    'data_quality': 'high' if not analysis_results.get('error') else 'low',
                    'analysis_completeness': 'high' if analysis_results.get('analysis_completed') else 'low'
                }
            }
            
        except Exception as e:
            return {
                'error': f"Confidence calculation failed: {str(e)}",
                'overall_confidence': 0.5,
                'confidence_level': 'low'
            }
    
    def _calculate_model_agreement(self, model_scores: Dict) -> float:
        """Calculate agreement between different models."""
        if len(model_scores) < 2:
            return 0.5
        
        scores = list(model_scores.values())
        variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
        
        # High agreement = low variance
        agreement = max(0, 1 - variance * 4)  # Scale variance to 0-1
        return agreement
    
    def _calculate_individual_confidence(self, confidence_scores: Dict) -> float:
        """Calculate average individual model confidence."""
        if not confidence_scores:
            return 0.5
        
        return sum(confidence_scores.values()) / len(confidence_scores)
    
    def _analyze_score_distribution(self, model_scores: Dict) -> float:
        """Analyze the distribution of authenticity scores."""
        if not model_scores:
            return 0.5
        
        scores = list(model_scores.values())
        mean_score = sum(scores) / len(scores)
        
        # Scores closer to extremes (0 or 1) indicate higher confidence
        extremity = max(mean_score, 1 - mean_score)
        return extremity
    
    def _analyze_anomaly_consistency(self, analysis_results: Dict) -> float:
        """Analyze consistency of anomaly detection results."""
        anomalies = analysis_results.get('visual_anomalies', {})
        if not anomalies:
            return 0.5
        
        suspicious_count = sum(1 for data in anomalies.values() 
                             if isinstance(data, dict) and data.get('suspicious', False))
        
        total_anomalies = len(anomalies)
        
        # Consistency in anomaly detection increases confidence
        if suspicious_count == 0 or suspicious_count == total_anomalies:
            return 0.8  # All consistent
        else:
            return 0.4  # Mixed results
