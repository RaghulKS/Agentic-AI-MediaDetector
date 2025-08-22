"""
Forensics Agent - Performs metadata extraction and digital forensics analysis.
Analyzes EXIF data, file properties, and digital artifacts to detect signs of manipulation or AI generation.
"""

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional
import os
import sys
from datetime import datetime

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

from metadata_tools import MetadataForensicsAnalyzer, MetadataExtractor


class ForensicsAgent:
    """
    Specialized agent for digital forensics and metadata analysis of images.
    Focuses on technical evidence that can indicate authenticity or manipulation.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the forensics agent."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=self.api_key
        )
        
        # Initialize forensics tools
        self.metadata_analyzer = MetadataForensicsAnalyzer()
        self.metadata_extractor = MetadataExtractor()
        
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the forensics agent with CrewAI."""
        return Agent(
            role="Digital Forensics and Metadata Specialist",
            goal="Perform comprehensive digital forensics analysis to detect signs of image manipulation, AI generation, or authenticity",
            backstory="""You are an expert digital forensics analyst specializing in image authenticity detection. 
            Your expertise includes EXIF metadata analysis, file structure examination, timestamp forensics, 
            and detection of digital manipulation artifacts. You understand camera technologies, image formats, 
            compression algorithms, and the telltale signs that distinguish authentic photographs from AI-generated 
            or manipulated images. Your analysis provides crucial technical evidence for authenticity assessment.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def perform_forensics_analysis(self, image_path: str, analysis_config: Dict = None) -> Dict:
        """
        Perform comprehensive digital forensics analysis.
        
        Args:
            image_path: Path to the image file
            analysis_config: Optional configuration for analysis parameters
            
        Returns:
            Dict containing forensics analysis results
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                return {
                    'error': f"Image file not found: {image_path}",
                    'authenticity_score': 0.5,
                    'analysis_completed': False
                }
            
            results = {
                'analysis_type': 'digital_forensics',
                'image_path': image_path,
                'analysis_completed': False,
                'authenticity_score': 0.5,
                'metadata_analysis': {},
                'file_analysis': {},
                'technical_indicators': {},
                'forensic_findings': {},
                'evidence_summary': []
            }
            
            # Perform metadata analysis
            metadata_results = self.metadata_analyzer.analyze_metadata(image_path)
            results['metadata_analysis'] = metadata_results
            
            # Extract specific metadata components
            camera_info = self.metadata_extractor.get_camera_info(image_path)
            creation_time = self.metadata_extractor.get_creation_time(image_path)
            gps_coords = self.metadata_extractor.get_gps_coordinates(image_path)
            
            results['camera_info'] = camera_info
            results['creation_time'] = str(creation_time) if creation_time else None
            results['gps_coordinates'] = gps_coords
            
            # Perform file-level analysis
            file_analysis = self._analyze_file_properties(image_path)
            results['file_analysis'] = file_analysis
            
            # Use AI agent for forensic interpretation
            forensic_interpretation = self._interpret_forensic_evidence(results, analysis_config)
            results.update(forensic_interpretation)
            
            # Calculate overall authenticity score from metadata
            if 'authenticity_score' in metadata_results:
                results['authenticity_score'] = metadata_results['authenticity_score']
            
            results['analysis_completed'] = True
            
            return results
            
        except Exception as e:
            return {
                'error': f"Forensics analysis failed: {str(e)}",
                'authenticity_score': 0.5,
                'analysis_completed': False,
                'image_path': image_path
            }
    
    def _analyze_file_properties(self, image_path: str) -> Dict:
        """Analyze file-level properties for forensic indicators."""
        try:
            file_stats = os.stat(image_path)
            
            analysis = {
                'file_size': file_stats.st_size,
                'file_size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                'created_time': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'accessed_time': datetime.fromtimestamp(file_stats.st_atime).isoformat()
            }
            
            # Analyze file extension and format consistency
            file_extension = os.path.splitext(image_path)[1].lower()
            analysis['file_extension'] = file_extension
            
            # Check for suspicious file properties
            suspicious_indicators = []
            
            # Very small or very large file sizes can be suspicious
            if analysis['file_size'] < 10000:  # Less than 10KB
                suspicious_indicators.append("Unusually small file size")
            elif analysis['file_size'] > 50 * 1024 * 1024:  # More than 50MB
                suspicious_indicators.append("Unusually large file size")
            
            # Check timestamp consistency
            created = file_stats.st_ctime
            modified = file_stats.st_mtime
            
            if abs(created - modified) > 86400:  # More than 1 day difference
                suspicious_indicators.append("Large gap between creation and modification times")
            
            analysis['suspicious_indicators'] = suspicious_indicators
            
            return analysis
            
        except Exception as e:
            return {'file_analysis_error': str(e)}
    
    def _interpret_forensic_evidence(self, forensic_data: Dict, config: Dict = None) -> Dict:
        """Use AI agent to interpret forensic evidence and provide expert analysis."""
        try:
            # Create forensic interpretation task
            interpretation_task = Task(
                description=f"""
                As a digital forensics expert, analyze the following technical evidence to assess image authenticity:
                
                METADATA ANALYSIS:
                {self._format_metadata_for_prompt(forensic_data.get('metadata_analysis', {}))}
                
                FILE PROPERTIES:
                {self._format_file_analysis_for_prompt(forensic_data.get('file_analysis', {}))}
                
                CAMERA INFORMATION:
                {forensic_data.get('camera_info', 'No camera info available')}
                
                CREATION TIME: {forensic_data.get('creation_time', 'Not available')}
                GPS COORDINATES: {forensic_data.get('gps_coordinates', 'Not available')}
                
                Analysis Configuration: {config or 'Standard forensic analysis'}
                
                Provide expert forensic analysis including:
                1. Assessment of metadata completeness and authenticity
                2. Evaluation of suspicious patterns or anomalies
                3. Technical indicators that suggest authenticity or manipulation
                4. Camera and device consistency analysis
                5. Timestamp and file property assessment
                6. Overall forensic authenticity score (0.0-1.0)
                7. Confidence level in the forensic findings
                8. Key evidence points supporting the conclusion
                9. Recommendations for additional forensic investigation
                
                Consider:
                - Known patterns of AI-generated images
                - Common manipulation techniques and their artifacts
                - Camera technology and metadata standards
                - File format specifications and normal behavior
                """,
                agent=self.agent,
                expected_output="Comprehensive forensic analysis with technical assessment, authenticity score, and evidence summary"
            )
            
            # Execute forensic interpretation
            crew = Crew(
                agents=[self.agent],
                tasks=[interpretation_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse forensic interpretation result
            interpretation = self._parse_forensic_interpretation(str(result))
            
            return {
                'forensic_interpretation': interpretation,
                'expert_assessment': interpretation.get('assessment', 'Analysis completed'),
                'forensic_score': interpretation.get('forensic_score', 0.5),
                'confidence_level': interpretation.get('confidence_level', 'medium'),
                'key_evidence': interpretation.get('key_evidence', []),
                'red_flags': interpretation.get('red_flags', []),
                'authenticity_indicators': interpretation.get('authenticity_indicators', []),
                'recommendations': interpretation.get('recommendations', [])
            }
            
        except Exception as e:
            return {
                'interpretation_error': f"Forensic interpretation failed: {str(e)}",
                'forensic_score': 0.5,
                'confidence_level': 'low'
            }
    
    def _format_metadata_for_prompt(self, metadata: Dict) -> str:
        """Format metadata analysis results for AI prompt."""
        if not metadata:
            return "No metadata analysis available"
        
        formatted = []
        
        # Authenticity score
        if 'authenticity_score' in metadata:
            formatted.append(f"Metadata Authenticity Score: {metadata['authenticity_score']:.3f}")
        
        # Authenticity indicators
        if 'authenticity_indicators' in metadata:
            indicators = metadata['authenticity_indicators']
            formatted.append("\nAuthenticity Indicators:")
            formatted.append(f"  Has Camera Info: {indicators.get('has_camera_info', False)}")
            formatted.append(f"  Has Timestamp: {indicators.get('has_timestamp', False)}")
            formatted.append(f"  Has GPS Data: {indicators.get('has_gps_data', False)}")
            formatted.append(f"  Camera Make: {indicators.get('camera_make', 'Not available')}")
            formatted.append(f"  Camera Model: {indicators.get('camera_model', 'Not available')}")
        
        # Suspicious patterns
        if 'suspicious_patterns' in metadata and metadata['suspicious_patterns']:
            formatted.append("\nSuspicious Patterns Detected:")
            for pattern in metadata['suspicious_patterns']:
                if isinstance(pattern, dict):
                    pattern_type = pattern.get('type', 'Unknown')
                    severity = pattern.get('severity', 'unknown')
                    formatted.append(f"  - {pattern_type} (Severity: {severity})")
        
        # Metadata completeness
        if 'metadata_completeness' in metadata:
            completeness = metadata['metadata_completeness']
            if 'overall' in completeness:
                overall = completeness['overall']
                percentage = overall.get('percentage', 0)
                formatted.append(f"\nMetadata Completeness: {percentage:.1f}%")
        
        return "\n".join(formatted) if formatted else "No detailed metadata information"
    
    def _format_file_analysis_for_prompt(self, file_analysis: Dict) -> str:
        """Format file analysis results for AI prompt."""
        if not file_analysis:
            return "No file analysis available"
        
        formatted = []
        
        if 'file_size_mb' in file_analysis:
            formatted.append(f"File Size: {file_analysis['file_size_mb']} MB")
        
        if 'file_extension' in file_analysis:
            formatted.append(f"File Extension: {file_analysis['file_extension']}")
        
        if 'created_time' in file_analysis:
            formatted.append(f"Created: {file_analysis['created_time']}")
        
        if 'modified_time' in file_analysis:
            formatted.append(f"Modified: {file_analysis['modified_time']}")
        
        if 'suspicious_indicators' in file_analysis and file_analysis['suspicious_indicators']:
            formatted.append("\nFile-level Suspicious Indicators:")
            for indicator in file_analysis['suspicious_indicators']:
                formatted.append(f"  - {indicator}")
        
        return "\n".join(formatted) if formatted else "Basic file information analyzed"
    
    def _parse_forensic_interpretation(self, result_text: str) -> Dict:
        """Parse forensic interpretation result into structured format."""
        interpretation = {
            'assessment': result_text[:300] + "..." if len(result_text) > 300 else result_text,
            'forensic_score': 0.5,
            'confidence_level': 'medium',
            'key_evidence': [],
            'red_flags': [],
            'authenticity_indicators': [],
            'recommendations': []
        }
        
        text_lower = result_text.lower()
        
        # Extract forensic score (simplified parsing)
        score_patterns = ['authenticity score:', 'forensic score:', 'score:']
        for pattern in score_patterns:
            if pattern in text_lower:
                try:
                    # Look for number after the pattern
                    start_idx = text_lower.find(pattern) + len(pattern)
                    score_section = text_lower[start_idx:start_idx + 20]
                    # Extract first number found
                    import re
                    numbers = re.findall(r'0\.\d+|\d\.\d+', score_section)
                    if numbers:
                        interpretation['forensic_score'] = float(numbers[0])
                        break
                except:
                    pass
        
        # Determine confidence level
        if 'high confidence' in text_lower or 'very confident' in text_lower:
            interpretation['confidence_level'] = 'high'
        elif 'low confidence' in text_lower or 'uncertain' in text_lower:
            interpretation['confidence_level'] = 'low'
        
        # Extract key findings
        if 'authentic' in text_lower and ('camera' in text_lower or 'metadata' in text_lower):
            interpretation['authenticity_indicators'].append("Camera metadata supports authenticity")
        
        if 'missing' in text_lower and 'metadata' in text_lower:
            interpretation['red_flags'].append("Critical metadata missing")
        
        if 'suspicious' in text_lower and 'timestamp' in text_lower:
            interpretation['red_flags'].append("Suspicious timestamp patterns")
        
        if 'ai' in text_lower and 'generated' in text_lower:
            interpretation['red_flags'].append("Indicators of AI generation")
        
        # Extract recommendations
        if 'additional' in text_lower and 'analysis' in text_lower:
            interpretation['recommendations'].append("Conduct additional forensic analysis")
        
        if 'verify' in text_lower:
            interpretation['recommendations'].append("Verify findings with additional methods")
        
        return interpretation
    
    def analyze_compression_artifacts(self, image_path: str) -> Dict:
        """
        Analyze compression artifacts that might indicate manipulation or generation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing compression analysis results
        """
        try:
            compression_task = Task(
                description=f"""
                Analyze compression artifacts in the image at {image_path} to detect signs of manipulation or AI generation.
                
                Focus on:
                1. JPEG compression patterns and quality assessment
                2. Unusual compression artifacts that don't match natural photo compression
                3. Multiple compression signatures indicating editing
                4. Compression consistency across the image
                5. Quality factors that might indicate synthetic generation
                
                Provide detailed analysis of compression-related authenticity indicators.
                """,
                agent=self.agent,
                expected_output="Compression analysis with authenticity assessment and technical details"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[compression_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'compression_analysis': str(result),
                'analysis_type': 'compression_artifacts'
            }
            
        except Exception as e:
            return {
                'error': f"Compression analysis failed: {str(e)}"
            }
    
    def detect_editing_software_traces(self, metadata: Dict) -> Dict:
        """
        Detect traces of editing software in metadata.
        
        Args:
            metadata: Extracted metadata dictionary
            
        Returns:
            Dict containing software trace analysis
        """
        try:
            software_traces = []
            confidence_indicators = []
            
            # Common editing software signatures
            editing_software = {
                'photoshop': ['adobe photoshop', 'photoshop'],
                'gimp': ['gimp', 'gnu image manipulation'],
                'lightroom': ['lightroom', 'adobe lightroom'],
                'canva': ['canva'],
                'ai_tools': ['midjourney', 'dall-e', 'stable diffusion', 'ai', 'generated']
            }
            
            # Check software field in metadata
            software_field = str(metadata.get('software', '')).lower()
            
            for category, patterns in editing_software.items():
                for pattern in patterns:
                    if pattern in software_field:
                        software_traces.append({
                            'category': category,
                            'pattern': pattern,
                            'field': 'software',
                            'severity': 'high' if category == 'ai_tools' else 'medium'
                        })
            
            # Analyze other metadata fields for software traces
            for field, value in metadata.items():
                if isinstance(value, str):
                    value_lower = value.lower()
                    for category, patterns in editing_software.items():
                        for pattern in patterns:
                            if pattern in value_lower and field != 'software':
                                software_traces.append({
                                    'category': category,
                                    'pattern': pattern,
                                    'field': field,
                                    'severity': 'medium'
                                })
            
            # Calculate confidence based on findings
            if software_traces:
                ai_traces = [t for t in software_traces if t['category'] == 'ai_tools']
                if ai_traces:
                    confidence_indicators.append("AI generation software detected")
                
                editing_traces = [t for t in software_traces if t['category'] != 'ai_tools']
                if editing_traces:
                    confidence_indicators.append("Image editing software detected")
            
            return {
                'software_traces': software_traces,
                'confidence_indicators': confidence_indicators,
                'total_traces_found': len(software_traces),
                'ai_generation_indicators': len([t for t in software_traces if t['category'] == 'ai_tools'])
            }
            
        except Exception as e:
            return {
                'error': f"Software trace detection failed: {str(e)}",
                'software_traces': []
            }
    
    def validate_camera_consistency(self, metadata: Dict) -> Dict:
        """
        Validate consistency of camera-related metadata.
        
        Args:
            metadata: Extracted metadata dictionary
            
        Returns:
            Dict containing camera consistency validation results
        """
        try:
            validation_results = {
                'is_consistent': True,
                'inconsistencies': [],
                'authenticity_score': 0.5,
                'validation_details': {}
            }
            
            # Extract camera-related fields
            camera_make = metadata.get('camera_make', metadata.get('Make', ''))
            camera_model = metadata.get('camera_model', metadata.get('Model', ''))
            lens_make = metadata.get('LensMake', '')
            lens_model = metadata.get('LensModel', '')
            
            # Check for camera make/model consistency
            if camera_make and camera_model:
                # Simple consistency checks (in production, use camera database)
                make_lower = camera_make.lower()
                model_lower = camera_model.lower()
                
                # Check if model contains make
                if make_lower not in model_lower:
                    # Some cameras don't include make in model, so this is just a flag
                    validation_results['validation_details']['make_model_relationship'] = 'different_naming'
                else:
                    validation_results['validation_details']['make_model_relationship'] = 'consistent'
                
                # Check for known camera brands
                known_brands = ['canon', 'nikon', 'sony', 'fuji', 'olympus', 'panasonic', 'leica']
                if any(brand in make_lower for brand in known_brands):
                    validation_results['authenticity_score'] += 0.2
                    validation_results['validation_details']['known_brand'] = True
                else:
                    validation_results['validation_details']['known_brand'] = False
            else:
                validation_results['inconsistencies'].append("Missing camera make or model")
                validation_results['is_consistent'] = False
            
            # Check lens consistency
            if lens_make and camera_make:
                if lens_make.lower() != camera_make.lower():
                    # Third-party lens is normal, not an inconsistency
                    validation_results['validation_details']['third_party_lens'] = True
                else:
                    validation_results['validation_details']['native_lens'] = True
            
            # Final consistency score
            if validation_results['is_consistent'] and not validation_results['inconsistencies']:
                validation_results['authenticity_score'] = min(validation_results['authenticity_score'] + 0.3, 1.0)
            
            return validation_results
            
        except Exception as e:
            return {
                'error': f"Camera validation failed: {str(e)}",
                'is_consistent': False,
                'authenticity_score': 0.0
            }
