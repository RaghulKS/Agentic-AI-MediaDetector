"""
Reporter Agent - Generates comprehensive PDF reports with visualizations.
Creates professional trust reports combining all analysis results with visual elements.
"""

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional
import os
import sys
from datetime import datetime

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

from report_tools import AuthenticityReportGenerator, VisualizationGenerator


class ReporterAgent:
    """
    Specialized agent for generating comprehensive authenticity reports.
    Creates professional PDF documents with analysis results, visualizations, and recommendations.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the reporter agent."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,  # Very low temperature for consistent, professional reports
            openai_api_key=self.api_key
        )
        
        # Initialize report generation tools
        self.report_generator = AuthenticityReportGenerator()
        self.visualization_generator = VisualizationGenerator()
        
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the reporter agent with CrewAI."""
        return Agent(
            role="Professional Report Generation Specialist",
            goal="Create comprehensive, professional PDF reports that clearly communicate image authenticity analysis results with supporting visualizations and actionable recommendations",
            backstory="""You are an expert technical writer and report generation specialist with deep expertise 
            in presenting complex analytical results in clear, professional formats. You excel at creating 
            comprehensive reports that serve both technical and business audiences, combining detailed analysis 
            with executive summaries, visual elements, and actionable recommendations. Your reports are known 
            for their clarity, accuracy, and professional presentation, helping organizations make informed 
            decisions about image authenticity with confidence in the underlying analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def generate_report(self, image_path: str, complete_analysis: Dict, report_config: Dict = None) -> Dict:
        """
        Generate comprehensive authenticity report.
        
        Args:
            image_path: Path to the analyzed image
            complete_analysis: Complete results from all analysis agents
            report_config: Optional configuration for report generation
            
        Returns:
            Dict containing report generation results and file paths
        """
        try:
            results = {
                'report_type': 'comprehensive_authenticity_report',
                'generation_completed': False,
                'report_file_path': '',
                'report_metadata': {},
                'visual_elements': {},
                'report_sections': [],
                'generation_errors': []
            }
            
            # Prepare report data
            report_data = self._prepare_report_data(image_path, complete_analysis, report_config)
            
            # Generate report structure and content
            report_structure = self._create_report_structure(report_data)
            results['report_sections'] = report_structure
            
            # Create visualizations
            visualizations = self._create_visualizations(report_data)
            results['visual_elements'] = visualizations
            
            # Generate the actual PDF report
            report_path = self.report_generator.generate_report(
                image_path=image_path,
                analysis_results=complete_analysis,
                output_filename=report_data.get('output_filename')
            )
            
            results['report_file_path'] = report_path
            
            # Create report metadata
            metadata = self._create_report_metadata(report_data, report_path)
            results['report_metadata'] = metadata
            
            results['generation_completed'] = True
            
            return results
            
        except Exception as e:
            return {
                'error': f"Report generation failed: {str(e)}",
                'generation_completed': False,
                'report_file_path': '',
                'image_path': image_path
            }
    
    def _prepare_report_data(self, image_path: str, analysis_results: Dict, config: Dict = None) -> Dict:
        """Prepare and structure data for report generation."""
        try:
            # Create report preparation task
            preparation_task = Task(
                description=f"""
                Prepare and structure the analysis data for professional report generation:
                
                IMAGE PATH: {image_path}
                
                COMPLETE ANALYSIS RESULTS:
                Overall Score: {analysis_results.get('overall_authenticity_score', 0.5):.2%}
                Vision Analysis: {analysis_results.get('vision_analysis', {})}
                Metadata Analysis: {analysis_results.get('metadata_analysis', {})}
                Search Analysis: {analysis_results.get('search_analysis', {})}
                Counterfactual Analysis: {analysis_results.get('counterfactual_analysis', {})}
                Explanation: {analysis_results.get('explanation', {})}
                
                Report Configuration: {config or 'Standard comprehensive report'}
                
                Organize the data into report-ready sections:
                1. Executive summary content
                2. Technical findings summary
                3. Evidence presentation structure
                4. Risk assessment and recommendations
                5. Supporting visualizations needed
                6. Appendix technical details
                
                Ensure all content is:
                - Professional and clear
                - Technically accurate
                - Accessible to both technical and business audiences
                - Well-organized with logical flow
                - Supported by evidence and reasoning
                """,
                agent=self.agent,
                expected_output="Structured report data with organized sections and clear content hierarchy"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[preparation_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Structure the prepared data
            report_data = {
                'image_path': image_path,
                'analysis_results': analysis_results,
                'report_config': config or {},
                'preparation_notes': str(result),
                'output_filename': self._generate_report_filename(image_path),
                'generation_timestamp': datetime.now().isoformat()
            }
            
            return report_data
            
        except Exception as e:
            return {
                'error': f"Report data preparation failed: {str(e)}",
                'image_path': image_path,
                'analysis_results': analysis_results,
                'output_filename': self._generate_report_filename(image_path)
            }
    
    def _create_report_structure(self, report_data: Dict) -> List[Dict]:
        """Create the structure and outline for the report."""
        try:
            structure_task = Task(
                description=f"""
                Create a comprehensive report structure for the image authenticity analysis report:
                
                REPORT DATA:
                {self._format_report_data_for_prompt(report_data)}
                
                Create a detailed report structure with:
                
                1. TITLE PAGE
                   - Report title
                   - Overall authenticity assessment
                   - Executive summary score visualization
                   - Image preview
                   - Analysis date and metadata
                
                2. EXECUTIVE SUMMARY (1-2 pages)
                   - Primary conclusion
                   - Key supporting evidence
                   - Risk assessment
                   - Recommendations summary
                
                3. METHODOLOGY OVERVIEW (1 page)
                   - Analysis approach explanation
                   - Tools and models used
                   - Process workflow
                
                4. DETAILED FINDINGS (3-4 pages)
                   - Computer vision analysis results
                   - Metadata and forensics findings
                   - Source verification results
                   - Consistency validation
                
                5. EVIDENCE ANALYSIS (2-3 pages)
                   - Supporting evidence summary
                   - Contradictory evidence discussion
                   - Evidence reliability assessment
                   - Uncertainty analysis
                
                6. RISK ASSESSMENT AND RECOMMENDATIONS (1-2 pages)
                   - Usage recommendations
                   - Risk mitigation strategies
                   - Additional verification suggestions
                
                7. TECHNICAL APPENDIX (2-3 pages)
                   - Detailed technical results
                   - Model outputs and scores
                   - Metadata tables
                   - References and methodology details
                
                For each section, specify:
                - Content outline
                - Key points to cover
                - Supporting visuals needed
                - Estimated length
                """,
                agent=self.agent,
                expected_output="Detailed report structure with section outlines and content specifications"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[structure_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse structure from result
            structure = self._parse_report_structure(str(result))
            return structure
            
        except Exception as e:
            # Return default structure on error
            return [
                {'section': 'Title Page', 'content': 'Report title and overview'},
                {'section': 'Executive Summary', 'content': 'Key findings and recommendations'},
                {'section': 'Detailed Analysis', 'content': 'Technical analysis results'},
                {'section': 'Recommendations', 'content': 'Usage guidance and next steps'},
                {'section': 'Technical Appendix', 'content': 'Detailed technical information'}
            ]
    
    def _create_visualizations(self, report_data: Dict) -> Dict:
        """Create visualizations for the report."""
        visualizations = {
            'created': [],
            'planned': [],
            'errors': []
        }
        
        try:
            analysis_results = report_data.get('analysis_results', {})
            
            # Create authenticity scores chart
            scores = {
                'Vision': analysis_results.get('vision_analysis', {}).get('overall_authenticity_score', 0.5),
                'Metadata': analysis_results.get('metadata_analysis', {}).get('authenticity_score', 0.5),
                'Search': analysis_results.get('search_analysis', {}).get('authenticity_indicators', {}).get('source_authenticity_score', 0.5),
                'Overall': analysis_results.get('overall_authenticity_score', 0.5)
            }
            
            try:
                chart_path = os.path.join(self.report_generator.output_dir, 'authenticity_scores.png')
                self.visualization_generator.create_score_chart(scores, chart_path)
                visualizations['created'].append({
                    'type': 'score_chart',
                    'path': chart_path,
                    'description': 'Authenticity scores by analysis component'
                })
            except Exception as e:
                visualizations['errors'].append(f"Score chart creation failed: {str(e)}")
            
            # Plan additional visualizations
            visualizations['planned'].extend([
                {'type': 'confidence_assessment', 'description': 'Confidence level visualization'},
                {'type': 'evidence_summary', 'description': 'Evidence strength breakdown'},
                {'type': 'risk_matrix', 'description': 'Risk assessment matrix'}
            ])
            
            return visualizations
            
        except Exception as e:
            visualizations['errors'].append(f"Visualization creation failed: {str(e)}")
            return visualizations
    
    def _create_report_metadata(self, report_data: Dict, report_path: str) -> Dict:
        """Create metadata for the generated report."""
        metadata = {
            'report_id': self._generate_report_id(),
            'creation_timestamp': datetime.now().isoformat(),
            'report_file_path': report_path,
            'image_analyzed': report_data.get('image_path'),
            'analysis_components': [],
            'overall_authenticity_score': 0.5,
            'confidence_level': 'medium',
            'report_version': '1.0',
            'generator_info': {
                'system': 'Agentic AI Image Authenticity Detector',
                'version': '1.0.0',
                'models_used': ['GPT-4o', 'CLIP', 'ViT']
            }
        }
        
        try:
            analysis_results = report_data.get('analysis_results', {})
            
            # Extract analysis components used
            if analysis_results.get('vision_analysis', {}).get('analysis_completed'):
                metadata['analysis_components'].append('Computer Vision Analysis')
            
            if analysis_results.get('metadata_analysis', {}).get('analysis_completed'):
                metadata['analysis_components'].append('Metadata Forensics')
            
            if analysis_results.get('search_analysis', {}).get('search_completed'):
                metadata['analysis_components'].append('Reverse Search Analysis')
            
            if analysis_results.get('counterfactual_analysis', {}).get('analysis_completed'):
                metadata['analysis_components'].append('Counterfactual Analysis')
            
            # Extract overall score and confidence
            metadata['overall_authenticity_score'] = analysis_results.get('overall_authenticity_score', 0.5)
            
            explanation = analysis_results.get('explanation', {})
            if explanation:
                confidence_assessment = explanation.get('confidence_assessment', {})
                metadata['confidence_level'] = confidence_assessment.get('confidence_level', 'medium')
            
            return metadata
            
        except Exception as e:
            metadata['metadata_error'] = str(e)
            return metadata
    
    def _generate_report_filename(self, image_path: str) -> str:
        """Generate filename for the report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        return f"authenticity_report_{image_name}_{timestamp}.pdf"
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"AUTH_REPORT_{timestamp}"
    
    def _format_report_data_for_prompt(self, report_data: Dict) -> str:
        """Format report data for AI prompt."""
        formatted = []
        
        analysis_results = report_data.get('analysis_results', {})
        formatted.append(f"Image Path: {report_data.get('image_path')}")
        formatted.append(f"Overall Score: {analysis_results.get('overall_authenticity_score', 0.5):.2%}")
        
        # Add component summaries
        components = ['vision_analysis', 'metadata_analysis', 'search_analysis', 'counterfactual_analysis']
        for component in components:
            component_data = analysis_results.get(component, {})
            if component_data:
                formatted.append(f"\n{component.replace('_', ' ').title()}:")
                if component_data.get('error'):
                    formatted.append(f"  Status: Error - {component_data['error']}")
                elif component_data.get('analysis_completed', False):
                    formatted.append(f"  Status: Completed successfully")
                else:
                    formatted.append(f"  Status: Incomplete or unavailable")
        
        # Add explanation summary if available
        explanation = analysis_results.get('explanation', {})
        if explanation:
            formatted.append(f"\nExplanation Available: Yes")
            formatted.append(f"Key Findings: {len(explanation.get('key_findings', []))}")
            formatted.append(f"Recommendations: {len(explanation.get('recommendations', []))}")
        
        return "\n".join(formatted)
    
    def _parse_report_structure(self, structure_text: str) -> List[Dict]:
        """Parse report structure from AI-generated text."""
        sections = []
        
        # Look for section headers
        lines = structure_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header (contains section number or key words)
            if any(keyword in line.lower() for keyword in ['title page', 'executive summary', 'methodology', 'findings', 'recommendations', 'appendix']):
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'section': line,
                    'content': '',
                    'estimated_pages': 1
                }
            elif current_section and line:
                current_section['content'] += line + ' '
        
        if current_section:
            sections.append(current_section)
        
        # Ensure we have basic sections even if parsing fails
        if not sections:
            sections = [
                {'section': 'Title Page', 'content': 'Report overview and summary'},
                {'section': 'Executive Summary', 'content': 'Key findings and recommendations'},
                {'section': 'Technical Analysis', 'content': 'Detailed analysis results'},
                {'section': 'Recommendations', 'content': 'Usage guidance and next steps'},
                {'section': 'Technical Appendix', 'content': 'Supporting technical details'}
            ]
        
        return sections
    
    def create_summary_report(self, image_path: str, analysis_results: Dict) -> Dict:
        """Create a condensed summary report for quick review."""
        try:
            summary_task = Task(
                description=f"""
                Create a condensed summary report (2-3 pages) for quick decision-making:
                
                IMAGE: {image_path}
                ANALYSIS RESULTS: {analysis_results}
                
                Include only:
                1. Executive summary with clear conclusion
                2. Key evidence summary (top 3-5 points)
                3. Risk assessment and primary recommendation
                4. Confidence level and any critical caveats
                
                Format for busy executives who need quick, actionable insights.
                """,
                agent=self.agent,
                expected_output="Condensed summary report content with key decision points"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[summary_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'summary_report_content': str(result),
                'report_type': 'executive_summary',
                'length': 'condensed'
            }
            
        except Exception as e:
            return {
                'error': f"Summary report creation failed: {str(e)}",
                'report_type': 'executive_summary'
            }
    
    def create_technical_report(self, image_path: str, analysis_results: Dict) -> Dict:
        """Create a detailed technical report for expert review."""
        try:
            technical_task = Task(
                description=f"""
                Create a comprehensive technical report for expert analysis:
                
                IMAGE: {image_path}
                DETAILED RESULTS: {analysis_results}
                
                Include:
                1. Detailed methodology and model specifications
                2. Complete technical findings with confidence intervals
                3. Statistical analysis and uncertainty quantification
                4. Comparative analysis between different methods
                5. Limitations and potential sources of error
                6. Raw data tables and technical appendices
                7. Recommendations for further technical validation
                
                Format for technical experts and researchers who need complete details.
                """,
                agent=self.agent,
                expected_output="Comprehensive technical report with full analytical details"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[technical_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'technical_report_content': str(result),
                'report_type': 'technical_detailed',
                'audience': 'technical_experts'
            }
            
        except Exception as e:
            return {
                'error': f"Technical report creation failed: {str(e)}",
                'report_type': 'technical_detailed'
            }
    
    def validate_report_quality(self, report_path: str, analysis_results: Dict) -> Dict:
        """Validate the quality and completeness of the generated report."""
        validation = {
            'report_exists': False,
            'file_size_mb': 0,
            'completeness_score': 0.0,
            'quality_issues': [],
            'recommendations': []
        }
        
        try:
            # Check if file exists
            if os.path.exists(report_path):
                validation['report_exists'] = True
                
                # Check file size
                file_size = os.path.getsize(report_path)
                validation['file_size_mb'] = round(file_size / (1024 * 1024), 2)
                
                # Validate file size is reasonable
                if file_size < 100000:  # Less than 100KB
                    validation['quality_issues'].append("Report file unusually small")
                elif file_size > 50 * 1024 * 1024:  # More than 50MB
                    validation['quality_issues'].append("Report file unusually large")
            else:
                validation['quality_issues'].append("Report file not found")
                return validation
            
            # Validate analysis completeness
            completeness_factors = []
            
            if analysis_results.get('overall_authenticity_score') is not None:
                completeness_factors.append(1.0)
            else:
                validation['quality_issues'].append("Missing overall authenticity score")
                completeness_factors.append(0.0)
            
            # Check for key analysis components
            required_components = ['vision_analysis', 'metadata_analysis', 'explanation']
            for component in required_components:
                if component in analysis_results:
                    completeness_factors.append(1.0)
                else:
                    validation['quality_issues'].append(f"Missing {component}")
                    completeness_factors.append(0.0)
            
            validation['completeness_score'] = sum(completeness_factors) / len(completeness_factors)
            
            # Generate recommendations
            if validation['completeness_score'] < 0.8:
                validation['recommendations'].append("Consider regenerating report with complete analysis")
            
            if validation['file_size_mb'] > 0 and validation['completeness_score'] > 0.8:
                validation['recommendations'].append("Report appears complete and ready for use")
            
            return validation
            
        except Exception as e:
            validation['validation_error'] = str(e)
            return validation
