"""
Explainer Agent - Synthesizes findings into human-readable explanations.
Creates comprehensive reasoning chains and explanations for authenticity assessments.
"""

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional
import os
from datetime import datetime


class ExplainerAgent:
    """
    Specialized agent for synthesizing analysis results into clear, human-readable explanations.
    Creates reasoning chains, confidence assessments, and actionable insights.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the explainer agent."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,  # Low temperature for consistent, logical explanations
            openai_api_key=self.api_key
        )
        
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the explainer agent with CrewAI."""
        return Agent(
            role="Image Authenticity Analysis Interpreter and Explainer",
            goal="Synthesize complex technical analysis results into clear, actionable explanations for image authenticity assessment",
            backstory="""You are an expert communication specialist with deep technical knowledge in image forensics, 
            computer vision, and digital authenticity detection. Your role is to take complex multi-modal analysis 
            results and transform them into clear, logical, and actionable explanations that both technical and 
            non-technical audiences can understand. You excel at creating coherent reasoning chains, explaining 
            confidence levels, identifying key evidence, and providing practical recommendations. Your explanations 
            help users make informed decisions about image authenticity while understanding the limitations and 
            uncertainties in the analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def generate_explanation(self, analysis_results: Dict, explanation_config: Dict = None) -> Dict:
        """
        Generate comprehensive explanation of image authenticity analysis.
        
        Args:
            analysis_results: Complete results from all analysis agents
            explanation_config: Optional configuration for explanation style and depth
            
        Returns:
            Dict containing structured explanation and reasoning
        """
        try:
            results = {
                'explanation_type': 'comprehensive_authenticity_assessment',
                'analysis_completed': False,
                'executive_summary': '',
                'detailed_explanation': '',
                'reasoning_chain': [],
                'evidence_summary': {},
                'confidence_assessment': {},
                'key_findings': [],
                'limitations_and_caveats': [],
                'recommendations': [],
                'technical_details': {}
            }
            
            # Extract and structure key information from analysis results
            structured_analysis = self._structure_analysis_results(analysis_results)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(structured_analysis, explanation_config)
            results['executive_summary'] = executive_summary
            
            # Generate detailed explanation
            detailed_explanation = self._generate_detailed_explanation(structured_analysis, explanation_config)
            results['detailed_explanation'] = detailed_explanation
            
            # Create reasoning chain
            reasoning_chain = self._create_reasoning_chain(structured_analysis)
            results['reasoning_chain'] = reasoning_chain
            
            # Summarize evidence
            evidence_summary = self._summarize_evidence(structured_analysis)
            results['evidence_summary'] = evidence_summary
            
            # Assess confidence
            confidence_assessment = self._assess_explanation_confidence(structured_analysis)
            results['confidence_assessment'] = confidence_assessment
            
            # Extract key findings
            key_findings = self._extract_key_findings(structured_analysis)
            results['key_findings'] = key_findings
            
            # Identify limitations
            limitations = self._identify_limitations(structured_analysis)
            results['limitations_and_caveats'] = limitations
            
            # Generate recommendations
            recommendations = self._generate_recommendations(structured_analysis)
            results['recommendations'] = recommendations
            
            # Compile technical details
            technical_details = self._compile_technical_details(structured_analysis)
            results['technical_details'] = technical_details
            
            results['analysis_completed'] = True
            
            return results
            
        except Exception as e:
            return {
                'error': f"Explanation generation failed: {str(e)}",
                'analysis_completed': False,
                'executive_summary': 'Analysis completed with errors',
                'detailed_explanation': f'Error in explanation generation: {str(e)}'
            }
    
    def _structure_analysis_results(self, analysis_results: Dict) -> Dict:
        """Structure and organize analysis results for explanation generation."""
        structured = {
            'overall_score': analysis_results.get('overall_authenticity_score', 0.5),
            'vision_analysis': analysis_results.get('vision_analysis', {}),
            'metadata_analysis': analysis_results.get('metadata_analysis', {}),
            'search_analysis': analysis_results.get('search_analysis', {}),
            'counterfactual_analysis': analysis_results.get('counterfactual_analysis', {}),
            'image_path': analysis_results.get('image_path', 'Unknown'),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Extract key metrics from each analysis component
        structured['component_scores'] = {
            'vision_score': structured['vision_analysis'].get('overall_authenticity_score', 0.5),
            'metadata_score': structured['metadata_analysis'].get('authenticity_score', 0.5),
            'search_score': structured['search_analysis'].get('authenticity_indicators', {}).get('source_authenticity_score', 0.5),
            'consistency_score': structured['counterfactual_analysis'].get('consistency_score', 0.5)
        }
        
        return structured
    
    def _generate_executive_summary(self, structured_analysis: Dict, config: Dict = None) -> str:
        """Generate executive summary of the analysis."""
        try:
            summary_task = Task(
                description=f"""
                Create a clear, concise executive summary of the image authenticity analysis:
                
                ANALYSIS RESULTS:
                Overall Authenticity Score: {structured_analysis['overall_score']:.2%}
                Vision Analysis Score: {structured_analysis['component_scores']['vision_score']:.2%}
                Metadata Analysis Score: {structured_analysis['component_scores']['metadata_score']:.2%}
                Search Analysis Score: {structured_analysis['component_scores']['search_score']:.2%}
                Consistency Score: {structured_analysis['component_scores']['consistency_score']:.2%}
                
                DETAILED COMPONENTS:
                {self._format_components_for_prompt(structured_analysis)}
                
                Configuration: {config or 'Standard executive summary'}
                
                Create a 3-4 paragraph executive summary that:
                1. States the primary conclusion about image authenticity
                2. Highlights the most compelling evidence supporting this conclusion
                3. Acknowledges any significant uncertainties or conflicting evidence
                4. Provides a clear recommendation for how to treat this image
                
                Write in clear, professional language suitable for decision-makers.
                Avoid technical jargon while maintaining accuracy.
                """,
                agent=self.agent,
                expected_output="Clear, professional executive summary with primary conclusion and key supporting evidence"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[summary_task],
                verbose=True
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            return f"Executive summary generation failed: {str(e)}"
    
    def _generate_detailed_explanation(self, structured_analysis: Dict, config: Dict = None) -> str:
        """Generate detailed explanation of the analysis process and results."""
        try:
            explanation_task = Task(
                description=f"""
                Create a comprehensive, detailed explanation of the image authenticity analysis:
                
                COMPLETE ANALYSIS DATA:
                {self._format_complete_analysis_for_prompt(structured_analysis)}
                
                Configuration: {config or 'Standard detailed explanation'}
                
                Create a detailed explanation that covers:
                
                1. METHODOLOGY OVERVIEW
                   - Brief description of the multi-agent analysis approach
                   - Types of analysis performed (computer vision, metadata, search, consistency)
                   
                2. COMPUTER VISION ANALYSIS
                   - How AI models (CLIP, ViT) assessed the image
                   - Visual anomalies or patterns detected
                   - Model confidence and agreement levels
                   
                3. METADATA AND FORENSICS ANALYSIS
                   - EXIF data findings and their significance
                   - Camera information and timestamp analysis
                   - Detection of suspicious patterns or missing data
                   
                4. SOURCE AND CONTEXT VERIFICATION
                   - Reverse search results and source credibility
                   - Contextual information gathered
                   - Cross-referencing with known sources
                   
                5. CONSISTENCY VALIDATION
                   - How different analysis methods agreed or disagreed
                   - Alternative explanations considered
                   - Robustness of conclusions
                   
                6. SYNTHESIS AND CONCLUSION
                   - How evidence was weighted and combined
                   - Final authenticity assessment reasoning
                   - Confidence level justification
                
                Use clear explanations that help readers understand both the process and the reasoning.
                Address any conflicting evidence or uncertainties transparently.
                """,
                agent=self.agent,
                expected_output="Comprehensive detailed explanation covering all analysis components with clear reasoning"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[explanation_task],
                verbose=True
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            return f"Detailed explanation generation failed: {str(e)}"
    
    def _create_reasoning_chain(self, structured_analysis: Dict) -> List[str]:
        """Create step-by-step reasoning chain for the authenticity assessment."""
        try:
            reasoning_task = Task(
                description=f"""
                Create a clear, logical reasoning chain that shows how the authenticity conclusion was reached:
                
                ANALYSIS COMPONENTS:
                {self._format_components_for_prompt(structured_analysis)}
                
                Create a step-by-step reasoning chain (8-12 steps) that shows:
                1. Initial hypothesis or question
                2. Evidence gathering process
                3. Analysis of each type of evidence
                4. Evaluation of evidence quality and reliability
                5. Consideration of alternative explanations
                6. Integration of multiple evidence sources
                7. Resolution of any conflicts or uncertainties
                8. Final conclusion and confidence assessment
                
                Each step should be clear, logical, and directly connected to the next.
                Show how evidence accumulates to support the final conclusion.
                Be transparent about assumptions and limitations.
                
                Format as a numbered list of reasoning steps.
                """,
                agent=self.agent,
                expected_output="Numbered list of clear reasoning steps showing logical flow to conclusion"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[reasoning_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse reasoning steps from result
            reasoning_steps = self._parse_reasoning_steps(str(result))
            return reasoning_steps
            
        except Exception as e:
            return [f"Reasoning chain generation failed: {str(e)}"]
    
    def _summarize_evidence(self, structured_analysis: Dict) -> Dict:
        """Summarize key evidence supporting the authenticity assessment."""
        evidence_summary = {
            'supporting_authenticity': [],
            'indicating_manipulation': [],
            'neutral_or_inconclusive': [],
            'evidence_strength': {},
            'evidence_reliability': {}
        }
        
        try:
            overall_score = structured_analysis['overall_score']
            
            # Vision analysis evidence
            vision_score = structured_analysis['component_scores']['vision_score']
            if vision_score > 0.6:
                evidence_summary['supporting_authenticity'].append("Computer vision models indicate likely authenticity")
                evidence_summary['evidence_strength']['vision_analysis'] = 'strong' if vision_score > 0.8 else 'moderate'
            elif vision_score < 0.4:
                evidence_summary['indicating_manipulation'].append("Computer vision models suggest AI generation or manipulation")
                evidence_summary['evidence_strength']['vision_analysis'] = 'strong' if vision_score < 0.2 else 'moderate'
            else:
                evidence_summary['neutral_or_inconclusive'].append("Computer vision analysis inconclusive")
                evidence_summary['evidence_strength']['vision_analysis'] = 'weak'
            
            # Metadata analysis evidence
            metadata_score = structured_analysis['component_scores']['metadata_score']
            metadata_analysis = structured_analysis.get('metadata_analysis', {})
            
            if metadata_analysis.get('authenticity_indicators', {}).get('has_camera_info'):
                evidence_summary['supporting_authenticity'].append("Complete camera metadata present")
            
            suspicious_patterns = len(metadata_analysis.get('suspicious_patterns', []))
            if suspicious_patterns > 0:
                evidence_summary['indicating_manipulation'].append(f"{suspicious_patterns} suspicious metadata patterns detected")
            
            # Search analysis evidence
            search_analysis = structured_analysis.get('search_analysis', {})
            credible_sources = len(search_analysis.get('positive_indicators', []))
            red_flags = len(search_analysis.get('red_flags', []))
            
            if credible_sources > 0:
                evidence_summary['supporting_authenticity'].append(f"Found on {credible_sources} credible sources")
            
            if red_flags > 0:
                evidence_summary['indicating_manipulation'].append(f"{red_flags} red flags in source analysis")
            
            # Assign evidence reliability based on analysis quality
            for component in ['vision_analysis', 'metadata_analysis', 'search_analysis']:
                component_data = structured_analysis.get(component, {})
                if component_data.get('error'):
                    evidence_summary['evidence_reliability'][component] = 'low'
                elif component_data.get('analysis_completed', False):
                    evidence_summary['evidence_reliability'][component] = 'high'
                else:
                    evidence_summary['evidence_reliability'][component] = 'medium'
            
            return evidence_summary
            
        except Exception as e:
            evidence_summary['error'] = f"Evidence summarization failed: {str(e)}"
            return evidence_summary
    
    def _assess_explanation_confidence(self, structured_analysis: Dict) -> Dict:
        """Assess confidence in the explanation and conclusions."""
        confidence_assessment = {
            'overall_confidence': 'medium',
            'confidence_score': 0.5,
            'confidence_factors': {},
            'uncertainty_sources': [],
            'confidence_justification': ''
        }
        
        try:
            # Calculate confidence based on multiple factors
            confidence_factors = []
            
            # Data completeness
            components_with_data = sum(1 for component in ['vision_analysis', 'metadata_analysis', 'search_analysis'] 
                                     if structured_analysis.get(component, {}).get('analysis_completed', False))
            
            data_completeness = components_with_data / 3.0
            confidence_factors.append(data_completeness)
            
            # Score consistency between components
            scores = list(structured_analysis['component_scores'].values())
            score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            consistency_factor = max(0, 1 - score_variance * 4)  # Lower variance = higher confidence
            confidence_factors.append(consistency_factor)
            
            # Counterfactual analysis consistency
            consistency_score = structured_analysis.get('counterfactual_analysis', {}).get('consistency_score', 0.5)
            confidence_factors.append(consistency_score)
            
            # Overall confidence calculation
            overall_confidence_score = sum(confidence_factors) / len(confidence_factors)
            confidence_assessment['confidence_score'] = overall_confidence_score
            
            if overall_confidence_score > 0.7:
                confidence_assessment['overall_confidence'] = 'high'
            elif overall_confidence_score > 0.4:
                confidence_assessment['overall_confidence'] = 'medium'
            else:
                confidence_assessment['overall_confidence'] = 'low'
            
            # Identify uncertainty sources
            if data_completeness < 0.8:
                confidence_assessment['uncertainty_sources'].append("Incomplete analysis data")
            
            if score_variance > 0.1:
                confidence_assessment['uncertainty_sources'].append("Inconsistent results between analysis methods")
            
            if consistency_score < 0.5:
                confidence_assessment['uncertainty_sources'].append("Low consistency in counterfactual analysis")
            
            # Store individual factor assessments
            confidence_assessment['confidence_factors'] = {
                'data_completeness': data_completeness,
                'score_consistency': consistency_factor,
                'counterfactual_consistency': consistency_score
            }
            
            return confidence_assessment
            
        except Exception as e:
            confidence_assessment['error'] = f"Confidence assessment failed: {str(e)}"
            return confidence_assessment
    
    def _extract_key_findings(self, structured_analysis: Dict) -> List[str]:
        """Extract key findings from the analysis."""
        key_findings = []
        
        try:
            overall_score = structured_analysis['overall_score']
            
            # Overall assessment finding
            if overall_score > 0.7:
                key_findings.append("Image shows strong indicators of authenticity")
            elif overall_score < 0.3:
                key_findings.append("Image shows strong indicators of being AI-generated or manipulated")
            else:
                key_findings.append("Image authenticity is uncertain and requires additional investigation")
            
            # Component-specific findings
            component_scores = structured_analysis['component_scores']
            
            # Vision analysis findings
            vision_score = component_scores['vision_score']
            if vision_score < 0.3:
                key_findings.append("Computer vision models strongly suggest AI generation")
            elif vision_score > 0.7:
                key_findings.append("Computer vision models support image authenticity")
            
            # Metadata findings
            metadata_analysis = structured_analysis.get('metadata_analysis', {})
            if metadata_analysis.get('suspicious_patterns'):
                key_findings.append("Suspicious patterns detected in image metadata")
            
            auth_indicators = metadata_analysis.get('authenticity_indicators', {})
            if auth_indicators.get('has_camera_info') and auth_indicators.get('has_timestamp'):
                key_findings.append("Complete camera metadata supports authenticity")
            
            # Search findings
            search_analysis = structured_analysis.get('search_analysis', {})
            if search_analysis.get('red_flags'):
                key_findings.append("Source analysis reveals credibility concerns")
            elif search_analysis.get('positive_indicators'):
                key_findings.append("Found on reputable sources supporting authenticity")
            
            # Consistency findings
            consistency_score = component_scores.get('consistency_score', 0.5)
            if consistency_score < 0.4:
                key_findings.append("Analysis methods show significant inconsistencies")
            elif consistency_score > 0.7:
                key_findings.append("High consistency across different analysis methods")
            
            return key_findings
            
        except Exception as e:
            return [f"Key findings extraction failed: {str(e)}"]
    
    def _identify_limitations(self, structured_analysis: Dict) -> List[str]:
        """Identify limitations and caveats in the analysis."""
        limitations = []
        
        try:
            # Check for missing analysis components
            if not structured_analysis.get('vision_analysis', {}).get('analysis_completed'):
                limitations.append("Computer vision analysis was incomplete or failed")
            
            if not structured_analysis.get('metadata_analysis', {}).get('analysis_completed'):
                limitations.append("Metadata analysis was incomplete or unavailable")
            
            if not structured_analysis.get('search_analysis', {}).get('search_completed'):
                limitations.append("Reverse search analysis was incomplete")
            
            # General limitations
            limitations.extend([
                "AI models may have biases or blind spots in detecting certain types of generated content",
                "Metadata can be stripped, modified, or spoofed in sophisticated manipulations",
                "Reverse search results depend on the image's previous online presence",
                "Analysis accuracy may vary for different image types, styles, or generation methods"
            ])
            
            # Specific limitations based on findings
            overall_score = structured_analysis['overall_score']
            if 0.4 <= overall_score <= 0.6:
                limitations.append("Inconclusive results require human expert judgment for final determination")
            
            # Model-specific limitations
            vision_analysis = structured_analysis.get('vision_analysis', {})
            if 'error' in vision_analysis:
                limitations.append("Computer vision model errors may affect accuracy")
            
            return limitations
            
        except Exception as e:
            return [f"Limitations identification failed: {str(e)}"]
    
    def _generate_recommendations(self, structured_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        try:
            overall_score = structured_analysis['overall_score']
            confidence = self._assess_explanation_confidence(structured_analysis)
            
            # Primary recommendations based on score and confidence
            if overall_score > 0.7 and confidence['overall_confidence'] == 'high':
                recommendations.append("Image can be treated as likely authentic for most purposes")
                recommendations.append("Consider the source context when making final decisions")
            elif overall_score < 0.3 and confidence['overall_confidence'] == 'high':
                recommendations.append("Exercise extreme caution - image likely AI-generated or manipulated")
                recommendations.append("Do not use for applications requiring authentic imagery")
                recommendations.append("Consider this image potentially deceptive")
            else:
                recommendations.append("Seek additional verification before making critical decisions")
                recommendations.append("Consider the intended use case and required certainty level")
                recommendations.append("Consult with domain experts if authenticity is crucial")
            
            # Specific recommendations based on findings
            metadata_analysis = structured_analysis.get('metadata_analysis', {})
            if metadata_analysis.get('suspicious_patterns'):
                recommendations.append("Investigate metadata anomalies with specialized forensic tools")
            
            search_analysis = structured_analysis.get('search_analysis', {})
            if len(search_analysis.get('red_flags', [])) > 0:
                recommendations.append("Cross-verify with additional source verification methods")
            
            # Confidence-based recommendations
            if confidence['overall_confidence'] == 'low':
                recommendations.append("Repeat analysis with updated models or additional methods")
                recommendations.append("Seek second opinion from other authenticity detection services")
            
            # Always include general best practices
            recommendations.extend([
                "Document the analysis results and decision rationale",
                "Stay informed about evolving AI generation capabilities",
                "Consider implementing multiple verification methods for critical applications"
            ])
            
            return recommendations
            
        except Exception as e:
            return [f"Recommendations generation failed: {str(e)}"]
    
    def _compile_technical_details(self, structured_analysis: Dict) -> Dict:
        """Compile technical details for transparency and reproducibility."""
        technical_details = {
            'analysis_timestamp': structured_analysis.get('analysis_timestamp'),
            'image_path': structured_analysis.get('image_path'),
            'analysis_methods': [],
            'model_versions': {},
            'component_scores': structured_analysis.get('component_scores', {}),
            'processing_notes': []
        }
        
        try:
            # Document analysis methods used
            if structured_analysis.get('vision_analysis', {}).get('analysis_completed'):
                technical_details['analysis_methods'].append('Computer Vision (CLIP, ViT)')
            
            if structured_analysis.get('metadata_analysis', {}).get('analysis_completed'):
                technical_details['analysis_methods'].append('Metadata Forensics (EXIF Analysis)')
            
            if structured_analysis.get('search_analysis', {}).get('search_completed'):
                technical_details['analysis_methods'].append('Reverse Image Search')
            
            if structured_analysis.get('counterfactual_analysis', {}).get('analysis_completed'):
                technical_details['analysis_methods'].append('Counterfactual Consistency Testing')
            
            # Note any processing issues
            for component_name, component_data in structured_analysis.items():
                if isinstance(component_data, dict) and 'error' in component_data:
                    technical_details['processing_notes'].append(f"{component_name}: {component_data['error']}")
            
            return technical_details
            
        except Exception as e:
            technical_details['compilation_error'] = str(e)
            return technical_details
    
    def _format_components_for_prompt(self, structured_analysis: Dict) -> str:
        """Format analysis components for AI prompt."""
        formatted = []
        
        for component in ['vision_analysis', 'metadata_analysis', 'search_analysis', 'counterfactual_analysis']:
            component_data = structured_analysis.get(component, {})
            if component_data:
                formatted.append(f"\n{component.replace('_', ' ').title()}:")
                
                # Add key information from each component
                if 'error' in component_data:
                    formatted.append(f"  Error: {component_data['error']}")
                elif component_data.get('analysis_completed', False):
                    formatted.append(f"  Status: Completed successfully")
                    
                    # Add component-specific details
                    if component == 'vision_analysis':
                        score = component_data.get('overall_authenticity_score', 0.5)
                        formatted.append(f"  Authenticity Score: {score:.3f}")
                        
                    elif component == 'metadata_analysis':
                        score = component_data.get('authenticity_score', 0.5)
                        formatted.append(f"  Metadata Score: {score:.3f}")
                        suspicious = len(component_data.get('suspicious_patterns', []))
                        formatted.append(f"  Suspicious Patterns: {suspicious}")
                        
                    elif component == 'search_analysis':
                        auth_indicators = component_data.get('authenticity_indicators', {})
                        score = auth_indicators.get('source_authenticity_score', 0.5)
                        formatted.append(f"  Source Score: {score:.3f}")
                        
                    elif component == 'counterfactual_analysis':
                        score = component_data.get('consistency_score', 0.5)
                        formatted.append(f"  Consistency Score: {score:.3f}")
                else:
                    formatted.append(f"  Status: Incomplete or unavailable")
        
        return "\n".join(formatted)
    
    def _format_complete_analysis_for_prompt(self, structured_analysis: Dict) -> str:
        """Format complete analysis data for detailed explanation prompt."""
        formatted = [f"Overall Authenticity Score: {structured_analysis['overall_score']:.3f}"]
        formatted.append(f"Analysis Timestamp: {structured_analysis.get('analysis_timestamp')}")
        formatted.append(f"Image Path: {structured_analysis.get('image_path')}")
        
        # Add component scores
        formatted.append("\nComponent Scores:")
        for component, score in structured_analysis.get('component_scores', {}).items():
            formatted.append(f"  {component.replace('_', ' ').title()}: {score:.3f}")
        
        # Add detailed component information
        formatted.append(self._format_components_for_prompt(structured_analysis))
        
        return "\n".join(formatted)
    
    def _parse_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """Parse reasoning steps from AI-generated text."""
        steps = []
        
        # Look for numbered items or bullet points
        lines = reasoning_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with number, bullet, or dash
            if any(line.startswith(marker) for marker in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '-', '•', '*']):
                # Clean up the step text
                clean_step = line
                for marker in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '-', '•', '*']:
                    if clean_step.startswith(marker):
                        clean_step = clean_step[len(marker):].strip()
                        break
                
                steps.append(clean_step)
        
        # If no numbered steps found, create basic reasoning steps
        if not steps:
            steps = [
                "Initial analysis identified key evidence from multiple sources",
                "Computer vision models assessed visual authenticity indicators", 
                "Metadata analysis examined technical evidence",
                "Source verification checked credibility and context",
                "Evidence was weighed based on reliability and consistency",
                "Final authenticity assessment reached through evidence synthesis"
            ]
        
        return steps
