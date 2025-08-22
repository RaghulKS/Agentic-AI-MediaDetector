"""
Counterfactual Agent - Generates counterfactual analysis for consistency testing.
Attempts to generate plausible alternatives or modifications to test image authenticity.
"""

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional
import os
import sys
from PIL import Image
import numpy as np


class CounterfactualAgent:
    """
    Specialized agent for counterfactual analysis and consistency testing.
    Generates hypothetical scenarios and alternatives to validate authenticity assessment.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the counterfactual agent."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,  # Slightly higher temperature for creative analysis
            openai_api_key=self.api_key
        )
        
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the counterfactual agent with CrewAI."""
        return Agent(
            role="Counterfactual Analysis and Consistency Testing Specialist",
            goal="Generate counterfactual scenarios and alternative hypotheses to test and validate image authenticity assessments",
            backstory="""You are an expert in counterfactual reasoning and consistency testing for image authenticity. 
            Your role involves generating plausible alternative explanations, testing hypotheses through thought experiments, 
            and validating authenticity assessments by considering what would be different if the image were authentic 
            versus AI-generated or manipulated. You use logical reasoning, creative analysis, and systematic testing 
            of alternative scenarios to strengthen or challenge initial authenticity conclusions. Your analysis helps 
            identify potential weaknesses in the assessment and provides additional confidence measures.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def perform_counterfactual_analysis(self, image_path: str, initial_analysis: Dict, config: Dict = None) -> Dict:
        """
        Perform comprehensive counterfactual analysis.
        
        Args:
            image_path: Path to the image file
            initial_analysis: Results from previous analysis agents
            config: Optional configuration for analysis parameters
            
        Returns:
            Dict containing counterfactual analysis results
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                return {
                    'error': f"Image file not found: {image_path}",
                    'analysis_completed': False,
                    'consistency_score': 0.5
                }
            
            results = {
                'analysis_type': 'counterfactual_consistency_testing',
                'image_path': image_path,
                'analysis_completed': False,
                'consistency_score': 0.5,
                'counterfactual_scenarios': [],
                'hypothesis_testing': {},
                'alternative_explanations': [],
                'consistency_validation': {},
                'confidence_adjustment': {}
            }
            
            # Extract key findings from initial analysis
            key_findings = self._extract_key_findings(initial_analysis)
            
            # Generate counterfactual scenarios
            scenarios = self._generate_counterfactual_scenarios(key_findings, image_path)
            results['counterfactual_scenarios'] = scenarios
            
            # Perform hypothesis testing
            hypothesis_results = self._test_alternative_hypotheses(key_findings, scenarios)
            results['hypothesis_testing'] = hypothesis_results
            
            # Generate alternative explanations
            alternatives = self._generate_alternative_explanations(key_findings, image_path)
            results['alternative_explanations'] = alternatives
            
            # Validate consistency of initial findings
            consistency_validation = self._validate_finding_consistency(initial_analysis, scenarios)
            results['consistency_validation'] = consistency_validation
            
            # Calculate overall consistency score
            consistency_score = self._calculate_consistency_score(results)
            results['consistency_score'] = consistency_score
            
            # Provide confidence adjustments based on counterfactual analysis
            confidence_adjustment = self._calculate_confidence_adjustment(results, initial_analysis)
            results['confidence_adjustment'] = confidence_adjustment
            
            results['analysis_completed'] = True
            
            return results
            
        except Exception as e:
            return {
                'error': f"Counterfactual analysis failed: {str(e)}",
                'analysis_completed': False,
                'consistency_score': 0.5,
                'image_path': image_path
            }
    
    def _extract_key_findings(self, initial_analysis: Dict) -> Dict:
        """Extract key findings from previous analysis stages."""
        findings = {
            'vision_findings': {},
            'metadata_findings': {},
            'search_findings': {},
            'overall_score': 0.5
        }
        
        try:
            # Extract vision analysis findings
            vision_analysis = initial_analysis.get('vision_analysis', {})
            if vision_analysis:
                findings['vision_findings'] = {
                    'overall_score': vision_analysis.get('overall_authenticity_score', 0.5),
                    'model_predictions': vision_analysis.get('model_predictions', {}),
                    'anomalies_detected': vision_analysis.get('visual_anomalies', {}),
                    'confidence_level': vision_analysis.get('confidence_level', 'medium')
                }
            
            # Extract metadata findings
            metadata_analysis = initial_analysis.get('metadata_analysis', {})
            if metadata_analysis:
                findings['metadata_findings'] = {
                    'authenticity_score': metadata_analysis.get('authenticity_score', 0.5),
                    'has_camera_info': metadata_analysis.get('authenticity_indicators', {}).get('has_camera_info', False),
                    'suspicious_patterns': len(metadata_analysis.get('suspicious_patterns', [])),
                    'metadata_completeness': metadata_analysis.get('metadata_completeness', {})
                }
            
            # Extract search findings
            search_analysis = initial_analysis.get('search_analysis', {})
            if search_analysis:
                auth_indicators = search_analysis.get('authenticity_indicators', {})
                findings['search_findings'] = {
                    'source_score': auth_indicators.get('source_authenticity_score', 0.5),
                    'credible_sources_found': len(search_analysis.get('positive_indicators', [])),
                    'red_flags_found': len(search_analysis.get('red_flags', []))
                }
            
            # Overall score
            findings['overall_score'] = initial_analysis.get('overall_authenticity_score', 0.5)
            
            return findings
            
        except Exception as e:
            return findings  # Return partial findings on error
    
    def _generate_counterfactual_scenarios(self, key_findings: Dict, image_path: str) -> List[Dict]:
        """Generate counterfactual scenarios for testing."""
        try:
            scenario_task = Task(
                description=f"""
                Generate counterfactual scenarios to test the robustness of image authenticity findings:
                
                Current Key Findings:
                {self._format_findings_for_prompt(key_findings)}
                
                Image Path: {image_path}
                
                Generate 5-7 counterfactual scenarios that test:
                1. "What if this image were authentic instead of AI-generated?" scenario
                2. "What if this image were AI-generated instead of authentic?" scenario  
                3. Alternative explanations for detected anomalies
                4. Different interpretations of metadata evidence
                5. Alternative source origin hypotheses
                6. Scenarios that challenge the strongest findings
                7. Edge cases that could explain conflicting evidence
                
                For each scenario, provide:
                - Scenario description
                - Expected evidence if this scenario were true
                - How it would change the interpretation of current findings
                - Likelihood assessment of this alternative
                - Key distinguishing factors to test this scenario
                
                Focus on realistic and testable alternative explanations.
                """,
                agent=self.agent,
                expected_output="List of detailed counterfactual scenarios with testing criteria"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[scenario_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse scenarios from result
            scenarios = self._parse_scenarios_from_result(str(result))
            
            return scenarios
            
        except Exception as e:
            return [{
                'scenario_type': 'error',
                'description': f"Scenario generation failed: {str(e)}",
                'likelihood': 0.0
            }]
    
    def _test_alternative_hypotheses(self, key_findings: Dict, scenarios: List[Dict]) -> Dict:
        """Test alternative hypotheses against current evidence."""
        try:
            hypothesis_task = Task(
                description=f"""
                Test alternative hypotheses against the current evidence using rigorous logical analysis:
                
                Current Evidence:
                {self._format_findings_for_prompt(key_findings)}
                
                Counterfactual Scenarios to Test:
                {scenarios}
                
                For each hypothesis, evaluate:
                1. Consistency with vision model predictions
                2. Alignment with metadata evidence
                3. Compatibility with search results
                4. Logical coherence of the explanation
                5. Parsimony (simpler explanations preferred)
                6. Falsifiability of the hypothesis
                
                Provide:
                - Hypothesis strength assessment (0.0-1.0)
                - Supporting evidence analysis
                - Contradicting evidence analysis
                - Overall plausibility ranking
                - Recommendations for additional testing
                
                Use Bayesian reasoning principles to update probability assessments.
                """,
                agent=self.agent,
                expected_output="Hypothesis testing results with strength assessments and evidence analysis"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[hypothesis_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'hypothesis_test_results': str(result),
                'tested_scenarios': len(scenarios),
                'analysis_method': 'bayesian_reasoning'
            }
            
        except Exception as e:
            return {
                'error': f"Hypothesis testing failed: {str(e)}",
                'tested_scenarios': 0
            }
    
    def _generate_alternative_explanations(self, key_findings: Dict, image_path: str) -> List[Dict]:
        """Generate alternative explanations for the observed evidence."""
        try:
            explanation_task = Task(
                description=f"""
                Generate creative but plausible alternative explanations for the observed evidence:
                
                Observed Evidence:
                {self._format_findings_for_prompt(key_findings)}
                
                Generate alternative explanations considering:
                1. Technical limitations of analysis methods
                2. Edge cases in authentic photography
                3. Unusual but legitimate image processing
                4. Camera or equipment peculiarities
                5. Environmental or situational factors
                6. Human error in analysis interpretation
                7. Rare but possible authentic scenarios
                
                For each explanation:
                - Provide detailed reasoning
                - Estimate likelihood (0.0-1.0)
                - Identify testable predictions
                - Suggest validation methods
                - Assess impact on overall authenticity assessment
                """,
                agent=self.agent,
                expected_output="Alternative explanations with likelihood assessments and validation suggestions"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[explanation_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse alternative explanations
            explanations = self._parse_explanations_from_result(str(result))
            
            return explanations
            
        except Exception as e:
            return [{
                'explanation_type': 'error',
                'description': f"Alternative explanation generation failed: {str(e)}",
                'likelihood': 0.0
            }]
    
    def _validate_finding_consistency(self, initial_analysis: Dict, scenarios: List[Dict]) -> Dict:
        """Validate consistency of findings across different scenarios."""
        try:
            consistency_task = Task(
                description=f"""
                Validate the internal consistency of findings across different analysis components and scenarios:
                
                Initial Analysis Results:
                {self._format_analysis_summary(initial_analysis)}
                
                Counterfactual Scenarios:
                {scenarios}
                
                Check for:
                1. Consistency between vision and metadata findings
                2. Alignment between search results and technical analysis
                3. Coherence of overall authenticity narrative
                4. Resolution of conflicting evidence
                5. Robustness of conclusions under alternative scenarios
                6. Identification of weak points in the analysis chain
                
                Provide:
                - Consistency score (0.0-1.0)
                - List of consistent findings
                - List of inconsistent or conflicting findings
                - Recommendations for resolving inconsistencies
                - Impact assessment on overall confidence
                """,
                agent=self.agent,
                expected_output="Consistency validation with score and resolution recommendations"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[consistency_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse consistency results
            consistency_results = self._parse_consistency_results(str(result))
            
            return consistency_results
            
        except Exception as e:
            return {
                'error': f"Consistency validation failed: {str(e)}",
                'consistency_score': 0.5
            }
    
    def _calculate_consistency_score(self, counterfactual_results: Dict) -> float:
        """Calculate overall consistency score from counterfactual analysis."""
        try:
            scores = []
            
            # Consistency validation score
            consistency_validation = counterfactual_results.get('consistency_validation', {})
            if 'consistency_score' in consistency_validation:
                scores.append(consistency_validation['consistency_score'])
            
            # Scenario strength analysis
            scenarios = counterfactual_results.get('counterfactual_scenarios', [])
            if scenarios:
                # Average likelihood of non-contradictory scenarios
                scenario_scores = []
                for scenario in scenarios:
                    if isinstance(scenario, dict) and 'likelihood' in scenario:
                        likelihood = scenario['likelihood']
                        # High likelihood of alternative explanations reduces consistency
                        consistency_contribution = 1.0 - likelihood
                        scenario_scores.append(consistency_contribution)
                
                if scenario_scores:
                    scores.append(sum(scenario_scores) / len(scenario_scores))
            
            # Alternative explanation impact
            alternatives = counterfactual_results.get('alternative_explanations', [])
            if alternatives:
                alt_scores = []
                for alt in alternatives:
                    if isinstance(alt, dict) and 'likelihood' in alt:
                        # High likelihood of alternatives reduces consistency of original
                        alt_scores.append(1.0 - alt['likelihood'])
                
                if alt_scores:
                    scores.append(sum(alt_scores) / len(alt_scores))
            
            if not scores:
                return 0.5
            
            # Weighted average (consistency validation gets highest weight)
            if len(scores) == 1:
                return scores[0]
            elif len(scores) == 2:
                return 0.6 * scores[0] + 0.4 * scores[1]  # Consistency validation weighted higher
            else:
                return 0.5 * scores[0] + 0.3 * scores[1] + 0.2 * scores[2]
            
        except Exception:
            return 0.5
    
    def _calculate_confidence_adjustment(self, counterfactual_results: Dict, initial_analysis: Dict) -> Dict:
        """Calculate confidence adjustments based on counterfactual analysis."""
        try:
            adjustment = {
                'original_confidence': 0.5,
                'adjusted_confidence': 0.5,
                'adjustment_factor': 0.0,
                'adjustment_reasoning': [],
                'recommendation': 'maintain'
            }
            
            # Get original confidence
            initial_score = initial_analysis.get('overall_authenticity_score', 0.5)
            adjustment['original_confidence'] = initial_score
            
            # Calculate adjustment based on consistency
            consistency_score = counterfactual_results.get('consistency_score', 0.5)
            
            # High consistency increases confidence, low consistency decreases it
            consistency_adjustment = (consistency_score - 0.5) * 0.2  # Max ±0.1 adjustment
            
            # Strong alternative explanations reduce confidence
            alternatives = counterfactual_results.get('alternative_explanations', [])
            alternative_adjustment = 0.0
            
            high_likelihood_alternatives = sum(1 for alt in alternatives 
                                             if isinstance(alt, dict) and alt.get('likelihood', 0) > 0.7)
            
            if high_likelihood_alternatives > 0:
                alternative_adjustment = -0.1 * high_likelihood_alternatives  # Max -0.3
            
            # Scenario robustness
            scenarios = counterfactual_results.get('counterfactual_scenarios', [])
            scenario_adjustment = 0.0
            
            contradictory_scenarios = sum(1 for scenario in scenarios 
                                        if isinstance(scenario, dict) and scenario.get('likelihood', 0) > 0.6)
            
            if contradictory_scenarios > 0:
                scenario_adjustment = -0.05 * contradictory_scenarios  # Max -0.25
            
            # Total adjustment
            total_adjustment = consistency_adjustment + alternative_adjustment + scenario_adjustment
            total_adjustment = max(-0.3, min(0.3, total_adjustment))  # Cap at ±0.3
            
            adjustment['adjustment_factor'] = total_adjustment
            adjustment['adjusted_confidence'] = max(0.0, min(1.0, initial_score + total_adjustment))
            
            # Reasoning
            if consistency_adjustment > 0:
                adjustment['adjustment_reasoning'].append("High consistency between findings increases confidence")
            elif consistency_adjustment < 0:
                adjustment['adjustment_reasoning'].append("Inconsistencies between findings reduce confidence")
            
            if high_likelihood_alternatives > 0:
                adjustment['adjustment_reasoning'].append(f"{high_likelihood_alternatives} plausible alternative explanations reduce confidence")
            
            if contradictory_scenarios > 0:
                adjustment['adjustment_reasoning'].append(f"{contradictory_scenarios} contradictory scenarios weaken conclusions")
            
            # Recommendation
            if abs(total_adjustment) < 0.05:
                adjustment['recommendation'] = 'maintain'
            elif total_adjustment > 0:
                adjustment['recommendation'] = 'increase_confidence'
            else:
                adjustment['recommendation'] = 'decrease_confidence'
            
            return adjustment
            
        except Exception as e:
            return {
                'error': f"Confidence adjustment calculation failed: {str(e)}",
                'adjustment_factor': 0.0,
                'recommendation': 'maintain'
            }
    
    def _format_findings_for_prompt(self, findings: Dict) -> str:
        """Format key findings for AI prompt."""
        formatted = []
        
        formatted.append(f"Overall Authenticity Score: {findings.get('overall_score', 0.5):.3f}")
        
        vision_findings = findings.get('vision_findings', {})
        if vision_findings:
            formatted.append(f"\nVision Analysis:")
            formatted.append(f"  Score: {vision_findings.get('overall_score', 0.5):.3f}")
            formatted.append(f"  Confidence: {vision_findings.get('confidence_level', 'medium')}")
            
            predictions = vision_findings.get('model_predictions', {})
            if predictions:
                formatted.append(f"  Predictions: {predictions}")
        
        metadata_findings = findings.get('metadata_findings', {})
        if metadata_findings:
            formatted.append(f"\nMetadata Analysis:")
            formatted.append(f"  Score: {metadata_findings.get('authenticity_score', 0.5):.3f}")
            formatted.append(f"  Has Camera Info: {metadata_findings.get('has_camera_info', False)}")
            formatted.append(f"  Suspicious Patterns: {metadata_findings.get('suspicious_patterns', 0)}")
        
        search_findings = findings.get('search_findings', {})
        if search_findings:
            formatted.append(f"\nSearch Analysis:")
            formatted.append(f"  Source Score: {search_findings.get('source_score', 0.5):.3f}")
            formatted.append(f"  Credible Sources: {search_findings.get('credible_sources_found', 0)}")
            formatted.append(f"  Red Flags: {search_findings.get('red_flags_found', 0)}")
        
        return "\n".join(formatted)
    
    def _format_analysis_summary(self, analysis: Dict) -> str:
        """Format analysis summary for prompt."""
        summary = []
        
        if 'overall_authenticity_score' in analysis:
            summary.append(f"Overall Score: {analysis['overall_authenticity_score']:.3f}")
        
        for component in ['vision_analysis', 'metadata_analysis', 'search_analysis']:
            if component in analysis:
                component_data = analysis[component]
                summary.append(f"\n{component.replace('_', ' ').title()}:")
                
                # Add key metrics from each component
                if isinstance(component_data, dict):
                    for key, value in component_data.items():
                        if isinstance(value, (int, float)) and 'score' in key.lower():
                            summary.append(f"  {key}: {value:.3f}")
        
        return "\n".join(summary)
    
    def _parse_scenarios_from_result(self, result_text: str) -> List[Dict]:
        """Parse counterfactual scenarios from AI result."""
        # Simplified parsing - in production, use more sophisticated NLP
        scenarios = []
        
        # Look for numbered scenarios or bullet points
        lines = result_text.split('\n')
        current_scenario = None
        
        for line in lines:
            line = line.strip()
            
            # Detect scenario start
            if any(marker in line.lower() for marker in ['scenario', 'alternative', 'hypothesis']):
                if current_scenario:
                    scenarios.append(current_scenario)
                
                current_scenario = {
                    'scenario_type': 'alternative_hypothesis',
                    'description': line,
                    'likelihood': 0.5,
                    'testing_criteria': []
                }
            elif current_scenario and line:
                # Add details to current scenario
                if 'likelihood' in line.lower():
                    try:
                        import re
                        numbers = re.findall(r'0\.\d+|\d\.\d+', line)
                        if numbers:
                            current_scenario['likelihood'] = float(numbers[0])
                    except:
                        pass
                
                current_scenario['description'] += " " + line
        
        if current_scenario:
            scenarios.append(current_scenario)
        
        # Ensure we have at least some default scenarios
        if not scenarios:
            scenarios = [
                {
                    'scenario_type': 'authentic_alternative',
                    'description': 'Image is authentic with unusual characteristics',
                    'likelihood': 0.3
                },
                {
                    'scenario_type': 'ai_generated_alternative', 
                    'description': 'Image is AI-generated with sophisticated metadata',
                    'likelihood': 0.4
                }
            ]
        
        return scenarios
    
    def _parse_explanations_from_result(self, result_text: str) -> List[Dict]:
        """Parse alternative explanations from AI result."""
        explanations = []
        
        # Simplified parsing
        if 'alternative' in result_text.lower() or 'explanation' in result_text.lower():
            explanations.append({
                'explanation_type': 'technical_limitation',
                'description': result_text[:200] + "..." if len(result_text) > 200 else result_text,
                'likelihood': 0.3
            })
        
        return explanations
    
    def _parse_consistency_results(self, result_text: str) -> Dict:
        """Parse consistency validation results."""
        try:
            # Look for consistency score
            consistency_score = 0.5
            
            import re
            score_matches = re.findall(r'consistency score[:\s]*([0-9.]+)', result_text.lower())
            if score_matches:
                consistency_score = float(score_matches[0])
            
            return {
                'consistency_score': consistency_score,
                'validation_summary': result_text[:300] + "..." if len(result_text) > 300 else result_text,
                'consistent_findings': [],
                'inconsistent_findings': []
            }
            
        except Exception:
            return {
                'consistency_score': 0.5,
                'validation_summary': 'Consistency analysis completed with mixed results'
            }
