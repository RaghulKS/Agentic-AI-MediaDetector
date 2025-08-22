import os
import sys
import argparse
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from planner_agent import PlannerAgent
from classifier_agent import ClassifierAgent
from forensics_agent import ForensicsAgent
from retriever_agent import RetrieverAgent
from counterfactual_agent import CounterfactualAgent
from explainer_agent import ExplainerAgent
from reporter_agent import ReporterAgent


class ImageAuthenticityDetector:
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.setup_logging()
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.agents = {}
        self.initialize_agents()
        
        self.current_analysis = {}
        self.analysis_results = {}
        
        self.logger.info("Image authenticity detector initialized")
    
    def setup_logging(self):
        log_level = self.config.get('log_level', 'INFO')
        
        os.makedirs('data', exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/authenticity_analysis.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_agents(self):
        try:
            self.agents = {
                'planner': PlannerAgent(self.api_key),
                'classifier': ClassifierAgent(self.api_key),
                'forensics': ForensicsAgent(self.api_key),
                'retriever': RetrieverAgent(self.api_key),
                'counterfactual': CounterfactualAgent(self.api_key),
                'explainer': ExplainerAgent(self.api_key),
                'reporter': ReporterAgent(self.api_key)
            }
            
            self.logger.info("Agents initialized")
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {str(e)}")
            raise
    
    def analyze_image(self, image_path: str, user_requirements: Dict = None) -> Dict:
        try:
            start_time = time.time()
            self.logger.info(f"Starting analysis for: {image_path}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            self.current_analysis = {
                'image_path': image_path,
                'start_time': start_time,
                'user_requirements': user_requirements or {},
                'completed_tasks': [],
                'intermediate_results': {}
            }
            
            self.logger.info("Creating analysis plan...")
            analysis_plan = self.agents['planner'].create_analysis_plan(image_path, user_requirements)
            
            if 'error' in analysis_plan:
                self.logger.warning(f"Planning error: {analysis_plan['error']}")
                analysis_plan = analysis_plan.get('fallback_plan', {})
            
            self.current_analysis['analysis_plan'] = analysis_plan
            self.logger.info(f"Plan created with {len(analysis_plan.get('workflow_steps', []))} steps")
            
            self.analysis_results = self._execute_analysis_workflow(analysis_plan)
            
            overall_score = self._calculate_overall_authenticity_score(self.analysis_results)
            self.analysis_results['overall_authenticity_score'] = overall_score
            
            self.logger.info("Generating explanation...")
            explanation_results = self.agents['explainer'].generate_explanation(
                self.analysis_results,
                self.config.get('explanation_config')
            )
            self.analysis_results['explanation'] = explanation_results
            
            self.logger.info("Generating report...")
            report_results = self.agents['reporter'].generate_report(
                image_path,
                self.analysis_results,
                self.config.get('report_config')
            )
            self.analysis_results['report'] = report_results
            
            end_time = time.time()
            analysis_duration = end_time - start_time
            
            self.analysis_results.update({
                'analysis_completed': True,
                'analysis_duration': analysis_duration,
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            })
            
            self.logger.info(f"Analysis completed in {analysis_duration:.2f} seconds")
            self.logger.info(f"Authenticity score: {overall_score:.2%}")
            
            self._save_analysis_results()
            
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {
                'error': str(e),
                'analysis_completed': False,
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_analysis_workflow(self, analysis_plan: Dict) -> Dict:
        """Execute the planned analysis workflow."""
        results = {
            'vision_analysis': {},
            'metadata_analysis': {},
            'search_analysis': {},
            'counterfactual_analysis': {},
            'workflow_execution': {}
        }
        
        try:
            workflow_steps = analysis_plan.get('workflow_steps', [])
            completed_tasks = []
            
            self.logger.info(f"Executing {len(workflow_steps)} workflow steps...")
            
            for step in workflow_steps:
                step_number = step.get('step', 0)
                agent_name = step.get('agent', '')
                task_name = step.get('task', '')
                dependencies = step.get('dependencies', [])
                
                self.logger.info(f"Step {step_number}: {task_name} using {agent_name}")
                
                # Check dependencies
                if not all(dep in completed_tasks for dep in dependencies):
                    missing_deps = [dep for dep in dependencies if dep not in completed_tasks]
                    self.logger.warning(f"Missing dependencies: {missing_deps}")
                    continue
                
                # Execute the task
                step_result = self._execute_analysis_step(agent_name, task_name, step)
                
                if step_result.get('analysis_completed', False) or step_result.get('search_completed', False):
                    completed_tasks.append(task_name)
                    self.logger.info(f"Step {step_number} completed successfully")
                else:
                    self.logger.warning(f"Step {step_number} completed with issues")
                
                # Store results
                if task_name == 'vision_analysis':
                    results['vision_analysis'] = step_result
                elif task_name == 'metadata_forensics':
                    results['metadata_analysis'] = step_result
                elif task_name == 'reverse_search':
                    results['search_analysis'] = step_result
                elif task_name == 'counterfactual_generation':
                    results['counterfactual_analysis'] = step_result
            
            results['workflow_execution'] = {
                'completed_steps': len(completed_tasks),
                'total_steps': len(workflow_steps),
                'completed_tasks': completed_tasks,
                'success_rate': len(completed_tasks) / len(workflow_steps) if workflow_steps else 0
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            results['workflow_execution'] = {'error': str(e)}
            return results
    
    def _execute_analysis_step(self, agent_name: str, task_name: str, step_config: Dict) -> Dict:
        """Execute a single analysis step."""
        try:
            image_path = self.current_analysis['image_path']
            
            if agent_name == 'ClassifierAgent':
                return self.agents['classifier'].perform_vision_analysis(
                    image_path,
                    step_config
                )
            elif agent_name == 'ForensicsAgent':
                return self.agents['forensics'].perform_forensics_analysis(
                    image_path,
                    step_config
                )
            elif agent_name == 'RetrieverAgent':
                return self.agents['retriever'].perform_search_analysis(
                    image_path,
                    step_config
                )
            elif agent_name == 'CounterfactualAgent':
                return self.agents['counterfactual'].perform_counterfactual_analysis(
                    image_path,
                    self.analysis_results,  # Pass intermediate results
                    step_config
                )
            else:
                return {'error': f"Unknown agent: {agent_name}"}
                
        except Exception as e:
            return {'error': f"Step execution failed: {str(e)}"}
    
    def _calculate_overall_authenticity_score(self, analysis_results: Dict) -> float:
        """Calculate weighted overall authenticity score."""
        try:
            scores = []
            weights = []
            
            # Vision analysis (weight: 0.35)
            vision_results = analysis_results.get('vision_analysis', {})
            if 'overall_authenticity_score' in vision_results:
                scores.append(vision_results['overall_authenticity_score'])
                weights.append(0.35)
            
            # Metadata analysis (weight: 0.25)
            metadata_results = analysis_results.get('metadata_analysis', {})
            if 'authenticity_score' in metadata_results:
                scores.append(metadata_results['authenticity_score'])
                weights.append(0.25)
            
            # Search analysis (weight: 0.20)
            search_results = analysis_results.get('search_analysis', {})
            search_score = search_results.get('authenticity_indicators', {}).get('source_authenticity_score', None)
            if search_score is not None:
                scores.append(search_score)
                weights.append(0.20)
            
            # Counterfactual consistency (weight: 0.20)
            counterfactual_results = analysis_results.get('counterfactual_analysis', {})
            if 'consistency_score' in counterfactual_results:
                consistency_score = counterfactual_results['consistency_score']
                scores.append(consistency_score)
                weights.append(0.20)
            
            if not scores:
                return 0.5  # Neutral score if no analysis completed
            
            # Calculate weighted average
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5
            
            # Apply counterfactual adjustment if available
            counterfactual_adjustment = counterfactual_results.get('confidence_adjustment', {})
            if 'adjusted_confidence' in counterfactual_adjustment:
                adjusted_score = counterfactual_adjustment['adjusted_confidence']
                # Blend original and adjusted scores
                overall_score = 0.7 * overall_score + 0.3 * adjusted_score
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            self.logger.error(f"Score calculation failed: {str(e)}")
            return 0.5
    
    def _save_analysis_results(self):
        """Save analysis results to file."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = os.path.splitext(os.path.basename(self.current_analysis['image_path']))[0]
            filename = f"analysis_{image_name}_{timestamp}.json"
            
            filepath = os.path.join('data', filename)
            
            # Prepare data for saving (remove non-serializable items)
            save_data = {}
            for key, value in self.analysis_results.items():
                try:
                    json.dumps(value)  # Test if serializable
                    save_data[key] = value
                except (TypeError, ValueError):
                    save_data[key] = str(value)  # Convert to string if not serializable
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            self.logger.info(f"Analysis results saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {str(e)}")
    
    def get_analysis_summary(self) -> Dict:
        """Get a concise summary of the analysis results."""
        if not self.analysis_results:
            return {'error': 'No analysis results available'}
        
        try:
            summary = {
                'image_path': self.analysis_results.get('image_path'),
                'overall_authenticity_score': self.analysis_results.get('overall_authenticity_score', 0.5),
                'authenticity_assessment': self._get_authenticity_assessment(),
                'confidence_level': self._get_confidence_level(),
                'key_findings': self._extract_key_findings_summary(),
                'recommendations': self._extract_recommendations_summary(),
                'analysis_duration': self.analysis_results.get('analysis_duration', 0),
                'report_path': self.analysis_results.get('report', {}).get('report_file_path', '')
            }
            
            return summary
            
        except Exception as e:
            return {'error': f"Summary generation failed: {str(e)}"}
    
    def _get_authenticity_assessment(self) -> str:
        """Get text assessment of authenticity."""
        score = self.analysis_results.get('overall_authenticity_score', 0.5)
        
        if score >= 0.75:
            return "Highly likely authentic"
        elif score >= 0.6:
            return "Likely authentic"
        elif score >= 0.4:
            return "Uncertain - requires further investigation"
        elif score >= 0.25:
            return "Likely AI-generated or manipulated"
        else:
            return "Highly likely AI-generated or manipulated"
    
    def _get_confidence_level(self) -> str:
        """Get overall confidence level."""
        explanation = self.analysis_results.get('explanation', {})
        confidence_assessment = explanation.get('confidence_assessment', {})
        return confidence_assessment.get('confidence_level', 'medium')
    
    def _extract_key_findings_summary(self) -> List[str]:
        """Extract key findings for summary."""
        explanation = self.analysis_results.get('explanation', {})
        return explanation.get('key_findings', [])[:3]  # Top 3 findings
    
    def _extract_recommendations_summary(self) -> List[str]:
        """Extract recommendations for summary."""
        explanation = self.analysis_results.get('explanation', {})
        return explanation.get('recommendations', [])[:2]  # Top 2 recommendations


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description='Agentic AI Image Authenticity Detector')
    parser.add_argument('image_path', help='Path to the image file to analyze')
    parser.add_argument('--config', help='Path to configuration file (JSON)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--output-dir', default='reports', help='Output directory for reports')
    parser.add_argument('--requirements', help='JSON string with user requirements')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = {'log_level': args.log_level}
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config.update(json.load(f))
        
        # Parse user requirements
        user_requirements = {}
        if args.requirements:
            user_requirements = json.loads(args.requirements)
        
        # Initialize detector
        detector = ImageAuthenticityDetector(config)
        
        # Perform analysis
        print(f"Analyzing image: {args.image_path}")
        print("This may take several minutes...")
        
        results = detector.analyze_image(args.image_path, user_requirements)
        
        if results.get('error'):
            print(f"Analysis failed: {results['error']}")
            return 1
        
        # Print summary
        summary = detector.get_analysis_summary()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Image: {os.path.basename(summary['image_path'])}")
        print(f"Overall Authenticity Score: {summary['overall_authenticity_score']:.2%}")
        print(f"Assessment: {summary['authenticity_assessment']}")
        print(f"Confidence Level: {summary['confidence_level']}")
        print(f"Analysis Duration: {summary['analysis_duration']:.2f} seconds")
        
        if summary.get('key_findings'):
            print("\nKey Findings:")
            for i, finding in enumerate(summary['key_findings'], 1):
                print(f"  {i}. {finding}")
        
        if summary.get('recommendations'):
            print("\nRecommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        if summary.get('report_path'):
            print(f"\nFull report generated: {summary['report_path']}")
        
        print("\n" + "="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
