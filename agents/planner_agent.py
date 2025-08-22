from crewai import Agent, Task, Crew
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any
import os
from datetime import datetime


class PlannerAgent:
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=self.api_key
        )
        
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        return Agent(
            role="Image Authenticity Analysis Coordinator",
            goal="Orchestrate comprehensive image authenticity analysis using specialized agents",
            backstory="Expert coordinator for image authenticity detection. Plans systematic analysis combining computer vision, forensics, metadata analysis, and contextual verification to determine if images are authentic, AI-generated, or manipulated.",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
    
    def create_analysis_plan(self, image_path: str, user_requirements: Dict = None) -> Dict:
        """
        Create a comprehensive analysis plan for image authenticity detection.
        
        Args:
            image_path: Path to the image to be analyzed
            user_requirements: Optional specific requirements from user
            
        Returns:
            Dict containing the analysis plan with tasks and workflow
        """
        try:
            # Define the planning task
            planning_task = Task(
                description=f"""
                Create a comprehensive analysis plan for determining the authenticity of an image located at: {image_path}
                
                The analysis should include:
                1. Computer vision analysis using CLIP and ViT models
                2. Metadata extraction and forensic analysis
                3. Reverse image search and context retrieval
                4. Cross-validation using counterfactual generation
                5. Explainable reasoning synthesis
                6. Professional report generation
                
                Consider these factors:
                - Image file properties and format
                - Available metadata richness
                - Computational requirements
                - Analysis confidence levels
                - Risk assessment priorities
                
                User requirements: {user_requirements or 'Standard comprehensive analysis'}
                
                Output a detailed plan with:
                - Task sequence and dependencies
                - Resource allocation
                - Expected confidence levels
                - Risk factors to investigate
                - Success criteria for each analysis component
                """,
                agent=self.agent,
                expected_output="A structured analysis plan with task definitions, sequencing, and success criteria"
            )
            
            # Execute planning
            crew = Crew(
                agents=[self.agent],
                tasks=[planning_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse and structure the result
            analysis_plan = self._parse_planning_result(str(result), image_path)
            
            return analysis_plan
            
        except Exception as e:
            return {
                'error': f"Planning failed: {str(e)}",
                'fallback_plan': self._create_fallback_plan(image_path)
            }
    
    def _parse_planning_result(self, planning_result: str, image_path: str) -> Dict:
        """Parse the planning result into a structured format."""
        
        # Create structured plan (in production, this would parse the LLM output)
        plan = {
            'analysis_id': self._generate_analysis_id(),
            'image_path': image_path,
            'created_at': datetime.now().isoformat(),
            'workflow_steps': [
                {
                    'step': 1,
                    'agent': 'ClassifierAgent',
                    'task': 'vision_analysis',
                    'description': 'Perform computer vision-based authenticity classification',
                    'dependencies': [],
                    'priority': 'high',
                    'estimated_duration': 30
                },
                {
                    'step': 2,
                    'agent': 'ForensicsAgent',
                    'task': 'metadata_forensics',
                    'description': 'Extract and analyze image metadata and EXIF data',
                    'dependencies': [],
                    'priority': 'high',
                    'estimated_duration': 15
                },
                {
                    'step': 3,
                    'agent': 'RetrieverAgent',
                    'task': 'reverse_search',
                    'description': 'Perform reverse image search and context retrieval',
                    'dependencies': [],
                    'priority': 'medium',
                    'estimated_duration': 45
                },
                {
                    'step': 4,
                    'agent': 'CounterfactualAgent',
                    'task': 'counterfactual_generation',
                    'description': 'Generate counterfactual analysis for consistency checking',
                    'dependencies': ['vision_analysis'],
                    'priority': 'medium',
                    'estimated_duration': 60
                },
                {
                    'step': 5,
                    'agent': 'ExplainerAgent',
                    'task': 'explanation_synthesis',
                    'description': 'Synthesize findings into explainable reasoning',
                    'dependencies': ['vision_analysis', 'metadata_forensics', 'reverse_search'],
                    'priority': 'high',
                    'estimated_duration': 20
                },
                {
                    'step': 6,
                    'agent': 'ReporterAgent',
                    'task': 'report_generation',
                    'description': 'Generate comprehensive trust report with visualizations',
                    'dependencies': ['explanation_synthesis'],
                    'priority': 'high',
                    'estimated_duration': 30
                }
            ],
            'success_criteria': {
                'vision_analysis': 'Confidence score > 0.7 from multiple models',
                'metadata_forensics': 'Complete EXIF extraction with anomaly detection',
                'reverse_search': 'At least 3 relevant sources identified',
                'counterfactual_generation': 'Consistency score calculated',
                'explanation_synthesis': 'Human-readable reasoning chain generated',
                'report_generation': 'Professional PDF report with visualizations'
            },
            'risk_factors': [
                'Limited metadata availability',
                'Model confidence variations',
                'Search result relevance',
                'Computational resource constraints'
            ],
            'planning_notes': planning_result
        }
        
        return plan
    
    def _create_fallback_plan(self, image_path: str) -> Dict:
        """Create a basic fallback plan if AI planning fails."""
        return {
            'analysis_id': self._generate_analysis_id(),
            'image_path': image_path,
            'created_at': datetime.now().isoformat(),
            'workflow_steps': [
                {
                    'step': 1,
                    'agent': 'ClassifierAgent',
                    'task': 'vision_analysis',
                    'description': 'Basic vision analysis',
                    'dependencies': [],
                    'priority': 'high'
                },
                {
                    'step': 2,
                    'agent': 'ForensicsAgent',
                    'task': 'metadata_forensics',
                    'description': 'Basic metadata analysis',
                    'dependencies': [],
                    'priority': 'high'
                },
                {
                    'step': 3,
                    'agent': 'ExplainerAgent',
                    'task': 'explanation_synthesis',
                    'description': 'Basic explanation synthesis',
                    'dependencies': ['vision_analysis', 'metadata_forensics'],
                    'priority': 'high'
                },
                {
                    'step': 4,
                    'agent': 'ReporterAgent',
                    'task': 'report_generation',
                    'description': 'Generate basic report',
                    'dependencies': ['explanation_synthesis'],
                    'priority': 'high'
                }
            ],
            'fallback_mode': True
        }
    
    def validate_plan(self, plan: Dict) -> Dict:
        """
        Validate the analysis plan for feasibility and completeness.
        
        Args:
            plan: The analysis plan to validate
            
        Returns:
            Dict containing validation results and any recommended adjustments
        """
        try:
            validation_task = Task(
                description=f"""
                Validate the following image authenticity analysis plan for feasibility and completeness:
                
                {plan}
                
                Check for:
                1. Logical task sequence and dependencies
                2. Resource requirements feasibility
                3. Missing critical analysis components
                4. Realistic time estimates
                5. Appropriate success criteria
                6. Risk mitigation coverage
                
                Provide specific recommendations for improvements or confirm the plan is ready for execution.
                """,
                agent=self.agent,
                expected_output="Validation assessment with specific recommendations or confirmation"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[validation_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'is_valid': True,  # In production, parse the LLM response
                'validation_result': str(result),
                'recommended_adjustments': [],
                'validated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': f"Validation failed: {str(e)}",
                'recommended_action': 'Use fallback plan'
            }
    
    def adjust_plan_based_on_results(self, plan: Dict, intermediate_results: Dict) -> Dict:
        """
        Adjust the analysis plan based on intermediate results.
        
        Args:
            plan: Original analysis plan
            intermediate_results: Results from completed tasks
            
        Returns:
            Dict containing adjusted plan
        """
        try:
            # Analyze intermediate results and adjust remaining tasks
            adjustment_task = Task(
                description=f"""
                Review the current analysis plan and intermediate results to optimize remaining tasks:
                
                Original Plan: {plan}
                Intermediate Results: {intermediate_results}
                
                Consider:
                1. Results quality and confidence from completed tasks
                2. Identified risk factors or anomalies
                3. Resource efficiency for remaining tasks
                4. Potential for enhanced analysis based on findings
                
                Recommend adjustments to:
                - Task priorities
                - Resource allocation
                - Success criteria
                - Additional analysis needed
                """,
                agent=self.agent,
                expected_output="Adjusted plan with reasoning for changes"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[adjustment_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # In production, parse the result and update the plan accordingly
            adjusted_plan = plan.copy()
            adjusted_plan['adjustments'] = {
                'adjusted_at': datetime.now().isoformat(),
                'reasoning': str(result),
                'based_on_results': list(intermediate_results.keys())
            }
            
            return adjusted_plan
            
        except Exception as e:
            # Return original plan if adjustment fails
            plan['adjustment_error'] = str(e)
            return plan
    
    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"IMG_AUTH_{timestamp}"
    
    def get_task_by_step(self, plan: Dict, step: int) -> Dict:
        """Get task details for a specific step."""
        workflow_steps = plan.get('workflow_steps', [])
        for task in workflow_steps:
            if task.get('step') == step:
                return task
        return {}
    
    def get_next_tasks(self, plan: Dict, completed_tasks: List[str]) -> List[Dict]:
        """Get next tasks that can be executed based on completed dependencies."""
        workflow_steps = plan.get('workflow_steps', [])
        next_tasks = []
        
        for task in workflow_steps:
            task_name = task.get('task', '')
            
            # Skip if already completed
            if task_name in completed_tasks:
                continue
            
            # Check if dependencies are met
            dependencies = task.get('dependencies', [])
            if all(dep in completed_tasks for dep in dependencies):
                next_tasks.append(task)
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        next_tasks.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 2), reverse=True)
        
        return next_tasks
    
    def estimate_total_duration(self, plan: Dict) -> int:
        """Estimate total analysis duration in seconds."""
        workflow_steps = plan.get('workflow_steps', [])
        return sum(task.get('estimated_duration', 30) for task in workflow_steps)
