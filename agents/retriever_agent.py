"""
Retriever Agent - Performs reverse image search and context retrieval.
Uses search engines to find similar images, sources, and contextual information.
"""

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional
import os
import sys

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

from search_tools import ImageSearchEngine, ContextRetriever


class RetrieverAgent:
    """
    Specialized agent for reverse image search and contextual information retrieval.
    Focuses on finding sources, similar images, and relevant context for authenticity assessment.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the retriever agent."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=self.api_key
        )
        
        # Initialize search tools
        self.search_engine = ImageSearchEngine()
        self.context_retriever = ContextRetriever()
        
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the retriever agent with CrewAI."""
        return Agent(
            role="Image Source and Context Research Specialist",
            goal="Conduct comprehensive reverse image searches and gather contextual information to verify image authenticity and origins",
            backstory="""You are an expert digital investigator specializing in image source verification and 
            contextual research. Your expertise includes reverse image search techniques, source credibility assessment, 
            cross-referencing multiple databases, and analyzing contextual information to determine image authenticity. 
            You understand how to trace image origins, identify original sources, detect manipulated versions, and 
            assess the reliability of different information sources. Your research provides crucial context for 
            authenticity determination.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def perform_search_analysis(self, image_path: str, search_config: Dict = None) -> Dict:
        """
        Perform comprehensive search and retrieval analysis.
        
        Args:
            image_path: Path to the image file
            search_config: Optional configuration for search parameters
            
        Returns:
            Dict containing search analysis results
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                return {
                    'error': f"Image file not found: {image_path}",
                    'search_completed': False,
                    'authenticity_indicators': {'source_authenticity_score': 0.5}
                }
            
            results = {
                'analysis_type': 'reverse_search_and_context',
                'image_path': image_path,
                'search_completed': False,
                'reverse_search_results': {},
                'context_analysis': {},
                'source_verification': {},
                'credibility_assessment': {},
                'authenticity_indicators': {}
            }
            
            # Perform reverse image search
            search_results = self.search_engine.perform_reverse_search(image_path)
            results['reverse_search_results'] = search_results
            
            # Extract search terms for context retrieval
            search_terms = self._extract_search_terms_from_results(search_results)
            
            # Perform context retrieval
            context_results = self.context_retriever.get_image_context(image_path, search_terms)
            results['context_analysis'] = context_results
            
            # Use AI agent for search interpretation and analysis
            search_interpretation = self._interpret_search_results(results, search_config)
            results.update(search_interpretation)
            
            results['search_completed'] = True
            
            return results
            
        except Exception as e:
            return {
                'error': f"Search analysis failed: {str(e)}",
                'search_completed': False,
                'authenticity_indicators': {'source_authenticity_score': 0.5},
                'image_path': image_path
            }
    
    def _extract_search_terms_from_results(self, search_results: Dict) -> List[str]:
        """Extract relevant search terms from initial search results."""
        search_terms = []
        
        # Extract from context information
        context = search_results.get('context_information', {})
        topics = context.get('topics', [])
        locations = context.get('locations', [])
        
        search_terms.extend(topics[:3])  # Limit to top 3 topics
        search_terms.extend(locations[:2])  # Limit to top 2 locations
        
        # Add generic authenticity terms
        search_terms.extend(['image verification', 'photo authenticity'])
        
        return search_terms
    
    def _interpret_search_results(self, search_data: Dict, config: Dict = None) -> Dict:
        """Use AI agent to interpret search results and provide analysis."""
        try:
            # Create search interpretation task
            interpretation_task = Task(
                description=f"""
                As an expert digital investigator, analyze the following reverse image search and context results 
                to assess image authenticity and source credibility:
                
                REVERSE SEARCH RESULTS:
                {self._format_search_results_for_prompt(search_data.get('reverse_search_results', {}))}
                
                CONTEXT ANALYSIS:
                {self._format_context_results_for_prompt(search_data.get('context_analysis', {}))}
                
                Search Configuration: {config or 'Standard search analysis'}
                
                Provide comprehensive analysis including:
                1. Assessment of source credibility and reliability
                2. Analysis of search result patterns and their significance
                3. Evaluation of contextual information relevance and consistency
                4. Identification of original sources vs. republished content
                5. Detection of potential image manipulation or generation indicators
                6. Cross-reference validation between different sources
                7. Overall source-based authenticity score (0.0-1.0)
                8. Confidence level in the search-based findings
                9. Key evidence points from search results
                10. Red flags or suspicious patterns in search results
                11. Recommendations for additional verification steps
                
                Consider:
                - Source reputation and credibility factors
                - Temporal patterns in search results
                - Geographic distribution of sources
                - Consistency of contextual information
                - Signs of viral or manipulated content spread
                """,
                agent=self.agent,
                expected_output="Comprehensive search analysis with authenticity assessment, credibility evaluation, and evidence summary"
            )
            
            # Execute search interpretation
            crew = Crew(
                agents=[self.agent],
                tasks=[interpretation_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse interpretation result
            interpretation = self._parse_search_interpretation(str(result))
            
            return {
                'search_interpretation': interpretation,
                'source_credibility_assessment': interpretation.get('credibility_assessment', {}),
                'authenticity_indicators': interpretation.get('authenticity_indicators', {'source_authenticity_score': 0.5}),
                'confidence_level': interpretation.get('confidence_level', 'medium'),
                'key_findings': interpretation.get('key_findings', []),
                'red_flags': interpretation.get('red_flags', []),
                'positive_indicators': interpretation.get('positive_indicators', []),
                'recommendations': interpretation.get('recommendations', [])
            }
            
        except Exception as e:
            return {
                'interpretation_error': f"Search interpretation failed: {str(e)}",
                'authenticity_indicators': {'source_authenticity_score': 0.5},
                'confidence_level': 'low'
            }
    
    def _format_search_results_for_prompt(self, search_results: Dict) -> str:
        """Format search results for AI prompt."""
        if not search_results:
            return "No reverse search results available"
        
        formatted = []
        
        # Search results summary
        results_list = search_results.get('search_results', [])
        formatted.append(f"Total Search Results Found: {len(results_list)}")
        
        # Source analysis
        source_analysis = search_results.get('source_analysis', {})
        if source_analysis:
            formatted.append(f"Unique Domains: {len(source_analysis.get('unique_domains', []))}")
            formatted.append(f"Credible Sources: {source_analysis.get('credible_sources', 0)}")
            formatted.append(f"Suspicious Sources: {source_analysis.get('suspicious_sources', 0)}")
            
            # Domain frequency
            domain_freq = source_analysis.get('domain_frequency', {})
            if domain_freq:
                formatted.append("Top Domains:")
                for domain, count in list(domain_freq.items())[:5]:
                    formatted.append(f"  {domain}: {count} results")
        
        # Credibility assessment
        credibility = search_results.get('credibility_assessment', {})
        if credibility:
            overall_score = credibility.get('overall_score', 0)
            formatted.append(f"Overall Credibility Score: {overall_score:.3f}")
            
            red_flags = credibility.get('red_flags', [])
            if red_flags:
                formatted.append("Red Flags Detected:")
                for flag in red_flags[:3]:  # Limit to top 3
                    formatted.append(f"  - {flag}")
            
            positive_indicators = credibility.get('positive_indicators', [])
            if positive_indicators:
                formatted.append("Positive Indicators:")
                for indicator in positive_indicators[:3]:  # Limit to top 3
                    formatted.append(f"  - {indicator}")
        
        # Context information
        context = search_results.get('context_information', {})
        if context:
            topics = context.get('topics', [])
            locations = context.get('locations', [])
            
            if topics:
                formatted.append(f"Identified Topics: {', '.join(topics[:5])}")
            if locations:
                formatted.append(f"Associated Locations: {', '.join(locations[:5])}")
        
        return "\n".join(formatted) if formatted else "Minimal search result information"
    
    def _format_context_results_for_prompt(self, context_results: Dict) -> str:
        """Format context results for AI prompt."""
        if not context_results:
            return "No context analysis available"
        
        formatted = []
        
        # Search terms used
        search_terms = context_results.get('search_terms_used', [])
        if search_terms:
            formatted.append(f"Context Search Terms: {', '.join(search_terms)}")
        
        # Contextual results
        contextual_results = context_results.get('contextual_results', [])
        formatted.append(f"Contextual Results Found: {len(contextual_results)}")
        
        # Relevance score
        relevance_score = context_results.get('relevance_score', 0)
        formatted.append(f"Context Relevance Score: {relevance_score:.3f}")
        
        # Sample contextual information
        if contextual_results:
            formatted.append("Sample Contextual Information:")
            for i, result in enumerate(contextual_results[:3]):  # Top 3
                title = result.get('title', 'No title')[:50]
                formatted.append(f"  {i+1}. {title}...")
        
        return "\n".join(formatted) if formatted else "Limited contextual information"
    
    def _parse_search_interpretation(self, result_text: str) -> Dict:
        """Parse search interpretation result into structured format."""
        interpretation = {
            'summary': result_text[:400] + "..." if len(result_text) > 400 else result_text,
            'credibility_assessment': {'overall_credible': True, 'credibility_score': 0.5},
            'authenticity_indicators': {'source_authenticity_score': 0.5},
            'confidence_level': 'medium',
            'key_findings': [],
            'red_flags': [],
            'positive_indicators': [],
            'recommendations': []
        }
        
        text_lower = result_text.lower()
        
        # Extract authenticity score
        score_patterns = ['authenticity score:', 'source score:', 'credibility score:']
        for pattern in score_patterns:
            if pattern in text_lower:
                try:
                    start_idx = text_lower.find(pattern) + len(pattern)
                    score_section = text_lower[start_idx:start_idx + 20]
                    import re
                    numbers = re.findall(r'0\.\d+|\d\.\d+', score_section)
                    if numbers:
                        score = float(numbers[0])
                        interpretation['authenticity_indicators']['source_authenticity_score'] = score
                        interpretation['credibility_assessment']['credibility_score'] = score
                        break
                except:
                    pass
        
        # Determine confidence level
        if 'high confidence' in text_lower or 'very confident' in text_lower:
            interpretation['confidence_level'] = 'high'
        elif 'low confidence' in text_lower or 'uncertain' in text_lower:
            interpretation['confidence_level'] = 'low'
        
        # Extract findings
        if 'credible source' in text_lower or 'reputable' in text_lower:
            interpretation['positive_indicators'].append("Found on credible sources")
        
        if 'suspicious source' in text_lower or 'questionable' in text_lower:
            interpretation['red_flags'].append("Found on suspicious sources")
        
        if 'original source' in text_lower:
            interpretation['positive_indicators'].append("Original source identified")
        
        if 'ai-generated' in text_lower or 'artificial' in text_lower:
            interpretation['red_flags'].append("Sources suggest AI generation")
        
        if 'no matches' in text_lower or 'limited results' in text_lower:
            interpretation['red_flags'].append("Limited or no reverse search matches")
        
        # Extract recommendations
        if 'additional verification' in text_lower:
            interpretation['recommendations'].append("Conduct additional source verification")
        
        if 'cross-reference' in text_lower:
            interpretation['recommendations'].append("Cross-reference with multiple sources")
        
        # Overall credibility assessment
        credibility_score = interpretation['authenticity_indicators']['source_authenticity_score']
        interpretation['credibility_assessment']['overall_credible'] = credibility_score > 0.5
        
        return interpretation
    
    def verify_specific_sources(self, image_url: str, suspected_sources: List[str]) -> Dict:
        """
        Verify image against specific suspected sources.
        
        Args:
            image_url: URL of the image to verify
            suspected_sources: List of suspected source URLs
            
        Returns:
            Dict containing source verification results
        """
        try:
            verification_results = []
            
            for source_url in suspected_sources:
                verification = self.context_retriever.verify_image_source(source_url)
                verification['suspected_source'] = source_url
                verification_results.append(verification)
            
            # Create verification analysis task
            verification_task = Task(
                description=f"""
                Analyze the verification results for specific suspected sources of an image:
                
                Image URL: {image_url}
                Suspected Sources: {suspected_sources}
                Verification Results: {verification_results}
                
                Determine:
                1. Which sources are most likely to be original
                2. Evidence of source authenticity or manipulation
                3. Temporal analysis of source appearances
                4. Credibility ranking of sources
                5. Recommendations for source validation
                """,
                agent=self.agent,
                expected_output="Source verification analysis with authenticity assessment and rankings"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[verification_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'verification_analysis': str(result),
                'suspected_sources': suspected_sources,
                'verification_results': verification_results,
                'analysis_type': 'specific_source_verification'
            }
            
        except Exception as e:
            return {
                'error': f"Source verification failed: {str(e)}",
                'suspected_sources': suspected_sources
            }
    
    def analyze_source_timeline(self, search_results: Dict) -> Dict:
        """
        Analyze the timeline of source appearances to detect original vs. copies.
        
        Args:
            search_results: Results from reverse image search
            
        Returns:
            Dict containing timeline analysis
        """
        try:
            timeline_task = Task(
                description=f"""
                Analyze the timeline and temporal patterns of image appearances across different sources:
                
                Search Results: {search_results}
                
                Determine:
                1. Earliest appearance date and source
                2. Pattern of image spread across sources
                3. Indicators of viral or manipulated content distribution
                4. Temporal inconsistencies that suggest manipulation
                5. Likelihood of identifying the original source
                
                Provide timeline analysis with authenticity implications.
                """,
                agent=self.agent,
                expected_output="Timeline analysis with original source assessment and distribution pattern analysis"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[timeline_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'timeline_analysis': str(result),
                'analysis_type': 'source_timeline'
            }
            
        except Exception as e:
            return {
                'error': f"Timeline analysis failed: {str(e)}"
            }
    
    def assess_source_diversity(self, search_results: Dict) -> Dict:
        """
        Assess the diversity and distribution of sources for authenticity insights.
        
        Args:
            search_results: Results from reverse image search
            
        Returns:
            Dict containing source diversity analysis
        """
        try:
            # Extract source information
            source_analysis = search_results.get('source_analysis', {})
            unique_domains = source_analysis.get('unique_domains', [])
            source_types = source_analysis.get('source_types', {})
            
            diversity_metrics = {
                'total_unique_sources': len(unique_domains),
                'source_type_distribution': source_types,
                'diversity_score': 0.0,
                'concentration_risk': 'low'
            }
            
            # Calculate diversity score
            if diversity_metrics['total_unique_sources'] > 0:
                # Higher diversity generally indicates more authentic spread
                diversity_score = min(diversity_metrics['total_unique_sources'] / 10.0, 1.0)
                diversity_metrics['diversity_score'] = diversity_score
                
                # Assess concentration risk
                if diversity_metrics['total_unique_sources'] < 3:
                    diversity_metrics['concentration_risk'] = 'high'
                elif diversity_metrics['total_unique_sources'] < 6:
                    diversity_metrics['concentration_risk'] = 'medium'
                else:
                    diversity_metrics['concentration_risk'] = 'low'
            
            # Assess source type balance
            news_sources = source_types.get('news_media', 0)
            social_sources = source_types.get('social_media', 0)
            ai_sources = source_types.get('ai_related', 0)
            
            diversity_metrics['source_balance'] = {
                'news_media_ratio': news_sources / max(diversity_metrics['total_unique_sources'], 1),
                'social_media_ratio': social_sources / max(diversity_metrics['total_unique_sources'], 1),
                'ai_related_ratio': ai_sources / max(diversity_metrics['total_unique_sources'], 1)
            }
            
            # Risk assessment based on AI source ratio
            if diversity_metrics['source_balance']['ai_related_ratio'] > 0.5:
                diversity_metrics['ai_generation_risk'] = 'high'
            elif diversity_metrics['source_balance']['ai_related_ratio'] > 0.2:
                diversity_metrics['ai_generation_risk'] = 'medium'
            else:
                diversity_metrics['ai_generation_risk'] = 'low'
            
            return diversity_metrics
            
        except Exception as e:
            return {
                'error': f"Source diversity analysis failed: {str(e)}",
                'diversity_score': 0.0,
                'concentration_risk': 'unknown'
            }
