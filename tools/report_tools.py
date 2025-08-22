"""
Report generation tools for image authenticity detection.
Generates comprehensive PDF reports with visualizations and analysis.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, white, red, green, blue, orange, gray
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import io
import base64


class AuthenticityReportGenerator:
    """Generate comprehensive authenticity reports in PDF format."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the report."""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center
        )
        
        # Section header style
        self.section_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Subsection style
        self.subsection_style = ParagraphStyle(
            'SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.darkgreen
        )
        
        # Warning style
        self.warning_style = ParagraphStyle(
            'Warning',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            borderColor=colors.red,
            borderWidth=1,
            borderPadding=6
        )
        
        # Success style
        self.success_style = ParagraphStyle(
            'Success',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.darkgreen,
            borderColor=colors.darkgreen,
            borderWidth=1,
            borderPadding=6
        )
    
    def generate_report(self, image_path: str, analysis_results: Dict, output_filename: str = None) -> str:
        """
        Generate comprehensive authenticity report.
        
        Args:
            image_path: Path to the analyzed image
            analysis_results: Complete analysis results from all agents
            output_filename: Optional custom filename
            
        Returns:
            Path to generated PDF report
        """
        try:
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"authenticity_report_{timestamp}.pdf"
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build report content
            story = self._build_report_content(image_path, analysis_results)
            
            # Generate PDF
            doc.build(story)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Report generation failed: {str(e)}")
    
    def _build_report_content(self, image_path: str, results: Dict) -> List:
        """Build the complete report content."""
        story = []
        
        # Title page
        story.extend(self._create_title_page(image_path, results))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(results))
        story.append(PageBreak())
        
        # Detailed analysis sections
        story.extend(self._create_vision_analysis_section(results.get('vision_analysis', {})))
        story.append(Spacer(1, 20))
        
        story.extend(self._create_metadata_analysis_section(results.get('metadata_analysis', {})))
        story.append(Spacer(1, 20))
        
        story.extend(self._create_search_analysis_section(results.get('search_analysis', {})))
        story.append(Spacer(1, 20))
        
        story.extend(self._create_forensics_section(results.get('forensics_analysis', {})))
        story.append(PageBreak())
        
        # Explanation and reasoning
        story.extend(self._create_explanation_section(results.get('explanation', {})))
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.extend(self._create_recommendations_section(results))
        story.append(Spacer(1, 20))
        
        # Technical appendix
        story.extend(self._create_technical_appendix(results))
        
        return story
    
    def _create_title_page(self, image_path: str, results: Dict) -> List:
        """Create the title page of the report."""
        elements = []
        
        # Title
        elements.append(Paragraph("Image Authenticity Analysis Report", self.title_style))
        elements.append(Spacer(1, 30))
        
        # Overall authenticity score
        overall_score = results.get('overall_authenticity_score', 0.5)
        score_color = self._get_score_color(overall_score)
        
        elements.append(Paragraph(
            f"<font color='{score_color}'>Overall Authenticity Score: {overall_score:.2%}</font>",
            ParagraphStyle('ScoreStyle', parent=self.styles['Normal'], fontSize=18, alignment=1)
        ))
        elements.append(Spacer(1, 30))
        
        # Image preview
        try:
            # Create image preview
            preview_path = self._create_image_preview(image_path)
            if preview_path and os.path.exists(preview_path):
                elements.append(RLImage(preview_path, width=4*inch, height=3*inch))
                elements.append(Spacer(1, 20))
        except:
            pass
        
        # Analysis summary table
        summary_data = [
            ['Analysis Type', 'Score', 'Status'],
            ['Vision Analysis', f"{results.get('vision_analysis', {}).get('overall_authenticity_score', 0.5):.2%}", 
             self._get_status_text(results.get('vision_analysis', {}).get('overall_authenticity_score', 0.5))],
            ['Metadata Analysis', f"{results.get('metadata_analysis', {}).get('authenticity_score', 0.5):.2%}",
             self._get_status_text(results.get('metadata_analysis', {}).get('authenticity_score', 0.5))],
            ['Search Analysis', f"{results.get('search_analysis', {}).get('authenticity_indicators', {}).get('source_authenticity_score', 0.5):.2%}",
             self._get_status_text(results.get('search_analysis', {}).get('authenticity_indicators', {}).get('source_authenticity_score', 0.5))]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 30))
        
        # Report metadata
        elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        elements.append(Paragraph(f"<b>Image Path:</b> {os.path.basename(image_path)}", self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self, results: Dict) -> List:
        """Create executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.section_style))
        
        # Overall assessment
        overall_score = results.get('overall_authenticity_score', 0.5)
        if overall_score >= 0.7:
            assessment = "The image shows strong indicators of authenticity."
            style = self.success_style
        elif overall_score <= 0.3:
            assessment = "The image shows significant indicators of being AI-generated or manipulated."
            style = self.warning_style
        else:
            assessment = "The image authenticity is uncertain and requires further investigation."
            style = self.styles['Normal']
        
        elements.append(Paragraph(assessment, style))
        elements.append(Spacer(1, 15))
        
        # Key findings
        elements.append(Paragraph("Key Findings:", self.subsection_style))
        
        findings = self._extract_key_findings(results)
        for finding in findings:
            elements.append(Paragraph(f"• {finding}", self.styles['Normal']))
        
        elements.append(Spacer(1, 15))
        
        # Confidence level
        confidence = self._calculate_overall_confidence(results)
        elements.append(Paragraph(f"<b>Analysis Confidence Level:</b> {confidence}", self.styles['Normal']))
        
        return elements
    
    def _create_vision_analysis_section(self, vision_results: Dict) -> List:
        """Create vision analysis section."""
        elements = []
        
        elements.append(Paragraph("Computer Vision Analysis", self.section_style))
        
        if not vision_results:
            elements.append(Paragraph("Vision analysis not available.", self.styles['Normal']))
            return elements
        
        # CLIP Analysis
        if 'authenticity_scores' in vision_results and 'clip' in vision_results['authenticity_scores']:
            elements.append(Paragraph("CLIP Model Analysis:", self.subsection_style))
            clip_score = vision_results['authenticity_scores']['clip']
            elements.append(Paragraph(f"Authenticity Score: {clip_score:.2%}", self.styles['Normal']))
            
            if 'model_predictions' in vision_results and 'clip' in vision_results['model_predictions']:
                prediction = vision_results['model_predictions']['clip']
                elements.append(Paragraph(f"Prediction: {prediction}", self.styles['Normal']))
        
        elements.append(Spacer(1, 10))
        
        # ViT Analysis
        if 'authenticity_scores' in vision_results and 'vit' in vision_results['authenticity_scores']:
            elements.append(Paragraph("Vision Transformer Analysis:", self.subsection_style))
            vit_score = vision_results['authenticity_scores']['vit']
            elements.append(Paragraph(f"Authenticity Score: {vit_score:.2%}", self.styles['Normal']))
        
        elements.append(Spacer(1, 10))
        
        # Visual Anomalies
        if 'visual_anomalies' in vision_results:
            elements.append(Paragraph("Visual Anomaly Detection:", self.subsection_style))
            anomalies = vision_results['visual_anomalies']
            
            for anomaly_type, anomaly_data in anomalies.items():
                if isinstance(anomaly_data, dict) and 'score' in anomaly_data:
                    score = anomaly_data['score']
                    suspicious = anomaly_data.get('suspicious', False)
                    status = "Suspicious" if suspicious else "Normal"
                    elements.append(Paragraph(f"{anomaly_type}: {score:.2f} - {status}", self.styles['Normal']))
        
        return elements
    
    def _create_metadata_analysis_section(self, metadata_results: Dict) -> List:
        """Create metadata analysis section."""
        elements = []
        
        elements.append(Paragraph("Metadata Analysis", self.section_style))
        
        if not metadata_results:
            elements.append(Paragraph("Metadata analysis not available.", self.styles['Normal']))
            return elements
        
        # EXIF Data Summary
        if 'authenticity_indicators' in metadata_results:
            elements.append(Paragraph("EXIF Data Summary:", self.subsection_style))
            indicators = metadata_results['authenticity_indicators']
            
            camera_info = "Present" if indicators.get('has_camera_info') else "Missing"
            timestamp = "Present" if indicators.get('has_timestamp') else "Missing"
            gps = "Present" if indicators.get('has_gps_data') else "Missing"
            
            elements.append(Paragraph(f"Camera Information: {camera_info}", self.styles['Normal']))
            elements.append(Paragraph(f"Timestamp Data: {timestamp}", self.styles['Normal']))
            elements.append(Paragraph(f"GPS Data: {gps}", self.styles['Normal']))
            
            if indicators.get('camera_make'):
                elements.append(Paragraph(f"Camera Make: {indicators['camera_make']}", self.styles['Normal']))
            if indicators.get('camera_model'):
                elements.append(Paragraph(f"Camera Model: {indicators['camera_model']}", self.styles['Normal']))
        
        elements.append(Spacer(1, 10))
        
        # Suspicious Patterns
        if 'suspicious_patterns' in metadata_results and metadata_results['suspicious_patterns']:
            elements.append(Paragraph("Suspicious Patterns Detected:", self.subsection_style))
            for pattern in metadata_results['suspicious_patterns']:
                if isinstance(pattern, dict):
                    pattern_type = pattern.get('type', 'Unknown')
                    severity = pattern.get('severity', 'low')
                    elements.append(Paragraph(f"• {pattern_type} (Severity: {severity})", self.warning_style))
        
        elements.append(Spacer(1, 10))
        
        # Metadata Completeness
        if 'metadata_completeness' in metadata_results:
            elements.append(Paragraph("Metadata Completeness:", self.subsection_style))
            completeness = metadata_results['metadata_completeness']
            
            if 'overall' in completeness:
                overall = completeness['overall']
                percentage = overall.get('percentage', 0)
                elements.append(Paragraph(f"Overall Completeness: {percentage:.1f}%", self.styles['Normal']))
        
        return elements
    
    def _create_search_analysis_section(self, search_results: Dict) -> List:
        """Create search analysis section."""
        elements = []
        
        elements.append(Paragraph("Reverse Search Analysis", self.section_style))
        
        if not search_results:
            elements.append(Paragraph("Search analysis not available.", self.styles['Normal']))
            return elements
        
        # Source Analysis
        if 'source_analysis' in search_results:
            elements.append(Paragraph("Source Analysis:", self.subsection_style))
            source_analysis = search_results['source_analysis']
            
            total_sources = source_analysis.get('total_sources', 0)
            unique_domains = len(source_analysis.get('unique_domains', []))
            credible_sources = source_analysis.get('credible_sources', 0)
            
            elements.append(Paragraph(f"Total Sources Found: {total_sources}", self.styles['Normal']))
            elements.append(Paragraph(f"Unique Domains: {unique_domains}", self.styles['Normal']))
            elements.append(Paragraph(f"Credible Sources: {credible_sources}", self.styles['Normal']))
        
        elements.append(Spacer(1, 10))
        
        # Credibility Assessment
        if 'credibility_assessment' in search_results:
            elements.append(Paragraph("Credibility Assessment:", self.subsection_style))
            credibility = search_results['credibility_assessment']
            
            overall_score = credibility.get('overall_score', 0.5)
            elements.append(Paragraph(f"Source Credibility Score: {overall_score:.2%}", self.styles['Normal']))
            
            # Red flags
            red_flags = credibility.get('red_flags', [])
            if red_flags:
                elements.append(Paragraph("Red Flags:", self.styles['Normal']))
                for flag in red_flags[:5]:  # Limit to 5
                    elements.append(Paragraph(f"• {flag}", self.warning_style))
            
            # Positive indicators
            positive_indicators = credibility.get('positive_indicators', [])
            if positive_indicators:
                elements.append(Paragraph("Positive Indicators:", self.styles['Normal']))
                for indicator in positive_indicators[:5]:  # Limit to 5
                    elements.append(Paragraph(f"• {indicator}", self.success_style))
        
        return elements
    
    def _create_forensics_section(self, forensics_results: Dict) -> List:
        """Create forensics analysis section."""
        elements = []
        
        elements.append(Paragraph("Digital Forensics Analysis", self.section_style))
        
        if not forensics_results:
            elements.append(Paragraph("Forensics analysis not available.", self.styles['Normal']))
            return elements
        
        # Add forensics content based on results
        elements.append(Paragraph("Forensics analysis completed with technical indicators.", self.styles['Normal']))
        
        return elements
    
    def _create_explanation_section(self, explanation_results: Dict) -> List:
        """Create explanation section."""
        elements = []
        
        elements.append(Paragraph("Analysis Explanation", self.section_style))
        
        if not explanation_results:
            elements.append(Paragraph("No detailed explanation available.", self.styles['Normal']))
            return elements
        
        # Add explanation content
        if 'summary' in explanation_results:
            elements.append(Paragraph(explanation_results['summary'], self.styles['Normal']))
        
        if 'reasoning' in explanation_results:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Reasoning Chain:", self.subsection_style))
            reasoning = explanation_results['reasoning']
            if isinstance(reasoning, list):
                for step in reasoning:
                    elements.append(Paragraph(f"• {step}", self.styles['Normal']))
            else:
                elements.append(Paragraph(str(reasoning), self.styles['Normal']))
        
        return elements
    
    def _create_recommendations_section(self, results: Dict) -> List:
        """Create recommendations section."""
        elements = []
        
        elements.append(Paragraph("Recommendations", self.section_style))
        
        overall_score = results.get('overall_authenticity_score', 0.5)
        
        if overall_score >= 0.7:
            elements.append(Paragraph("The image appears authentic with high confidence.", self.success_style))
            elements.append(Paragraph("Recommendations:", self.subsection_style))
            elements.append(Paragraph("• The image can likely be trusted for most purposes", self.styles['Normal']))
            elements.append(Paragraph("• Consider the source and context when making final decisions", self.styles['Normal']))
        elif overall_score <= 0.3:
            elements.append(Paragraph("The image shows strong indicators of being artificial or manipulated.", self.warning_style))
            elements.append(Paragraph("Recommendations:", self.subsection_style))
            elements.append(Paragraph("• Exercise extreme caution when using this image", self.styles['Normal']))
            elements.append(Paragraph("• Seek additional verification if authenticity is critical", self.styles['Normal']))
            elements.append(Paragraph("• Consider the image potentially AI-generated or manipulated", self.styles['Normal']))
        else:
            elements.append(Paragraph("Image authenticity is uncertain.", self.warning_style))
            elements.append(Paragraph("Recommendations:", self.subsection_style))
            elements.append(Paragraph("• Conduct additional verification if authenticity is important", self.styles['Normal']))
            elements.append(Paragraph("• Consider the context and source of the image", self.styles['Normal']))
            elements.append(Paragraph("• Use with appropriate caution", self.styles['Normal']))
        
        return elements
    
    def _create_technical_appendix(self, results: Dict) -> List:
        """Create technical appendix."""
        elements = []
        
        elements.append(Paragraph("Technical Appendix", self.section_style))
        
        # Analysis methodology
        elements.append(Paragraph("Analysis Methodology:", self.subsection_style))
        elements.append(Paragraph("This report was generated using a multi-agent AI system that combines:", self.styles['Normal']))
        elements.append(Paragraph("• Computer vision models (CLIP, ViT)", self.styles['Normal']))
        elements.append(Paragraph("• Metadata extraction and analysis", self.styles['Normal']))
        elements.append(Paragraph("• Reverse image search", self.styles['Normal']))
        elements.append(Paragraph("• Digital forensics techniques", self.styles['Normal']))
        
        elements.append(Spacer(1, 15))
        
        # Technical details
        elements.append(Paragraph("Technical Details:", self.subsection_style))
        elements.append(Paragraph("Report generated by Agentic AI Image Authenticity Detector", self.styles['Normal']))
        elements.append(Paragraph(f"Analysis timestamp: {datetime.now().isoformat()}", self.styles['Normal']))
        
        return elements
    
    def _create_image_preview(self, image_path: str) -> str:
        """Create a preview image for the report."""
        try:
            with Image.open(image_path) as img:
                # Resize for report
                img.thumbnail((400, 300), Image.Resampling.LANCZOS)
                
                # Save preview
                preview_path = os.path.join(self.output_dir, "temp_preview.jpg")
                img.save(preview_path, "JPEG")
                
                return preview_path
        except:
            return None
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on authenticity score."""
        if score >= 0.7:
            return 'green'
        elif score <= 0.3:
            return 'red'
        else:
            return 'orange'
    
    def _get_status_text(self, score: float) -> str:
        """Get status text based on score."""
        if score >= 0.7:
            return "Likely Authentic"
        elif score <= 0.3:
            return "Likely Artificial"
        else:
            return "Uncertain"
    
    def _extract_key_findings(self, results: Dict) -> List[str]:
        """Extract key findings from analysis results."""
        findings = []
        
        overall_score = results.get('overall_authenticity_score', 0.5)
        
        # Vision analysis findings
        vision_results = results.get('vision_analysis', {})
        if vision_results.get('authenticity_scores', {}).get('clip', 0) < 0.3:
            findings.append("CLIP model indicates likely AI generation")
        
        # Metadata findings
        metadata_results = results.get('metadata_analysis', {})
        if metadata_results.get('suspicious_patterns'):
            findings.append("Suspicious patterns detected in metadata")
        
        # Search findings
        search_results = results.get('search_analysis', {})
        credibility = search_results.get('credibility_assessment', {})
        if credibility.get('overall_score', 0.5) < 0.3:
            findings.append("Low credibility sources found in reverse search")
        
        # Overall assessment
        if overall_score >= 0.7:
            findings.append("Multiple indicators support image authenticity")
        elif overall_score <= 0.3:
            findings.append("Multiple indicators suggest artificial generation")
        
        if not findings:
            findings.append("Mixed indicators require further analysis")
        
        return findings
    
    def _calculate_overall_confidence(self, results: Dict) -> str:
        """Calculate overall confidence level."""
        # Simple confidence calculation based on available data
        data_sources = 0
        if results.get('vision_analysis'):
            data_sources += 1
        if results.get('metadata_analysis'):
            data_sources += 1
        if results.get('search_analysis'):
            data_sources += 1
        
        if data_sources >= 3:
            return "High"
        elif data_sources >= 2:
            return "Medium"
        else:
            return "Low"


class VisualizationGenerator:
    """Generate visualizations for the report."""
    
    @staticmethod
    def create_score_chart(scores: Dict, output_path: str):
        """Create a chart showing authenticity scores."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            labels = list(scores.keys())
            values = list(scores.values())
            colors = ['green' if v >= 0.7 else 'red' if v <= 0.3 else 'orange' for v in values]
            
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_ylabel('Authenticity Score')
            ax.set_title('Authenticity Scores by Analysis Type')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating score chart: {e}")
    
    @staticmethod
    def create_image_overlay(image_path: str, annotations: List[Dict], output_path: str):
        """Create image with authenticity annotations."""
        try:
            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)
                
                # Add annotations (simplified example)
                for annotation in annotations:
                    if 'bbox' in annotation:
                        bbox = annotation['bbox']
                        color = 'red' if annotation.get('suspicious') else 'green'
                        draw.rectangle(bbox, outline=color, width=3)
                
                img.save(output_path)
                
        except Exception as e:
            print(f"Error creating image overlay: {e}")
