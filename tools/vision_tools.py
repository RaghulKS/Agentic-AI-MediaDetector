import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class VisionAuthenticityDetector:
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models()
        
    def load_models(self):
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            
            self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.vit_model.to(self.device)
            
            print(f"Models loaded on {self.device}")
            
        except Exception as e:
            print(f"Model loading error: {str(e)}")
            raise
    
    def analyze_image_authenticity(self, image_path: str) -> Dict:
        try:
            image = Image.open(image_path).convert('RGB')
            
            results = {
                'authenticity_scores': {},
                'visual_anomalies': {},
                'model_predictions': {},
                'confidence_scores': {}
            }
            
            clip_results = self._clip_authenticity_check(image)
            results['authenticity_scores']['clip'] = clip_results['authenticity_score']
            results['model_predictions']['clip'] = clip_results['prediction']
            results['confidence_scores']['clip'] = clip_results['confidence']
            
            vit_results = self._vit_analysis(image)
            results['authenticity_scores']['vit'] = vit_results['authenticity_score']
            results['model_predictions']['vit'] = vit_results['prediction']
            results['confidence_scores']['vit'] = vit_results['confidence']
            
            anomalies = self._detect_visual_anomalies(image_path)
            results['visual_anomalies'] = anomalies
            
            overall_score = self._calculate_overall_score(results)
            results['overall_authenticity_score'] = overall_score
            
            return results
            
        except Exception as e:
            return {
                'error': f"Vision analysis failed: {str(e)}",
                'overall_authenticity_score': 0.5,
                'authenticity_scores': {},
                'visual_anomalies': {},
                'model_predictions': {}
            }
    
    def _clip_authenticity_check(self, image: Image.Image) -> Dict:
        try:
            text_prompts = [
                "a real photograph taken with a camera",
                "an AI-generated artificial image", 
                "a digitally manipulated photograph",
                "an authentic unedited photograph"
            ]
            
            inputs = self.clip_processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            real_photo_prob = probs[0]
            authentic_photo_prob = probs[3] 
            ai_generated_prob = probs[1]
            manipulated_prob = probs[2]
            
            authenticity_score = (real_photo_prob + authentic_photo_prob) / 2
            
            prediction = "authentic" if authenticity_score > 0.5 else "ai_generated_or_manipulated"
            confidence = max(authenticity_score, 1 - authenticity_score)
            
            return {
                'authenticity_score': float(authenticity_score),
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    'real_photo': float(real_photo_prob),
                    'ai_generated': float(ai_generated_prob),
                    'manipulated': float(manipulated_prob),
                    'authentic_unedited': float(authentic_photo_prob)
                }
            }
            
        except Exception as e:
            return {
                'authenticity_score': 0.5,
                'prediction': 'uncertain', 
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _vit_analysis(self, image: Image.Image) -> Dict:
        """Use ViT for additional image analysis."""
        try:
            # Process image
            inputs = self.vit_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence = torch.max(predictions).item()
            
            # For ViT, we use the confidence as a proxy for authenticity
            # Higher confidence in natural object classification suggests authenticity
            authenticity_score = min(confidence, 0.9)  # Cap at 0.9
            
            prediction = "authentic" if authenticity_score > 0.5 else "questionable"
            
            return {
                'authenticity_score': float(authenticity_score),
                'prediction': prediction,
                'confidence': float(confidence)
            }
            
        except Exception as e:
            return {
                'authenticity_score': 0.5,
                'prediction': 'uncertain',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _detect_visual_anomalies(self, image_path: str) -> Dict:
        """Detect visual anomalies that might indicate AI generation."""
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            anomalies = {}
            
            # Check for unusual noise patterns
            noise_score = self._analyze_noise_patterns(gray)
            anomalies['noise_anomaly'] = {
                'score': noise_score,
                'suspicious': noise_score > 0.7
            }
            
            # Check for edge artifacts
            edge_score = self._analyze_edge_artifacts(gray)
            anomalies['edge_artifacts'] = {
                'score': edge_score,
                'suspicious': edge_score > 0.6
            }
            
            # Check for compression artifacts
            compression_score = self._analyze_compression_artifacts(img)
            anomalies['compression_artifacts'] = {
                'score': compression_score,
                'suspicious': compression_score < 0.3  # Too little compression can be suspicious
            }
            
            return anomalies
            
        except Exception as e:
            return {
                'error': f"Anomaly detection failed: {str(e)}",
                'noise_anomaly': {'score': 0.5, 'suspicious': False},
                'edge_artifacts': {'score': 0.5, 'suspicious': False},
                'compression_artifacts': {'score': 0.5, 'suspicious': False}
            }
    
    def _analyze_noise_patterns(self, gray_image: np.ndarray) -> float:
        """Analyze noise patterns in the image."""
        try:
            # Calculate noise using Laplacian variance
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            
            # Normalize to 0-1 scale (higher values indicate more noise)
            normalized_noise = min(laplacian_var / 1000, 1.0)
            return normalized_noise
            
        except:
            return 0.5
    
    def _analyze_edge_artifacts(self, gray_image: np.ndarray) -> float:
        """Analyze edge artifacts that might indicate AI generation."""
        try:
            # Detect edges
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Unusual edge patterns might indicate AI generation
            return edge_density
            
        except:
            return 0.5
    
    def _analyze_compression_artifacts(self, image: np.ndarray) -> float:
        """Analyze compression artifacts."""
        try:
            # Convert to float
            img_float = image.astype(np.float32) / 255.0
            
            # Calculate variance in 8x8 blocks (JPEG compression analysis)
            h, w = img_float.shape[:2]
            block_variances = []
            
            for i in range(0, h - 8, 8):
                for j in range(0, w - 8, 8):
                    block = img_float[i:i+8, j:j+8]
                    if block.shape[0] == 8 and block.shape[1] == 8:
                        block_variances.append(np.var(block))
            
            if block_variances:
                # Higher variance indicates more compression artifacts
                avg_variance = np.mean(block_variances)
                return min(avg_variance * 10, 1.0)  # Scale appropriately
            
            return 0.5
            
        except:
            return 0.5
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall authenticity score from all analyses."""
        scores = []
        weights = []
        
        # CLIP score (high weight)
        if 'clip' in results['authenticity_scores']:
            scores.append(results['authenticity_scores']['clip'])
            weights.append(0.4)
        
        # ViT score (medium weight)
        if 'vit' in results['authenticity_scores']:
            scores.append(results['authenticity_scores']['vit'])
            weights.append(0.3)
        
        # Visual anomalies (lower weight but important)
        if 'noise_anomaly' in results['visual_anomalies']:
            noise_score = 1 - results['visual_anomalies']['noise_anomaly']['score']
            scores.append(noise_score)
            weights.append(0.2)
        
        if 'edge_artifacts' in results['visual_anomalies']:
            edge_score = 1 - results['visual_anomalies']['edge_artifacts']['score']
            scores.append(edge_score)
            weights.append(0.1)
        
        if not scores:
            return 0.5
        
        # Weighted average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5


class ImagePreprocessor:
    """Image preprocessing utilities."""
    
    @staticmethod
    def prepare_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Prepare image for model input."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            raise ValueError(f"Could not prepare image: {str(e)}")
    
    @staticmethod
    def extract_image_features(image_path: str) -> Dict:
        """Extract basic image features."""
        try:
            image = Image.open(image_path)
            
            features = {
                'dimensions': image.size,
                'mode': image.mode,
                'format': image.format,
                'has_transparency': image.mode in ('RGBA', 'LA', 'P'),
            }
            
            # Calculate basic statistics
            if image.mode == 'RGB':
                rgb_array = np.array(image)
                features['mean_rgb'] = [float(rgb_array[:,:,i].mean()) for i in range(3)]
                features['std_rgb'] = [float(rgb_array[:,:,i].std()) for i in range(3)]
            
            return features
            
        except Exception as e:
            return {'error': str(e)}
