#!/usr/bin/env python3
"""
CLIP Object Detection and Classification Script
Analyzes historical photos using CLIP (Contrastive Language-Image Pre-training) models
for semantic understanding and object classification
"""

import torch
import clip
import numpy as np
import json
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import cv2

class CLIPDetector:
    def __init__(self, model_name='ViT-B/32'):
        """
        Initialize CLIP detector
        model_name: 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model_name = model_name
        
        # Historical context categories for 20th century European photos
        self.historical_categories = [
            # People and social context - highest priority
            "a person in formal clothing",
            "a person in military uniform", 
            "a woman in period dress",
            "a man in suit and hat",
            "children playing",
            "family portrait",
            "wedding photograph",
            "group of people",
            
            # Transportation - crucial for dating
            "horse and carriage",
            "steam locomotive",
            "early automobile",
            "vintage car",
            "bicycle",
            "horse-drawn cart",
            "railway station",
            "train platform",
            
            # Architecture and buildings
            "European building",
            "church or cathedral",
            "town square",
            "residential house",
            "brick building",
            "stone building",
            "wooden building",
            "castle or manor",
            
            # Daily life objects
            "period furniture",
            "antique chair",
            "wooden table",
            "vintage clock",
            "old books",
            "traditional tableware",
            "ceramic vase",
            "oil lamp",
            "candles",
            
            # Agricultural and rural life
            "farm animals",
            "horses working",
            "cattle or cows",
            "sheep in field",
            "agricultural tools",
            "farming equipment",
            "rural landscape",
            "barn or stable",
            
            # Military and historical events
            "military equipment",
            "soldiers marching",
            "ceremonial event",
            "flags or banners",
            "uniforms and medals",
            "wartime scene",
            
            # Cultural and religious
            "religious ceremony",
            "traditional clothing",
            "cultural celebration",
            "musical instruments",
            "religious artifacts",
            
            # Technology and tools
            "early industrial equipment",
            "mechanical tools",
            "craftsmanship tools",
            "printing equipment",
            "scientific instruments",
            
            # Nature and environment
            "European countryside",
            "forest or woods",
            "river or stream",
            "mountain landscape",
            "garden or park",
            "domestic animals",
            "wild birds"
        ]
        
        # Priority categories (same logic as YOLO)
        self.priority_categories = [
            "a person in formal clothing",
            "a person in military uniform",
            "family portrait",
            "horse and carriage",
            "steam locomotive", 
            "early automobile",
            "European building",
            "period furniture",
            "farm animals",
            "military equipment",
            "traditional clothing",
            "antique chair",
            "vintage clock",
            "old books"
        ]
        
        # Historical context mapping
        self.historical_context = {
            "a person in formal clothing": "Formal wear indicates social status, special occasions, dating clues",
            "a person in military uniform": "Military service, wartime context, rank and regiment identification",
            "family portrait": "Family relationships, social customs, formal photography practices",
            "horse and carriage": "Primary transportation, social status, pre-automotive era",
            "steam locomotive": "Railway development, industrial revolution, long-distance travel",
            "early automobile": "Technological advancement, dating photos to specific decades",
            "European building": "Architecture styles help date and locate photographs",
            "period furniture": "Interior design, social status, craftsmanship of the era",
            "farm animals": "Agricultural life, rural economy, working animals",
            "military equipment": "Military technology, warfare methods, historical conflicts",
            "traditional clothing": "Regional customs, cultural identity, fashion evolution",
            "antique chair": "Furniture styles, domestic life, social gatherings",
            "vintage clock": "Timekeeping technology, household items, daily life",
            "old books": "Literacy, education, intellectual life, religious practices"
        }
        
        # Precompute text features for categories
        self.text_features = self._encode_text_categories()
    
    def _encode_text_categories(self):
        """Precompute text embeddings for all categories"""
        text_inputs = clip.tokenize(self.historical_categories).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def analyze_image_semantics(self, image_path, confidence_threshold=0.3):
        """Analyze image using CLIP semantic understanding"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode image
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                similarities = similarities.cpu().numpy()[0]
            
            # Extract relevant detections above threshold
            detections = []
            for i, (category, similarity) in enumerate(zip(self.historical_categories, similarities)):
                if similarity >= confidence_threshold:
                    detections.append({
                        'category': category,
                        'confidence': float(similarity),
                        'category_id': i,
                        'historical_context': self.historical_context.get(category, 'Historical relevance')
                    })
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections
            
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return []
    
    def analyze_photo(self, image_path, confidence_threshold=0.3):
        """Analyze a single photo for historical content"""
        try:
            detections = self.analyze_image_semantics(image_path, confidence_threshold)
            
            # Count categories
            category_counts = {}
            for detection in detections:
                category = detection['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Calculate priority score
            priority_score = 0
            for detection in detections:
                category = detection['category']
                if category in self.priority_categories:
                    priority_index = self.priority_categories.index(category)
                    priority_score += detection['confidence'] * (len(self.priority_categories) - priority_index)
            
            return {
                'filename': Path(image_path).name,
                'total_detections': len(detections),
                'category_counts': category_counts,
                'priority_score': float(priority_score),
                'detections': detections,
                'top_categories': [d['category'] for d in detections[:5]]
            }
            
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None
    
    def create_semantic_visualization(self, image_path, detections, output_path=None, show_image=False):
        """Create visualization showing semantic analysis results"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Create figure with image and results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Show original image
            ax1.imshow(image)
            ax1.set_title(f"Original Image - {Path(image_path).name}", fontsize=12, weight='bold')
            ax1.axis('off')
            
            # Show top detections as bar chart
            if detections:
                top_detections = detections[:10]  # Top 10
                categories = [d['category'] for d in top_detections]
                confidences = [d['confidence'] for d in top_detections]
                
                # Create horizontal bar chart
                y_pos = np.arange(len(categories))
                bars = ax2.barh(y_pos, confidences, alpha=0.8)
                
                # Color bars based on confidence
                for i, (bar, conf) in enumerate(zip(bars, confidences)):
                    if conf > 0.7:
                        bar.set_color('darkgreen')
                    elif conf > 0.5:
                        bar.set_color('orange')
                    else:
                        bar.set_color('lightcoral')
                
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels([cat[:30] + '...' if len(cat) > 30 else cat for cat in categories])
                ax2.set_xlabel('Confidence Score')
                ax2.set_title('CLIP Semantic Analysis Results', fontsize=12, weight='bold')
                ax2.set_xlim(0, 1.0)
                
                # Add confidence values as text
                for i, conf in enumerate(confidences):
                    ax2.text(conf + 0.01, i, f'{conf:.3f}', va='center')
            else:
                ax2.text(0.5, 0.5, 'No detections above threshold', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title('CLIP Semantic Analysis Results', fontsize=12, weight='bold')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            if show_image:
                plt.show()
            else:
                plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error creating visualization for {image_path}: {e}")
            return False
    
    def analyze_directory(self, input_dir='../sample_photos', output_file='clip_analysis.json', 
                         confidence_threshold=0.3, create_visualizations=True):
        """Analyze all photos in a directory"""
        input_dir = Path(input_dir)
        results = []
        
        # Create visualizations directory
        if create_visualizations:
            viz_dir = Path('semantic_visualizations')
            viz_dir.mkdir(exist_ok=True)
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        # Find all image files
        for ext in extensions:
            image_paths.extend(list(input_dir.glob(f'*{ext}')))
            image_paths.extend(list(input_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_paths)} images to analyze with CLIP")
        print(f"Using {self.model_name} model")
        
        # Analyze each image
        for i, image_path in enumerate(image_paths):
            print(f"Analyzing {i+1}/{len(image_paths)}: {image_path.name}")
            
            result = self.analyze_photo(image_path, confidence_threshold)
            if result:
                results.append(result)
                
                # Create visualization if requested
                if create_visualizations and result['total_detections'] > 0:
                    viz_path = viz_dir / f"{Path(image_path).stem}_semantic.png"
                    self.create_semantic_visualization(image_path, result['detections'], viz_path)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Save results to JSON
        output_data = {
            'model_info': {
                'model_name': self.model_name,
                'confidence_threshold': confidence_threshold,
                'total_photos_analyzed': len(results),
                'device': self.device
            },
            'summary': summary,
            'photos': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nCLIP analysis complete! Results saved to {output_file}")
        if create_visualizations:
            print(f"Visualizations saved to semantic_visualizations/ directory")
        
        return results
    
    def _calculate_summary(self, results):
        """Calculate summary statistics"""
        if not results:
            return {}
        
        # Count all categories
        all_categories = {}
        photos_with_detections = 0
        total_detections = 0
        
        for result in results:
            if result['total_detections'] > 0:
                photos_with_detections += 1
                total_detections += result['total_detections']
                
                for category, count in result['category_counts'].items():
                    all_categories[category] = all_categories.get(category, 0) + count
        
        # Sort categories by frequency
        sorted_categories = dict(sorted(all_categories.items(), key=lambda x: x[1], reverse=True))
        
        # Find most common categories
        top_categories = dict(list(sorted_categories.items())[:10])
        
        return {
            'photos_with_detections': photos_with_detections,
            'photos_without_detections': len(results) - photos_with_detections,
            'total_detections': total_detections,
            'average_detections_per_photo': round(total_detections / len(results), 2),
            'unique_categories': len(all_categories),
            'most_common_categories': top_categories,
            'all_category_counts': sorted_categories
        }
    
    def analyze_single_photo(self, image_path, confidence_threshold=0.3, show_image=True, save_result=True):
        """Analyze a single photo and optionally display it"""
        print(f"Analyzing single photo: {image_path}")
        
        # Check if file exists
        if not Path(image_path).exists():
            print(f"Error: File not found - {image_path}")
            return None
        
        # Analyze the photo
        result = self.analyze_photo(image_path, confidence_threshold)
        if not result:
            print("Failed to analyze photo")
            return None
        
        # Print results
        print(f"\n{'='*50}")
        print(f"CLIP SEMANTIC ANALYSIS: {result['filename']}")
        print(f"{'='*50}")
        print(f"Total detections: {result['total_detections']}")
        print(f"Priority score: {result['priority_score']:.3f}")
        
        if result['detections']:
            print(f"\nTop semantic categories found:")
            for i, detection in enumerate(result['detections'][:8], 1):
                context = detection['historical_context']
                print(f"  {i}. {detection['category']} ({detection['confidence']:.3f})")
                print(f"     → {context}")
        else:
            print("No categories detected above threshold")
        
        # Show image if requested
        if show_image:
            print(f"\nDisplaying semantic analysis visualization...")
            self.create_semantic_visualization(image_path, result['detections'], show_image=True)
        
        # Save individual result if requested
        if save_result:
            output_file = f"single_photo_clip_{Path(image_path).stem}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'model_info': {
                        'model_name': self.model_name,
                        'confidence_threshold': confidence_threshold,
                        'device': self.device
                    },
                    'photo': result
                }, f, indent=2)
            print(f"Result saved to: {output_file}")
        
        return result
    
    def analyze_random_photos(self, input_dir='../sample_photos', num_photos=5, 
                            confidence_threshold=0.3, show_images=True):
        """Analyze random photos from directory"""
        input_dir = Path(input_dir)
        
        # Find all image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(list(input_dir.glob(f'*{ext}')))
            image_paths.extend(list(input_dir.glob(f'*{ext.upper()}')))
        
        if len(image_paths) == 0:
            print(f"No images found in {input_dir}")
            return []
        
        # Select random photos
        num_photos = min(num_photos, len(image_paths))
        selected_photos = random.sample(image_paths, num_photos)
        
        print(f"Analyzing {num_photos} random photos from {len(image_paths)} total images")
        
        results = []
        for i, image_path in enumerate(selected_photos, 1):
            print(f"\n--- Random Photo {i}/{num_photos} ---")
            result = self.analyze_single_photo(image_path, confidence_threshold, 
                                             show_image=show_images, save_result=False)
            if result:
                results.append(result)
        
        # Save combined results
        output_file = f"random_{num_photos}_photos_clip.json"
        with open(output_file, 'w') as f:
            json.dump({
                'model_info': {
                    'model_name': self.model_name,
                    'confidence_threshold': confidence_threshold,
                    'num_photos': len(results),
                    'device': self.device
                },
                'photos': results
            }, f, indent=2)
        
        print(f"\nRandom CLIP analysis complete! Results saved to: {output_file}")
        return results
    
    def compare_with_text_queries(self, image_path, custom_queries=None):
        """Compare image against custom text queries"""
        if custom_queries is None:
            custom_queries = [
                "a historical photograph from early 1900s",
                "a family portrait from the 20th century", 
                "people in vintage clothing",
                "military or war-related scene",
                "rural or countryside setting",
                "urban or city environment",
                "formal or ceremonial occasion",
                "everyday life scene",
                "transportation or vehicles",
                "religious or cultural event"
            ]
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode text queries
            text_inputs = clip.tokenize(custom_queries).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_inputs)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                similarities = similarities.cpu().numpy()[0]
            
            # Create results
            results = []
            for query, similarity in zip(custom_queries, similarities):
                results.append({
                    'query': query,
                    'similarity': float(similarity)
                })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results
            
        except Exception as e:
            print(f"Error comparing with text queries: {e}")
            return []
    
    def interactive_mode(self):
        """Interactive mode for choosing analysis type"""
        print("\n" + "="*60)
        print("CLIP SEMANTIC ANALYSIS - INTERACTIVE MODE")
        print("="*60)
        print("Choose analysis mode:")
        print("1. Analyze all photos in directory")
        print("2. Analyze 5 random photos")
        print("3. Analyze specific photo (enter path)")
        print("4. Compare photo with custom text queries")
        print("5. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == '1':
                    print("\nAnalyzing all photos with CLIP...")
                    self.analyze_directory()
                    break
                
                elif choice == '2':
                    print("\nAnalyzing 5 random photos with CLIP...")
                    self.analyze_random_photos(num_photos=5, show_images=True)
                    break
                
                elif choice == '3':
                    photo_path = input("Enter photo path: ").strip()
                    if photo_path:
                        self.analyze_single_photo(photo_path, show_image=True)
                    else:
                        print("Invalid path")
                    break
                
                elif choice == '4':
                    photo_path = input("Enter photo path: ").strip()
                    if photo_path and Path(photo_path).exists():
                        queries = input("Enter custom queries (comma-separated, or press Enter for defaults): ").strip()
                        custom_queries = [q.strip() for q in queries.split(',')] if queries else None
                        results = self.compare_with_text_queries(photo_path, custom_queries)
                        
                        print(f"\nText Query Comparison Results:")
                        for i, result in enumerate(results[:10], 1):
                            print(f"  {i}. {result['query']}: {result['similarity']:.3f}")
                    else:
                        print("Invalid path")
                    break
                
                elif choice == '5':
                    print("Exiting...")
                    break
                
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                break

def main():
    # Default settings - using smaller model for faster inference
    model_name = 'ViT-B/32'  # Balanced speed and accuracy
    
    print("Initializing CLIP detector...")
    detector = CLIPDetector(model_name=model_name)
    
    print(f"Historical categories ({len(detector.historical_categories)}):")
    for category in detector.historical_categories[:10]:
        print(f"  {category}")
    print("  ... and more")
    
    # Run interactive mode
    detector.interactive_mode()

if __name__ == "__main__":
    main()