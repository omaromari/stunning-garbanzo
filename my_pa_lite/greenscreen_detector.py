"""
Greenscreen Area Detector - Image-Based Analysis
Analyzes photos to identify optimal greenscreen placement zones.
No 3D reconstruction required!
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install: pip install matplotlib")


class GreenscreenDetector:
    """
    Detects optimal greenscreen areas in film set photos.
    Identifies large, flat, uniform surfaces suitable for VFX replacement.
    """
    
    def __init__(self):
        self.results = []
    
    def analyze_image(self, image_path: str, min_area_ratio: float = 0.1) -> Dict:
        """
        Analyze a single image for greenscreen-suitable areas.
        
        Args:
            image_path: Path to image file
            min_area_ratio: Minimum area as fraction of image (0.1 = 10%)
            
        Returns:
            Dictionary with detected zones and metadata
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Run detection pipeline
        print("→ Detecting flat surfaces...")
        flat_regions = self._detect_flat_regions(img_rgb)
        
        print("→ Analyzing color uniformity...")
        uniform_regions = self._detect_uniform_regions(img_rgb)
        
        print("→ Detecting unobstructed areas...")
        clear_regions = self._detect_clear_areas(img_rgb)
        
        print("→ Computing zone scores...")
        candidate_zones = self._combine_regions(flat_regions, uniform_regions, clear_regions)
        
        print("→ Filtering by size...")
        min_area = int(w * h * min_area_ratio)
        valid_zones = self._filter_by_area(candidate_zones, min_area)
        
        print("→ Ranking zones...")
        ranked_zones = self._rank_zones(valid_zones, w, h)
        
        result = {
            'image_path': image_path,
            'image_size': (w, h),
            'zones': ranked_zones,
            'zone_count': len(ranked_zones)
        }
        
        self.results.append(result)
        
        print(f"\n✓ Found {len(ranked_zones)} suitable greenscreen zones")
        return result
    
    def _detect_flat_regions(self, img: np.ndarray) -> np.ndarray:
        """
        Detect flat/planar regions using edge detection.
        Flat surfaces have low edge density.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Invert: flat regions have few edges
        flat_mask = cv2.bitwise_not(edges_dilated)
        
        # Morphological closing to fill small gaps
        flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        return flat_mask
    
    def _detect_uniform_regions(self, img: np.ndarray) -> np.ndarray:
        """
        Detect regions with uniform color/texture.
        Uses variance in local windows.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calculate local standard deviation
        # Low std = uniform region
        kernel_size = 15
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        mean_sq = cv2.blur(gray**2, (kernel_size, kernel_size))
        std = np.sqrt(np.abs(mean_sq - mean**2))
        
        # Threshold: uniform regions have low std
        uniform_mask = (std < 20).astype(np.uint8) * 255
        
        # Clean up noise
        kernel = np.ones((7, 7), np.uint8)
        uniform_mask = cv2.morphologyEx(uniform_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return uniform_mask
    
    def _detect_clear_areas(self, img: np.ndarray) -> np.ndarray:
        """
        Detect areas free of obstacles/furniture.
        Uses superpixel segmentation to find large continuous regions.
        """
        # Convert to HSV for better segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Detect walls/large surfaces (usually have similar hue in regions)
        h, w = img.shape[:2]
        
        # Simple approach: segment by color similarity
        # Reshape to 2D array of pixels
        pixels = hsv.reshape((-1, 3)).astype(np.float32)
        
        # K-means clustering to find dominant regions
        k = 5  # Number of clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # Reshape back to image
        labels = labels.reshape((h, w))
        
        # Find largest clusters (likely walls/large surfaces)
        clear_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(k):
            cluster_mask = (labels == i).astype(np.uint8)
            area = np.sum(cluster_mask)
            
            # Keep large clusters (>5% of image)
            if area > (h * w * 0.05):
                clear_mask = cv2.bitwise_or(clear_mask, cluster_mask * 255)
        
        # Clean up
        kernel = np.ones((15, 15), np.uint8)
        clear_mask = cv2.morphologyEx(clear_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return clear_mask
    
    def _combine_regions(self, flat_mask: np.ndarray, uniform_mask: np.ndarray, 
                        clear_mask: np.ndarray) -> List[Dict]:
        """
        Combine all detection masks to find candidate zones.
        """
        # Combine masks (all conditions must be met)
        combined = cv2.bitwise_and(flat_mask, uniform_mask)
        combined = cv2.bitwise_and(combined, clear_mask)
        
        # Find contours (potential greenscreen zones)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zones = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum size threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                
                # Calculate rectangularity (how well it fits bounding box)
                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0
                
                zones.append({
                    'id': i,
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'rectangularity': rectangularity
                })
        
        return zones
    
    def _filter_by_area(self, zones: List[Dict], min_area: int) -> List[Dict]:
        """Filter zones by minimum area."""
        return [z for z in zones if z['area'] >= min_area]
    
    def _rank_zones(self, zones: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """
        Rank zones by suitability for greenscreen.
        
        Scoring criteria:
        - Size (larger is better)
        - Rectangularity (more rectangular is better)
        - Aspect ratio (closer to 16:9 or 4:3 is better)
        - Position (centered is slightly preferred)
        """
        img_area = img_w * img_h
        
        for zone in zones:
            score = 0.0
            
            # Size score (0-40 points)
            size_ratio = zone['area'] / img_area
            score += min(size_ratio * 100, 40)
            
            # Rectangularity score (0-25 points)
            score += zone['rectangularity'] * 25
            
            # Aspect ratio score (0-20 points)
            # Prefer 16:9 (1.78), 4:3 (1.33), or vertical walls
            ar = zone['aspect_ratio']
            if 1.5 <= ar <= 2.0:  # Close to 16:9
                score += 20
            elif 1.2 <= ar <= 1.5:  # Close to 4:3
                score += 18
            elif ar > 2.5 or ar < 0.4:  # Very wide or very tall (walls)
                score += 15
            else:
                score += 10
            
            # Position score (0-15 points)
            # Prefer regions not at extreme edges
            x, y, w, h = zone['bbox']
            center_x = x + w/2
            center_y = y + h/2
            dist_from_center = np.sqrt((center_x - img_w/2)**2 + (center_y - img_h/2)**2)
            max_dist = np.sqrt((img_w/2)**2 + (img_h/2)**2)
            position_score = 15 * (1 - dist_from_center / max_dist)
            score += position_score
            
            zone['score'] = score
            zone['rank'] = 0  # Will be set after sorting
        
        # Sort by score (descending)
        zones.sort(key=lambda z: z['score'], reverse=True)
        
        # Assign ranks
        for i, zone in enumerate(zones):
            zone['rank'] = i + 1
        
        return zones
    
    def visualize_results(self, result: Dict, output_path: str = None, 
                         show_top_n: int = 5) -> None:
        """
        Create visualization showing detected greenscreen zones.
        
        Args:
            result: Result dictionary from analyze_image
            output_path: Where to save the visualization
            show_top_n: Number of top zones to highlight
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Error: matplotlib required for visualization")
            return
        
        # Load original image
        img = cv2.imread(result['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.imshow(img_rgb)
        
        # Draw top zones
        zones_to_show = result['zones'][:show_top_n]
        colors = ['red', 'orange', 'yellow', 'green', 'cyan']
        
        for i, zone in enumerate(zones_to_show):
            x, y, w, h = zone['bbox']
            color = colors[i % len(colors)]
            
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                    edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            label = f"#{zone['rank']} (score: {zone['score']:.1f})"
            ax.text(x, y - 10, label, color=color, fontsize=12, 
                   fontweight='bold', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
        
        ax.axis('off')
        plt.title(f"Detected Greenscreen Zones: {Path(result['image_path']).name}", 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {output_path}")
        
        plt.show()
    
    def analyze_multiple_images(self, image_paths: List[str], 
                               output_dir: str = "greenscreen_results") -> Dict:
        """
        Analyze multiple images and generate comprehensive report.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory for output files
            
        Returns:
            Summary dictionary with all results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("GREENSCREEN AREA DETECTION")
        print("="*60)
        print(f"Analyzing {len(image_paths)} images...")
        
        # Analyze each image
        for img_path in image_paths:
            result = self.analyze_image(img_path)
            
            # Generate visualization
            img_name = Path(img_path).stem
            viz_path = output_dir / f"{img_name}_greenscreen_zones.png"
            self.visualize_results(result, output_path=str(viz_path))
        
        # Generate summary report
        summary = self._generate_summary()
        
        # Save JSON report
        report_path = output_dir / "greenscreen_detection_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Processed {len(self.results)} images")
        print(f"✓ Results saved to: {output_dir}")
        print(f"✓ Report: {report_path}")
        
        return summary
    
    def _generate_summary(self) -> Dict:
        """Generate summary report from all analyzed images."""
        total_zones = sum(r['zone_count'] for r in self.results)
        
        # Find best zone overall
        best_zone = None
        best_score = 0
        best_image = None
        
        for result in self.results:
            if result['zones']:
                top_zone = result['zones'][0]
                if top_zone['score'] > best_score:
                    best_score = top_zone['score']
                    best_zone = top_zone
                    best_image = result['image_path']
        
        summary = {
            'total_images_analyzed': len(self.results),
            'total_zones_detected': total_zones,
            'best_zone': {
                'image': best_image,
                'score': best_score,
                'bbox': best_zone['bbox'] if best_zone else None,
                'area_sqpx': best_zone['area'] if best_zone else None
            },
            'per_image_results': [
                {
                    'image': r['image_path'],
                    'zones_found': r['zone_count'],
                    'top_zone_score': r['zones'][0]['score'] if r['zones'] else 0
                }
                for r in self.results
            ]
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    detector = GreenscreenDetector()
    
    # Analyze your film set images
    image_paths = [
        "view_1.jpg",
        "view_2.jpg",
        "view_3.jpg"
    ]
    
    # Run detection
    summary = detector.analyze_multiple_images(
        image_paths=image_paths,
        output_dir="greenscreen_results"
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best greenscreen location found in: {summary['best_zone']['image']}")
    print(f"Score: {summary['best_zone']['score']:.1f}/100")
    print(f"Total zones across all images: {summary['total_zones_detected']}")
