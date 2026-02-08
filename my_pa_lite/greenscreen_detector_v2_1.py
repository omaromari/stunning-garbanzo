"""
Greenscreen Area Detector v2.1 - Balanced Detection with Debug Mode
Intelligently identifies greenscreen-suitable areas while excluding furniture.
Includes debug visualization to troubleshoot detection issues.
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
    print("Warning: matplotlib not available")


class SmartGreenscreenDetector:
    """
    Enhanced detector with balanced thresholds and debug capabilities.
    """
    
    def __init__(self):
        self.results = []
    
    def analyze_image(self, image_path: str, 
                     min_area_ratio: float = 0.05,
                     max_area_ratio: float = 0.6,
                     debug_output_dir: str = None) -> Dict:
        """
        Analyze image for greenscreen-suitable areas.
        
        Args:
            image_path: Path to image
            min_area_ratio: Minimum zone size (5% of image)
            max_area_ratio: Maximum zone size (60% of image)
            debug_output_dir: If provided, save debug visualizations
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # STEP 1: Detect and exclude furniture/objects
        print("→ Detecting furniture and obstacles...")
        obstacle_mask = self._detect_obstacles(img_rgb)
        print(f"  Obstacle coverage: {np.sum(obstacle_mask > 0) / (h*w) * 100:.1f}%")
        
        # STEP 2: Find wall-like surfaces
        print("→ Identifying wall surfaces...")
        wall_mask = self._detect_walls(img_rgb)
        print(f"  Wall-like area: {np.sum(wall_mask > 0) / (h*w) * 100:.1f}%")
        
        # STEP 3: Find highly uniform regions
        print("→ Analyzing color uniformity...")
        uniform_mask = self._detect_uniform_regions_balanced(img_rgb)
        print(f"  Uniform area: {np.sum(uniform_mask > 0) / (h*w) * 100:.1f}%")
        
        # Save debug images if requested
        if debug_output_dir:
            debug_dir = Path(debug_output_dir)
            debug_dir.mkdir(exist_ok=True)
            img_name = Path(image_path).stem
            
            cv2.imwrite(str(debug_dir / f"{img_name}_1_obstacles.png"), obstacle_mask)
            cv2.imwrite(str(debug_dir / f"{img_name}_2_walls.png"), wall_mask)
            cv2.imwrite(str(debug_dir / f"{img_name}_3_uniform.png"), uniform_mask)
            print(f"  Debug images saved to: {debug_dir}")
        
        # STEP 4: Combine criteria
        print("→ Finding valid greenscreen zones...")
        # Must be: wall-like AND uniform AND NOT obstructed
        valid_mask = cv2.bitwise_and(wall_mask, uniform_mask)
        combined_before_obstacles = np.sum(valid_mask > 0) / (h*w) * 100
        valid_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(obstacle_mask))
        print(f"  Combined area (before obstacle filter): {combined_before_obstacles:.1f}%")
        print(f"  Combined area (after obstacle filter): {np.sum(valid_mask > 0) / (h*w) * 100:.1f}%")
        
        if debug_output_dir:
            cv2.imwrite(str(debug_dir / f"{img_name}_4_combined.png"), valid_mask)
        
        # STEP 5: Find candidate zones
        zones = self._extract_zones(valid_mask, img_rgb)
        print(f"  Found {len(zones)} initial zones")
        
        # STEP 6: Filter by area
        min_area = int(w * h * min_area_ratio)
        max_area = int(w * h * max_area_ratio)
        print(f"→ Filtering zones (min: {min_area}px, max: {max_area}px)...")
        valid_zones = [z for z in zones if min_area <= z['area'] <= max_area]
        print(f"  {len(valid_zones)} zones after size filter")
        
        # STEP 7: Quality check each zone
        print("→ Quality checking zones...")
        checked_zones = self._quality_check_zones_relaxed(valid_zones, img_rgb, obstacle_mask)
        print(f"  {len(checked_zones)} zones after quality check")
        
        # STEP 8: Rank zones
        print("→ Ranking zones...")
        ranked_zones = self._rank_zones_v2(checked_zones, w, h)
        
        result = {
            'image_path': image_path,
            'image_size': (w, h),
            'zones': ranked_zones,
            'zone_count': len(ranked_zones)
        }
        
        self.results.append(result)
        
        print(f"\n✓ Found {len(ranked_zones)} suitable greenscreen zones")
        return result
    
    def _detect_obstacles(self, img: np.ndarray) -> np.ndarray:
        """
        Detect furniture, objects, and other obstacles.
        Uses edge density and texture analysis.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # High edge density = furniture/objects
        edges = cv2.Canny(gray, 30, 100)
        
        # Count edges in local windows
        kernel_size = 31
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
        
        # High edge density = obstacles
        obstacle_mask = (edge_density > 15).astype(np.uint8) * 255
        
        # Also detect high-contrast objects
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        contrast = cv2.absdiff(gray, blurred)
        high_contrast = (contrast > 30).astype(np.uint8) * 255
        
        # Combine
        obstacle_mask = cv2.bitwise_or(obstacle_mask, high_contrast)
        
        # Dilate to create safety margins around obstacles
        kernel = np.ones((21, 21), np.uint8)
        obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=2)
        
        return obstacle_mask
    
    def _detect_walls(self, img: np.ndarray) -> np.ndarray:
        """
        Detect wall-like surfaces using orientation and gradient analysis.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude (low = flat surface)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Walls have low gradients
        wall_mask = (grad_mag < 20).astype(np.uint8) * 255
        
        # Additionally, look for large connected regions
        kernel = np.ones((15, 15), np.uint8)
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, connectivity=8)
        
        filtered_mask = np.zeros_like(wall_mask)
        min_wall_area = (h * w) * 0.03  # Walls should be at least 3% of image
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_wall_area:
                filtered_mask[labels == i] = 255
        
        return filtered_mask
    
    def _detect_uniform_regions_balanced(self, img: np.ndarray) -> np.ndarray:
        """
        Balanced uniformity detection - more forgiving than strict version.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Calculate local standard deviation
        kernel_size = 25
        
        uniform_masks = []
        for channel in range(3):
            ch = lab[:, :, channel].astype(np.float32)
            mean = cv2.blur(ch, (kernel_size, kernel_size))
            mean_sq = cv2.blur(ch**2, (kernel_size, kernel_size))
            std = np.sqrt(np.abs(mean_sq - mean**2))
            
            # More relaxed threshold (was 10, now 18)
            uniform_masks.append((std < 18).astype(np.uint8) * 255)
        
        # All channels must be uniform
        uniform_mask = cv2.bitwise_and(uniform_masks[0], uniform_masks[1])
        uniform_mask = cv2.bitwise_and(uniform_mask, uniform_masks[2])
        
        # Clean up
        kernel = np.ones((11, 11), np.uint8)
        uniform_mask = cv2.morphologyEx(uniform_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        uniform_mask = cv2.morphologyEx(uniform_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return uniform_mask
    
    def _extract_zones(self, mask: np.ndarray, img: np.ndarray) -> List[Dict]:
        """Extract potential greenscreen zones from mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zones = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 5000:  # Skip tiny regions
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate properties
            aspect_ratio = w / h if h > 0 else 0
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Get perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Compactness
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            zones.append({
                'id': i,
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'rectangularity': rectangularity,
                'compactness': compactness
            })
        
        return zones
    
    def _quality_check_zones_relaxed(self, zones: List[Dict], img: np.ndarray, 
                            obstacle_mask: np.ndarray) -> List[Dict]:
        """
        Relaxed quality checks - more forgiving than strict version.
        """
        h, w = img.shape[:2]
        checked_zones = []
        
        for zone in zones:
            x, y, zw, zh = zone['bbox']
            
            # Bounds checking
            if x < 0 or y < 0 or x+zw > w or y+zh > h:
                continue
            
            # Extract zone region
            zone_img = img[y:y+zh, x:x+zw]
            zone_obstacles = obstacle_mask[y:y+zh, x:x+zw]
            
            # Check 1: Obstacle coverage (relaxed to 25% from 15%)
            obstacle_ratio = np.sum(zone_obstacles > 0) / (zw * zh)
            if obstacle_ratio > 0.25:
                continue
            
            # Check 2: Color variance (relaxed to 35 from 25)
            color_std = np.std(zone_img, axis=(0, 1)).mean()
            if color_std > 35:
                continue
            
            # Check 3: Shape reasonableness (relaxed range)
            ar = zone['aspect_ratio']
            if ar < 0.15 or ar > 10.0:  # Was 0.2-8.0
                continue
            
            # Check 4: Not tiny edge artifacts (relaxed)
            center_x = x + zw/2
            center_y = y + zh/2
            
            margin = 0.03  # Was 0.05
            if (x < w * margin or (x + zw) > w * (1 - margin) or
                y < h * margin or (y + zh) > h * (1 - margin)):
                # Only reject if also small
                if zone['area'] < (w * h * 0.08):  # Was 0.15
                    continue
            
            # Passed all checks
            zone['obstacle_ratio'] = obstacle_ratio
            zone['color_std'] = color_std
            checked_zones.append(zone)
        
        return checked_zones
    
    def _rank_zones_v2(self, zones: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """
        Enhanced ranking system.
        """
        img_area = img_w * img_h
        
        for zone in zones:
            score = 0.0
            
            # Size score (0-35 points)
            size_ratio = zone['area'] / img_area
            if 0.15 <= size_ratio <= 0.50:
                score += 35
            elif 0.10 <= size_ratio <= 0.60:
                score += 30
            elif size_ratio < 0.10:
                score += size_ratio * 200
            else:
                score += 20
            
            # Uniformity score (0-30 points)
            uniformity_score = max(0, 30 - zone['color_std'])
            score += uniformity_score
            
            # Obstacle-free score (0-20 points)
            obstacle_score = (1 - zone['obstacle_ratio']) * 20
            score += obstacle_score
            
            # Shape score (0-15 points)
            rect_score = zone['rectangularity'] * 10
            ar = zone['aspect_ratio']
            
            if 1.3 <= ar <= 2.0:
                rect_score += 5
            elif 0.3 <= ar <= 0.7:
                rect_score += 4
            
            score += rect_score
            
            zone['score'] = score
        
        # Sort by score
        zones.sort(key=lambda z: z['score'], reverse=True)
        
        # Assign ranks
        for i, zone in enumerate(zones):
            zone['rank'] = i + 1
        
        return zones
    
    def visualize_results(self, result: Dict, output_path: str = None, 
                         show_top_n: int = 5, show_debug: bool = False) -> None:
        """
        Visualize detected zones.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Error: matplotlib required")
            return
        
        img = cv2.imread(result['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.imshow(img_rgb)
        
        # Draw zones
        zones_to_show = result['zones'][:show_top_n]
        colors = ['#FF0000', '#FF8800', '#FFFF00', '#00FF00', '#00FFFF']
        
        for i, zone in enumerate(zones_to_show):
            x, y, w, h = zone['bbox']
            color = colors[i % len(colors)]
            
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=4, 
                                    edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Create label
            label_lines = [
                f"Rank #{zone['rank']}",
                f"Score: {zone['score']:.1f}",
                f"Size: {zone['area']/(img_rgb.shape[0]*img_rgb.shape[1])*100:.1f}%"
            ]
            
            if show_debug:
                label_lines.extend([
                    f"Uniformity: {zone['color_std']:.1f}",
                    f"Obstacles: {zone['obstacle_ratio']*100:.1f}%"
                ])
            
            label = '\n'.join(label_lines)
            
            label_y = y - 15 if y > 80 else y + h + 15
            
            ax.text(x, label_y, label, color=color, fontsize=11, 
                   fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='white', alpha=0.9, edgecolor=color, linewidth=2),
                   verticalalignment='bottom' if y > 80 else 'top')
        
        ax.axis('off')
        title = f"Greenscreen Zones: {Path(result['image_path']).name}"
        if result['zone_count'] == 0:
            title += " (No suitable zones found)"
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        plt.show()
    
    def analyze_multiple_images(self, image_paths: List[str], 
                               output_dir: str = "greenscreen_results_v2",
                               enable_debug: bool = False) -> Dict:
        """
        Analyze multiple images.
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory
            enable_debug: If True, save debug masks for each detection step
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        debug_dir = None
        if enable_debug:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("GREENSCREEN DETECTOR V2.1 - BALANCED DETECTION")
        print("="*60)
        print(f"Analyzing {len(image_paths)} images...\n")
        
        for img_path in image_paths:
            result = self.analyze_image(img_path, debug_output_dir=debug_dir)
            
            img_name = Path(img_path).stem
            viz_path = output_dir / f"{img_name}_zones_v2.png"
            self.visualize_results(result, output_path=str(viz_path))
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save report
        report_path = output_dir / "detection_report_v2.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Processed: {len(self.results)} images")
        print(f"✓ Results: {output_dir}")
        print(f"✓ Report: {report_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in summary['per_image_results']:
            print(f"\n{Path(r['image']).name}:")
            print(f"  Zones found: {r['zones_found']}")
            if r['zones_found'] > 0:
                print(f"  Best score: {r['top_zone_score']:.1f}/100")
        
        if summary['best_zone']['score'] > 0:
            print(f"\nBest zone overall:")
            print(f"  Image: {Path(summary['best_zone']['image']).name}")
            print(f"  Score: {summary['best_zone']['score']:.1f}/100")
        else:
            print(f"\nNo suitable zones found in any image.")
        
        return summary
    
    def _generate_summary(self) -> Dict:
        """Generate summary report."""
        total_zones = sum(r['zone_count'] for r in self.results)
        
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
                'area_sqpx': best_zone['area'] if best_zone else None,
                'area_percent': (best_zone['area'] / (self.results[0]['image_size'][0] * 
                                self.results[0]['image_size'][1]) * 100) if best_zone else 0
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
    detector = SmartGreenscreenDetector()
    
    image_paths = [
        "view_1.jpg",
        "view_2.jpg",
        "view_3.jpg"
    ]
    
    # Run with debug mode enabled to see what's being detected
    summary = detector.analyze_multiple_images(
        image_paths=image_paths,
        output_dir="greenscreen_results_v2",
        enable_debug=True  # This will save debug masks
    )
