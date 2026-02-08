# my_pa_lite

A lightweight Production Assistant tool for location-assessment crews to identify potential areas suitable for greenscreen VFX integration.

## Overview

**my_pa_lite** analyzes location photographs to detect areas that would be suitable for greenscreen/chroma key VFX work. By uploading three images of a location captured in a triangulation pattern, the tool processes the imagery to identify optimal greenscreen placement zones.

This tool is designed for:
- Location scouts assessing VFX-friendly filming locations
- Production assistants preparing technical location reports
- VFX supervisors evaluating greenscreen setup possibilities

## Requirements

- **Python**: Version 3.8 to 3.12
- **Dependencies**: Listed in [requirements.txt](requirements.txt)
  - opencv-python
  - numpy
  - matplotlib

## Installation

1. Clone or download this repository

2. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your location images**:
   - Capture 3 photographs of your area of interest
   - Use a triangulation pattern (different angles/perspectives of the same area)
   - Save the images in the project directory

2. **Run the greenscreen detector**:
   ```bash
   python greenscreen_detector_v2_1.py
   ```

3. **Review results**:
   - Analysis results will be saved in the `greenscreen_results_v2/` subfolder
   - Review the processed images and detection data to identify suitable greenscreen areas

## Project Structure

```
my_pa_lite/
├── greenscreen_detector_v2_1.py    # Main analysis script (latest version)
├── greenscreen_detector_v2.py      # Previous version
├── greenscreen_detector.py         # Original version
├── requirements.txt                 # Python dependencies
├── greenscreen_results_v2/         # Output folder for analysis results
├── view_1.jpg                      # Example: First triangulation image
├── view_2.jpg                      # Example: Second triangulation image
└── view_3.jpg                      # Example: Third triangulation image
```

## Example Workflow

1. Visit your filming location
2. Take 3 photos from different angles covering your area of interest
3. Place the images in the project directory
4. Run `greenscreen_detector_v2_1.py`
5. Check the `greenscreen_results_v2/` folder for processed results
6. Use the analysis to plan greenscreen placement and VFX shots

## Notes

- Ensure adequate lighting and image quality for best results
- The triangulation pattern helps the tool analyze the space from multiple perspectives
- Results are advisory and should be reviewed by VFX professionals for final decisions

---

*my_pa_lite - Streamlining location assessment for VFX production*
