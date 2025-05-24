# License Plate Recognition Project

## Overview

This project implements a License Plate Recognition system using Python. It leverages machine learning techniques to detect and recognize license plates from images. The system uses a pre-trained model (`best.pt`) for detection and includes a Flask-based web application for user interaction.

## Project Structure

- `static/`: Contains static files like CSS, JavaScript, and images used by the web app.
- `templates/`: Stores HTML templates for the Flask web application.
- `app.py`: The main Flask application script that handles routing and logic for the web interface.
- `best.pt`: Pre-trained model file used for license plate detection (e.g., YOLO model).
- `plate_recognition.csv`: Sample dataset or output file containing license plate recognition results.
- `requirements.txt`: Lists all Python dependencies required for the project.

## Prerequisites

- Python 3.8 or higher
- Git (to clone the repository)
- A compatible environment for running the pre-trained model (e.g., GPU for faster inference, if available)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/license-plate-recognition.git
   cd license-plate-recognition
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the pre-trained model** (if not already included): Ensure the `best.pt` file is in the project root directory. If it's missing, you may need to download it from the model provider or train your own.

## Usage

1. **Run the Flask application**:

   ```bash
   python app.py
   ```

   The app will start on `http://localhost:5000` by default.

2. **Access the web interface**:

   - Open your browser and navigate to `http://localhost:5000`.
   - Upload an image containing a license plate to see the recognition results.

3. **View results**:

   - Recognized license plates will be displayed on the web page.
   - Results are also logged in `plate_recognition.csv` for further analysis.

## Model Details

- The `best.pt` model is likely a YOLO-based model fine-tuned for license plate detection.
- It detects the license plate region in an image, and OCR techniques (e.g., using Tesseract) may be applied to extract the text.

## Dependencies

See `requirements.txt` for the full list. Key dependencies include:

- Flask (for the web app)
- OpenCV (for image processing)
- PyTorch (for model inference)
- Transformers (for TrOCR-based text extraction)

## Contributing

Feel free to submit issues or pull requests on GitHub. For major changes, please open an issue first to discuss your ideas.

## License

This project is licensed under the MIT License. See the LICENSE file for details (if applicable).

## Acknowledgments

- Thanks to the open-source community for providing tools like YOLO, Flask, and TrOCR.
- Inspired by various license plate recognition projects online.
