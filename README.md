# LexoRead: AI-Supported Reading Assistant for Dyslexia

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/openimpactai/lexoread.svg)](https://github.com/openimpactai/lexoread/issues)
[![GitHub Stars](https://img.shields.io/github/stars/openimpactai/lexoread.svg)](https://github.com/openimpactai/lexoread/stargazers)

LexoRead is an open-source AI-powered reading assistant designed specifically for individuals with dyslexia and reading impairments. The project aims to make reading more accessible through advanced text processing, OCR, and speech technologies.

## Features

- **Intelligent Text Adaptation**: Adjusts text display based on individual reading preferences and dyslexia patterns
- **Text-to-Speech**: High-quality audio rendering of text with adjustable speed and voice options
- **OCR Capabilities**: Extract text from images, documents, and physical books
- **Reading Comprehension Tools**: Highlighting, summarization, and vocabulary assistance
- **Progress Tracking**: Monitor reading improvement over time
- **Cross-Platform Support**: Web, mobile, and desktop applications

## Project Structure

```
lexoread/
│
├── api/               # Backend API for the application
├── datasets/          # Training and evaluation datasets
├── docs/              # Documentation and guides
├── frontend/          # User interface components
├── models/            # ML models for text processing and speech
├── notebooks/         # Research and development notebooks
├── scripts/           # Utility scripts
└── tests/             # Testing suite
```

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+ (for frontend)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/openimpactai/lexoread.git
   cd lexoread
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the datasets:
   ```bash
   cd datasets
   pip install -r requirements.txt
   python download_scripts/download_all.py
   ```

5. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

1. Start the backend server:
   ```bash
   python api/app.py
   ```

2. In a separate terminal, start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Dataset Preparation

LexoRead relies on several datasets for training and evaluation. The `datasets` directory contains scripts to download and prepare these datasets:

```bash
cd datasets
python download_scripts/download_all.py --sample  # For a small sample
```

See the [datasets README](datasets/README.md) for more information.

## Model Training

To train the models, use the scripts in the `models` directory:

```bash
cd models
python train_tts_model.py
python train_ocr_model.py
```

## Contributing

We welcome contributions to LexoRead! Please check out our [Contributing Guide](CONTRIBUTING.md) for guidelines on how to contribute.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Roadmap

- [x] Project setup and basic architecture
- [x] Dataset preparation scripts
- [ ] Basic text-to-speech model implementation
- [ ] OCR model for text extraction from images
- [ ] Text processing for dyslexia-friendly visualization
- [ ] Web application frontend
- [ ] Mobile application development
- [ ] User testing and feedback integration
- [ ] Customizable reading profiles

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenDyslexic Font](https://opendyslexic.org/)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- [Project Gutenberg](https://www.gutenberg.org/)
- [TextOCR Dataset](https://textvqa.org/textocr/)

## Contact

For questions or support, please [open an issue](https://github.com/openimpactai/lexoread/issues) or contact the maintainers.

---

<p align="center">Made with ❤️ by OpenImpactAI</p>
