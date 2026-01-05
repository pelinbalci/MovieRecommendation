# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.0] - 2025-01-05

### Added
Animated progress bar during model training with status messages

Redesigned Explore page with interactive visualizations:
  - Key metrics dashboard (users, movies, ratings, avg rating)
  - Top rated movies horizontal bar chart
  - Genre distribution chart with color gradient
  - Most popular movie analysis with rating trend over time
  - Interactive scatter plot with filters (minimum ratings, rating range)
  - Rating distribution histogram


Improved Recommendation page with step-by-step flow:
* Clear 3-step structure (Customize → Rate → Get Results)
* Better labels and helper text
* Info box for new users
* Cleaner UI with dividers

### Changed

* Recommendation page text from informal to professional tone
* Slider labels simplified (removed ALL CAPS)
* Tutorial page completely rewritten with better structure
* References page redesigned with sections and links


## [2.0.1] - 2025-01-04

### Fixed
- **Terminology correction**: Changed "Content Based Filtering" to "Collaborative Filtering" throughout the codebase
  - `utils.py`: Updated docstrings in both `cofi_cost_func()` and `cofi_cost_func_v()` functions
  - `train_predict_page_v2.py`: Fixed subtitle from "with TensorFlow and Content Based Filtering" to "with TensorFlow and Collaborative Filtering"

### Added
- **README.md enhancements**:
  - Notation Reference table mapping mathematical notation to Python variables
  - Default Hyperparameters table (λ=1.0, learning rate=0.1, iterations=100, features=100)
  - Normalization section explaining how mean normalization addresses the cold-start problem
  - "Adding New User Ratings" section with code examples showing matrix integration
  - Note clarifying that bias term `b` is intentionally not regularized
  - Updated Table of Contents with new sections

### Documentation
- Improved algorithm explanations with clearer mathematical representations
- Added code snippets demonstrating normalization and prediction formulas

---

## [2.0.0] - 2023-12-07

### Added
- Complete README.md documentation with:
  - Algorithm explanation (Matrix Factorization, Cost Function, Training Process)
  - Project structure overview
  - Comprehensive functions reference
  - Installation and usage instructions
- MIT License
- Version badges

### Changed
- Restructured project for better modularity
- Improved code organization across multiple pages

### Features
- Interactive movie rating interface
- Genre-based filtering for recommendations
- Customizable hyperparameters (iterations, features, optimizer)
- Real-time model training with TensorFlow
- Data exploration and visualization page

---

## [1.0.0] -  2023-08-22 - Initial Release

### Added
- Core recommendation engine using Collaborative Filtering
- Streamlit web interface with multiple pages:
  - Recommendation page (`train_predict_page_v2.py`)
  - Hyperparameter tuning page (`tuning_page.py`)
  - Data exploration page (`explore_page.py`)
  - Tutorial page (`tutorial.py`)
  - References page (`references_page.py`)
- MovieLens dataset integration
- TensorFlow-based training with gradient descent optimization
- Support for multiple optimizers (Adam, SGD, RMSprop)

---

## Version History Summary

| Version | Date       | Highlights |
|---------|------------|------------|
| **v2.1.0** | 2025-01-05 |Redesigned Explore page, animated training progress, improved UX |
| 2.0.1 | 2025-01-04 | Terminology fixes, enhanced documentation |
| 2.0.0 | -          | Major documentation update, README overhaul |
| 1.0.0 | -          | Initial release with core functionality |

---

## Upcoming / Planned

- [ ] Fix notebook error: length mismatch between `my_predictions` and `df_ratings_mean` after duplicate removal
- [ ] Add model persistence (save/load trained models)
- [ ] Implement additional recommendation algorithms for comparison
- [ ] Add unit tests for core functions
- [ ] Performance optimization for larger datasets