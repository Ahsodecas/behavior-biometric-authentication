# Python Authentication App

This project is a desktop application designed for user authentication using machine learning. It continuously collects user data and authenticates users based on the collected data.

## Project Structure

```
python-auth-app
├── datasets                  # Required datasets
├── extracted_features        # Extracted Features
├── collected data            # Raw collcted data
├── src
│   ├── main.py               # Entry point of the application
│   ├── gui                   # Main application window class
│   ├── auth                  # User authentication logic
│   ├── ml                    # Machine Learning logic
│   └── utils                 # Utils module initializer
├── logs                      # Folder containing application logs
├── metrics                   # Folder containing settings and metrics
├── models
│   └── model.pkl             # Serialized machine learning model
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd python-auth-app
   ```
2. Setup Virtual Environment (Optional):
   ```
   python -m venv venv
   # Activate the environment:
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python src/main.py
   ```
5. Additional Notes}
    - Make sure your Python version is compatible.
    - If you encounter permission issues, try **pip install --user -r requirements.txt**.
    - item The installation has not been thoroughly tested on macOS/Linux, but it does work on these platforms.


## Usage Guidelines

- The application will prompt users for their credentials.
- User data will be collected continuously for authentication purposes.
- The machine learning model will validate the credentials based on the collected data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.