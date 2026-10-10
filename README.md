# 🔥 HEATGUARD AI

**ANN-Based Heat Wave Prediction System for May (Mumbai)**

This is a Soft Computing academic project. An artificial neural network, written **from scratch in NumPy**, predicts whether heat-wave conditions are likely for a May day in Mumbai from six climate variables — and the application *shows you the mathematics while it happens*.

> No TensorFlow. No PyTorch. No Keras. Every weight, gradient, and activation displayed in the UI is computed by `ann.py` at run time.

## 🚀 How to Run

1. **Navigate into the project directory:**
   ```bash
   cd heatguard
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # On Windows use: .venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

The application will open in your browser at **http://localhost:8501**.

On the first launch, it will create a demo CSV dataset automatically if none is present.

## 📂 Project Structure

- `heatguard/app.py`: Streamlit User Interface.
- `heatguard/ann.py`: The Artificial Neural Network implementation (forward pass, backpropagation, gradient descent).
- `heatguard/data_utils.py`: Data loading, demo-data generation, and preprocessing.
- `heatguard/heatwave.py`: Heat-wave domain rules.
- `heatguard/config.py`: Tunable settings.
- `heatguard/requirements.txt`: Python dependencies (streamlit, numpy, pandas, plotly).

For more detailed information on the mathematics, equations, and data processing, please refer to the [detailed README inside the `heatguard` folder](heatguard/README.md).
