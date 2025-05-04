import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif
import time
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="The One With The Stock Analysis",
    page_icon="📊",
    layout="wide"
)

# Friends color palette
FRIENDS_COLORS = {
    'purple': '#5C4F79',
    'yellow': '#F3C969',
    'brown': '#A67F5D',
    'orange': '#F29C50',
    'blue': '#6CB4EE',
    'pink': '#FFC0CB'
}

# CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #FFF9E3;
    }
    .stButton>button {
        background-color: #5C4F79;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #F3C969;
        color: #5C4F79;
    }
    h1, h2, h3 {
        color: #5C4F79;
    }
    .stProgress .st-bo {
        background-color: #F3C969;
    }
    .stSidebar {
        background-color: #F5F1E3;
    }
    .highlight {
        background-color: #F3C969;
        padding: 5px;
        border-radius: 5px;
        font-weight: bold;
    }
    .friends-header {
        font-family: 'Segoe UI Black', 'Arial Black', sans-serif;
        font-size: 4.5em;
        color: #5C4F79;
        font-weight: bold;
        letter-spacing: 2px;
        text-align: center;
    }
    .friends-container {
        border-radius: 25px;
        padding: 20px;
        margin: 10px 0px;
        background-color: #F5F1E3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = None
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Linear Regression"
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# Function to reset the app
def reset_app():
    # Create a list of keys to reset
    keys_to_reset = [
        'data', 'X_train', 'X_test', 'y_train', 'y_test', 
        'model', 'predictions', 'metrics', 'features', 
        'target', 'model_type', 'feature_importance'
    ]
    
    # Reset each key individually
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Set step to 0
    st.session_state.step = 0
    st.session_state.model_type = "Linear Regression"

# App header
def show_header():
    st.markdown('<p class="friends-header">The One With The Stock Analysis</p>', unsafe_allow_html=True)
    # Display Friends GIF
    st.markdown("""
    <div style="display: flex; justify-content: center;">
        <img src="https://media.giphy.com/media/lSbTmUmQwxUmiExV4h/giphy.gif" width="400" alt="Friends dancing in the fountain">
    </div>
    """, unsafe_allow_html=True)
    st.write("## 👋 Welcome to your financial analysis buddy!")
    st.markdown("""
    <div style="display: flex; align-items: center;">
        <div style="flex: 3;">
            <p>As Joey would say: <b>"How YOU doin'?\"</b> with your stock investments?</p>
            <p>This app helps you analyze stock market data with machine learning models, Friends-style!</p>
            <p>Follow the steps below to get started. Could this BE any more helpful?</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
def show_sidebar():
    with st.sidebar:
        st.markdown('<h2 style="color: #5C4F79;">Monica\'s Organized Panel</h2>', unsafe_allow_html=True)
        
        st.markdown("### 📥 The One Where We Choose Data")
        
        data_option = st.radio("Choose data source:", 
                              ["Upload CSV/Excel", "Fetch from Yahoo Finance"])
        
        if data_option == "Upload CSV/Excel":
            uploaded_file = st.file_uploader("Upload your financial dataset", 
                                             type=['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state.data = df
                    st.success(f"Uploaded {uploaded_file.name} successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        else:
            ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL):", "AAPL")
            period = st.selectbox("Select time period:", 
                                 ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])
            interval = st.selectbox("Select interval:", 
                                   ["1d", "5d", "1wk", "1mo", "3mo"])
            
            if st.button("Fetch Stock Data"):
                with st.spinner("Ross is fetching your data... it's PIVOT time!"):
                    try:
                        # Add a small delay to avoid rate limiting
                        time.sleep(2)
                        
                        # Validate ticker symbol
                        if not ticker.strip():
                            st.error("Please enter a valid ticker symbol")
                            return
                            
                        # Create Ticker object first
                        stock = yf.Ticker(ticker)
                        
                        # Get stock info to verify ticker exists
                        info = stock.info
                        if not info:
                            st.error(f"Ticker {ticker} not found. Please check if the symbol is correct.")
                            st.info("Common ticker symbols include: AAPL (Apple), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon)")
                            return
                            
                        # Now fetch historical data
                        df = stock.history(period=period, interval=interval)
                        
                        if df.empty:
                            st.error(f"Data is empty for {ticker}. This might be due to market closure or data unavailability.")
                            st.info("Try selecting a different time period or check if the market is open.")
                            return
                            
                        # Validate the data
                        if len(df) < 2:
                            st.error(f"Not enough data points for {ticker}. Try a longer time period.")
                            return
                            
                        df.reset_index(inplace=True)
                        st.session_state.data = df
                        st.success(f"Successfully downloaded {ticker} stock data!")
                        
                        # Show some basic information about the data
                        st.write(f"Data range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
                        st.write(f"Number of data points: {len(df)}")
                        
                    except Exception as e:
                        if "rate limit" in str(e).lower():
                            st.error("Rate limit exceeded. Please wait a few minutes before trying again.")
                            st.info("You can try a different ticker symbol or wait a few minutes before trying again.")
                        elif "timeout" in str(e).lower():
                            st.error("Connection timeout. Please check your internet connection.")
                            st.info("Try again in a few moments.")
                        else:
                            st.error(f"Error fetching data: {str(e)}")
                            st.info("Please check your internet connection and try again.")
        
        st.markdown("### 🧠 The One With The Model")
        st.session_state.model_type = st.selectbox(
            "Select Machine Learning Model:",
            ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
        )
        
        if st.button("Reset App", key="reset_button"):
            reset_app()
            st.rerun()

# Function to display data
def show_data():
    if st.session_state.data is not None:
        st.markdown('<div class="friends-container">', unsafe_allow_html=True)
        st.markdown("## 📥 The One Where We Upload the Data")
        
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="flex: 3;">
                <p>Chandler says: <b>"Could these numbers BE any more interesting?"</b></p>
            </div>
            <div style="flex: 1; text-align: right;">
                <img src="https://media.giphy.com/media/iqGVczYpIBqMM/giphy.gif" width="150" alt="Chandler">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(st.session_state.data.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total rows:** {st.session_state.data.shape[0]}")
            st.write(f"**Total columns:** {st.session_state.data.shape[1]}")
        
        with col2:
            missing_data = st.session_state.data.isnull().sum().sum()
            st.write(f"**Missing values:** {missing_data}")
            st.write(f"**Data types:** {', '.join(st.session_state.data.dtypes.astype(str).unique())}")
        
        if st.button("Continue to Data Cleaning", key="to_cleaning"):
            st.session_state.step = 1
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Function to clean data
def clean_data():
    if st.session_state.data is not None:
        st.markdown('<div class="friends-container">', unsafe_allow_html=True)
        st.markdown("## 🧼 The One With Data Cleaning")
        
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="flex: 3;">
                <p>Monica would say: <b>"I'm going to clean this data so thoroughly!"</b></p>
            </div>
            <div style="flex: 1; text-align: right;">
                <img src="https://media.giphy.com/media/QscbCkTeoBUkU/giphy.gif" width="150" alt="Monica Cleaning">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.data.copy()
        
        # Display info about missing values
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        
        if len(missing_cols) > 0:
            st.write("### Missing Values")
            
            for col in missing_cols.index:
                st.write(f"- {col}: {missing_cols[col]} missing values")
            
            fill_method = st.selectbox("How to handle missing values?", 
                                      ["Drop rows", "Fill with mean/mode", "Fill with 0"])
            
            if fill_method == "Drop rows":
                df = df.dropna()
                st.success(f"Dropped {len(st.session_state.data) - len(df)} rows with missing values!")
            elif fill_method == "Fill with mean/mode":
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                st.success("Filled missing values with mean/mode!")
            else:
                df = df.fillna(0)
                st.success("Filled missing values with 0!")
        else:
            st.success("No missing values found! Monica would be proud!")
        
        # Handle date columns
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                date_cols.append(col)
        
        if date_cols:
            st.write("### Date Columns")
            st.write("The following columns appear to be dates:")
            for col in date_cols:
                st.write(f"- {col}")
            
            date_handling = st.multiselect(
                "Select extra features to extract from dates:",
                ["Year", "Month", "Day", "Day of Week", "Is Month End", "Is Month Start"]
            )
            
            for col in date_cols:
                if "Year" in date_handling:
                    df[f"{col}_year"] = pd.to_datetime(df[col]).dt.year
                if "Month" in date_handling:
                    df[f"{col}_month"] = pd.to_datetime(df[col]).dt.month
                if "Day" in date_handling:
                    df[f"{col}_day"] = pd.to_datetime(df[col]).dt.day
                if "Day of Week" in date_handling:
                    df[f"{col}_dayofweek"] = pd.to_datetime(df[col]).dt.dayofweek
                if "Is Month End" in date_handling:
                    df[f"{col}_is_month_end"] = pd.to_datetime(df[col]).dt.is_month_end.astype(int)
                if "Is Month Start" in date_handling:
                    df[f"{col}_is_month_start"] = pd.to_datetime(df[col]).dt.is_month_start.astype(int)
        
        # Handle categorical variables
        cat_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' and col not in date_cols:
                cat_cols.append(col)
        
        if cat_cols:
            st.write("### Categorical Columns")
            st.write("The following columns are categorical:")
            for col in cat_cols:
                st.write(f"- {col}: {df[col].nunique()} unique values")
            
            cat_handling = st.selectbox(
                "How to handle categorical variables?",
                ["One-Hot Encoding", "Label Encoding", "Keep as is"]
            )
            
            if cat_handling == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=cat_cols)
                st.success(f"Applied one-hot encoding to {len(cat_cols)} columns!")
            elif cat_handling == "Label Encoding":
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                st.success(f"Applied label encoding to {len(cat_cols)} columns!")
        
        st.session_state.data = df
        
        st.write("### Cleaned Data Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go Back", key="back_to_data"):
                st.session_state.step = 0
                st.rerun()
        
        with col2:
            if st.button("Continue to Feature Engineering", key="to_feature"):
                st.session_state.step = 2
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def clean_numeric_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove any commas
        value = value.replace(',', '')
        # Handle K (thousands)
        if 'K' in value:
            return float(value.replace('K', '')) * 1000
        # Handle M (millions)
        elif 'M' in value:
            return float(value.replace('M', '')) * 1000000
        # Handle B (billions)
        elif 'B' in value:
            return float(value.replace('B', '')) * 1000000000
        # Try to convert to float directly
        try:
            return float(value)
        except ValueError:
            return None
    return None

# Function for feature engineering
def feature_engineering():
    if st.session_state.data is not None:
        st.markdown('<div class="friends-container">', unsafe_allow_html=True)
        st.markdown("## 🧪 The One With Feature Engineering")
        
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="flex: 3;">
                <p>Ross says: <b>"I have a PhD in Paleontology, but today I'm a feature engineer!"</b></p>
            </div>
            <div style="flex: 1; text-align: right;">
                <img src="https://media.giphy.com/media/l41YeqkqbS29kGnCw/giphy.gif" width="150" alt="Ross explaining">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.data.copy()
        
        # Clean numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        for col in df.columns:
            if col not in numeric_cols:
                # Try to convert to numeric
                df[col] = df[col].apply(clean_numeric_value)
        
        # Drop columns that couldn't be converted to numeric
        df = df.dropna(axis=1, how='all')
        
        # Select target variable
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # For clustering, we don't need a target
        if st.session_state.model_type != "K-Means Clustering":
            target_col = st.selectbox("Select target variable (what you want to predict):", 
                                     all_cols)
            st.session_state.target = target_col
            
            potential_features = [col for col in all_cols if col != target_col]
        else:
            potential_features = all_cols
            st.session_state.target = None
        
        # Select features
        selected_features = st.multiselect("Select features for your model:", 
                                          potential_features,
                                          default=potential_features[:min(5, len(potential_features))])
        
        st.session_state.features = selected_features
        
        # Feature scaling
        scaling_option = st.checkbox("Apply Standard Scaling (recommended)", value=True)
        
        # Add technical indicators for stock data
        if 'Close' in df.columns and 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns:
            st.write("### Technical Indicators")
            st.write("Detected stock price data. Would you like to add technical indicators?")
            
            add_ma = st.checkbox("Add Moving Averages")
            add_rsi = st.checkbox("Add Relative Strength Index (RSI)")
            add_bb = st.checkbox("Add Bollinger Bands")
            
            if add_ma:
                ma_periods = st.multiselect("Select MA periods:", [5, 10, 20, 50, 100, 200], default=[20])
                for period in ma_periods:
                    df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
                    st.session_state.features.append(f'MA_{period}')
            
            if add_rsi:
                rsi_period = st.slider("RSI Period:", min_value=5, max_value=30, value=14)
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                st.session_state.features.append('RSI')
            
            if add_bb:
                bb_period = st.slider("Bollinger Bands Period:", min_value=5, max_value=30, value=20)
                df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
                df['BB_Std'] = df['Close'].rolling(window=bb_period).std()
                df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
                df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
                st.session_state.features.extend(['BB_Middle', 'BB_Upper', 'BB_Lower'])
        
        # Remove NaN values created by technical indicators
        df = df.dropna()
        
        # Prepare data
        if st.session_state.model_type != "K-Means Clustering":
            X = df[st.session_state.features]
            y = df[st.session_state.target]
            
            # For Logistic Regression, ensure target is binary
            if st.session_state.model_type == "Logistic Regression":
                if len(y.unique()) > 2:
                    st.warning("Logistic Regression requires a binary target. Converting based on median value.")
                    threshold = y.median()
                    y = (y > threshold).astype(int)
                    st.write(f"Target converted to binary: > {threshold} = 1, <= {threshold} = 0")
        else:
            X = df[st.session_state.features]
            y = None
        
        # Apply scaling if selected
        if scaling_option:
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns)
            except Exception as e:
                st.error(f"Error during scaling: {str(e)}")
                st.warning("Proceeding without scaling.")
        
        # Feature importance for Linear and Logistic Regression
        if st.session_state.model_type in ["Linear Regression", "Logistic Regression"] and len(selected_features) > 1:
            st.write("### Feature Importance Analysis")
            st.write("Let's see which features Ross thinks are most important!")
            
            try:
                with st.spinner("Analyzing feature importance..."):
                    if st.session_state.model_type == "Linear Regression":
                        selector = SelectKBest(score_func=f_regression, k='all')
                    else:  # Logistic Regression
                        selector = SelectKBest(score_func=mutual_info_classif, k='all')
                    
                    selector.fit(X, y)
                    feature_scores = pd.DataFrame({
                        'Feature': X.columns,
                        'Score': selector.scores_
                    }).sort_values('Score', ascending=False)
                    
                    fig = px.bar(
                        feature_scores,
                        x='Score',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance',
                        color='Score',
                        color_continuous_scale=px.colors.sequential.Viridis,
                    )
                    fig.update_layout(height=400, width=700)
                    st.plotly_chart(fig)
                    
                    st.session_state.feature_importance = feature_scores
            except Exception as e:
                st.error(f"Error calculating feature importance: {e}")
        
        # Store the prepared data in session state
        st.session_state.X = X
        st.session_state.y = y
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go Back", key="back_to_cleaning"):
                st.session_state.step = 1
                st.rerun()
        
        with col2:
            if st.button("Continue to Train/Test Split", key="to_split"):
                st.session_state.step = 3
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Function for train test split
def train_test_split_func():
    if hasattr(st.session_state, 'X') and st.session_state.X is not None:
        st.markdown('<div class="friends-container">', unsafe_allow_html=True)
        st.markdown("## 🔪 The One With the Train/Test Split")
        
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="flex: 3;">
                <p>Phoebe says: <b>"I'm cutting the data... but it doesn't hurt, I promise!"</b></p>
            </div>
            <div style="flex: 1; text-align: right;">
                <img src="https://media.giphy.com/media/kzuZeA5WB1hgQ/giphy.gif" width="150" alt="Phoebe being quirky">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Set the split ratio
        test_size = st.slider("Test size (% of data used for testing):", 
                             min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        
        # Set random state for reproducibility
        random_state = st.number_input("Random state (for reproducibility):", 
                                      min_value=0, max_value=1000, value=42)
        
        # Perform the split
        if st.session_state.model_type != "K-Means Clustering":
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.X, 
                st.session_state.y, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Display the split results
            st.write(f"Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.1f}%)")
            st.write(f"Testing set: {X_test.shape[0]} samples ({test_size*100:.1f}%)")
            
            # Show pie chart of split
            fig = go.Figure(
                go.Pie(
                    labels=['Training Set', 'Testing Set'],
                    values=[X_train.shape[0], X_test.shape[0]],
                    hole=0.4,
                    marker=dict(colors=[FRIENDS_COLORS['purple'], FRIENDS_COLORS['yellow']])
                )
            )
            fig.update_layout(
                title="Train/Test Split",
                height=400
            )
            st.plotly_chart(fig)
            
            # Store the split data in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
        else:
            # For K-Means, we don't need a train/test split
            st.write("K-Means Clustering doesn't require a train/test split.")
            st.session_state.X_train = st.session_state.X
            st.session_state.X_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go Back", key="back_to_feature"):
                st.session_state.step = 2
                st.rerun()
        
        with col2:
            if st.button("Continue to Model Training", key="to_training"):
                st.session_state.step = 4
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Function to train model
def train_model():
    if st.session_state.X_train is not None or st.session_state.model_type == "K-Means Clustering":
        st.markdown('<div class="friends-container">', unsafe_allow_html=True)
        st.markdown("## 🧠 The One With the Model Training")
        
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="flex: 3;">
                <p>Rachel says: <b>"This model is training harder than I did for my first marathon!"</b></p>
            </div>
            <div style="flex: 1; text-align: right;">
                <img src="https://media.giphy.com/media/S4NlVXztGfqWc/giphy.gif" width="150" alt="Rachel working hard">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Show progress bar
            progress_bar = st.progress(0)
            
            # Train different models based on selection
            if st.session_state.model_type == "Linear Regression":
                progress_bar.progress(20)
                model = LinearRegression()
                model.fit(st.session_state.X_train, st.session_state.y_train)
                progress_bar.progress(60)
                
                # Make predictions
                y_pred = model.predict(st.session_state.X_test)
                progress_bar.progress(80)
                
                # Calculate metrics
                mse = mean_squared_error(st.session_state.y_test, y_pred)
                r2 = r2_score(st.session_state.y_test, y_pred)
                
                st.session_state.metrics = {
                    'MSE': mse,
                    'R²': r2,
                    'RMSE': np.sqrt(mse)
                }
                
                # Store model coefficients
                coefficients = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', ascending=False)
                
                st.session_state.coefficients = coefficients
                st.session_state.predictions = y_pred
                st.session_state.model = model  # Store the model
                progress_bar.progress(100)
                
            elif st.session_state.model_type == "Logistic Regression":
                progress_bar.progress(20)
                model = LogisticRegression(max_iter=1000)
                model.fit(st.session_state.X_train, st.session_state.y_train)
                progress_bar.progress(60)
                
                # Make predictions
                y_pred = model.predict(st.session_state.X_test)
                progress_bar.progress(80)
                
                # Calculate metrics
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                
                st.session_state.metrics = {
                    'Accuracy': accuracy
                }
                
                # Store model coefficients
                coefficients = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Coefficient': model.coef_[0]
                }).sort_values('Coefficient', ascending=False)
                
                st.session_state.coefficients = coefficients
                st.session_state.predictions = y_pred
                st.session_state.model = model  # Store the model
                progress_bar.progress(100)
                
            else:  # K-Means Clustering
                progress_bar.progress(20)
                # For K-Means, ask for number of clusters
                n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=3)
                
                model = KMeans(n_clusters=n_clusters, random_state=42)
                model.fit(st.session_state.X_train)
                progress_bar.progress(60)
                
                # Make predictions
                clusters = model.predict(st.session_state.X_train)
                progress_bar.progress(80)
                
                # Calculate silhouette score if we have enough samples
                if len(st.session_state.X_train) > n_clusters:
                    silhouette = silhouette_score(st.session_state.X_train, clusters)
                    st.session_state.metrics = {
                        'Silhouette Score': silhouette,
                        'Inertia': model.inertia_
                    }
                else:
                    st.session_state.metrics = {
                        'Inertia': model.inertia_
                    }
                
                # Add cluster labels to the original data
                cluster_df = st.session_state.X_train.copy()
                cluster_df['Cluster'] = clusters
                st.session_state.cluster_df = cluster_df
                st.session_state.model = model  # Store the model
                progress_bar.progress(100)
            
            # Show success message
            st.success("Model training completed successfully!")
            
            # Show metrics
            st.write("### Model Performance Metrics")
            for metric, value in st.session_state.metrics.items():
                st.write(f"- {metric}: {value:.4f}")
            
            # Add navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Go Back", key="back_to_split"):
                    st.session_state.step = 3
                    st.rerun()
            
            with col2:
                if st.button("View Results", key="view_results"):
                    st.session_state.step = 5
                    st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred during model training: {str(e)}")
            st.info("Please check your data and try again. Make sure all features are numeric and there are no missing values.")
            
            # Add a button to go back
            if st.button("Go Back to Data Preparation", key="back_to_prep"):
                st.session_state.step = 2
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_results():
    st.markdown('<div class="friends-container">', unsafe_allow_html=True)
    st.markdown("## 📊 The One With The Results")
    
    st.markdown("""
    <div style="display: flex; align-items: center;">
        <div style="flex: 3;">
            <p>Chandler says: <b>"Could these results BE any more interesting?"</b></p>
        </div>
        <div style="flex: 1; text-align: right;">
            <img src="https://media.giphy.com/media/iqGVczYpIBqMM/giphy.gif" width="150" alt="Chandler">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show model metrics
    if hasattr(st.session_state, 'metrics'):
        st.write("### Model Performance Metrics")
        for metric, value in st.session_state.metrics.items():
            st.write(f"- {metric}: {value:.4f}")
    
    # Create visualizations based on model type
    if st.session_state.model_type == "Linear Regression":
        if hasattr(st.session_state, 'predictions') and hasattr(st.session_state, 'y_test'):
            # Actual vs Predicted plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.y_test,
                y=st.session_state.predictions,
                mode='markers',
                name='Predictions'
            ))
            fig.add_trace(go.Scatter(
                x=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                y=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash')
            ))
            fig.update_layout(
                title='Actual vs Predicted Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                height=500
            )
            st.plotly_chart(fig)
        
        if hasattr(st.session_state, 'coefficients'):
            # Feature importance plot
            fig = px.bar(
                st.session_state.coefficients,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title='Feature Importance (Coefficients)',
                color='Coefficient',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig)
    
    elif st.session_state.model_type == "Logistic Regression":
        if hasattr(st.session_state, 'predictions') and hasattr(st.session_state, 'y_test'):
            # Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Negative', 'Positive'],
                y=['Negative', 'Positive'],
                title='Confusion Matrix'
            )
            st.plotly_chart(fig)
        
        if hasattr(st.session_state, 'coefficients'):
            # Feature importance plot
            fig = px.bar(
                st.session_state.coefficients,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title='Feature Importance (Coefficients)',
                color='Coefficient',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig)
    
    else:  # K-Means Clustering
        if hasattr(st.session_state, 'cluster_df'):
            # Scatter plot of clusters
            if len(st.session_state.features) >= 2:
                fig = px.scatter(
                    st.session_state.cluster_df,
                    x=st.session_state.features[0],
                    y=st.session_state.features[1],
                    color='Cluster',
                    title='Cluster Visualization',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig)
            
            # Cluster sizes
            cluster_sizes = st.session_state.cluster_df['Cluster'].value_counts().sort_index()
            fig = px.bar(
                x=cluster_sizes.index,
                y=cluster_sizes.values,
                title='Cluster Sizes',
                labels={'x': 'Cluster', 'y': 'Number of Points'}
            )
            st.plotly_chart(fig)
    
    # Download results
    st.write("### Download Results")
    
    # Create a DataFrame with results
    if st.session_state.model_type != "K-Means Clustering":
        if hasattr(st.session_state, 'predictions') and hasattr(st.session_state, 'y_test'):
            results_df = pd.DataFrame({
                'Actual': st.session_state.y_test,
                'Predicted': st.session_state.predictions
            })
            # Convert DataFrame to CSV
            csv = results_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="model_results.csv">Download Results as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        if hasattr(st.session_state, 'cluster_df'):
            results_df = st.session_state.cluster_df
            # Convert DataFrame to CSV
            csv = results_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="model_results.csv">Download Results as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Add navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go Back to Model Training", key="back_to_training"):
            st.session_state.step = 4
            st.rerun()
    
    with col2:
        if st.button("Start New Analysis", key="new_analysis"):
            reset_app()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    show_header()
    show_sidebar()
    
    if st.session_state.step == 0:
        show_data()
    elif st.session_state.step == 1:
        clean_data()
    elif st.session_state.step == 2:
        feature_engineering()
    elif st.session_state.step == 3:
        train_test_split_func()
    elif st.session_state.step == 4:
        train_model()
    elif st.session_state.step == 5:
        show_results()