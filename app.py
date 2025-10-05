"""
Unified Exoplanet Detection System with ML
Complete system with training, prediction, and SHAP explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap
import pickle
import os
from io import BytesIO
import PyPDF2

# Page config
st.set_page_config(
    page_title="SpaceHunt - ExoPlanet",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - White background + colorful UI
st.markdown("""
<style>
    
    
    .main, .stApp {
        background: #ffffff;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #FF6B6B, #FFD93D, #6BCB77);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(255,107,107,0.4);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(255,107,107,0.4)); }
        to { filter: drop-shadow(0 0 15px rgba(107,203,119,0.6)); }
    }
    
    .subtitle {
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        color: #555555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FF6B6B20, #FFD93D20, #6BCB7720);
        border: 2px solid rgba(107,203,119,0.5);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(107,203,119,0.3);
    }
    
    .result-exoplanet {
        background: linear-gradient(135deg, #6BCB77AA, #3BB54AAA);
        border-left: 5px solid #10b981;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .result-not-exoplanet {
        background: linear-gradient(135deg, #FF6B6BAA, #FF4E4EAA);
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #FFD93D20, #FF6B6B20, #6BCB7720);
        border: 1px solid rgba(107,203,119,0.5);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B, #FFD93D, #6BCB77);
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(107,203,119,0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,107,107,0.6);
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #333333;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #FF6B6B20, #FFD93D20, #6BCB7720);
        border-radius: 10px;
        padding: 1rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# Helper Functions
def find_column(df, possible_names, exclude_keywords=['err', 'error']):
    """
    Flexible column finder that matches multiple possible names
    """
    for name in possible_names:
        for col in df.columns:
            col_lower = col.lower()
            if name.lower() in col_lower:
                # Check exclusions
                if not any(exc in col_lower for exc in exclude_keywords):
                    return col
    return None

@st.cache_data
def load_datasets():
    """Load all three datasets"""
    datasets = {}
    data_files = [
        'data/cumulative_2025.10.04_07.26.30.csv',
        'data/k2pandc_2025.10.04_07.51.06.csv',
        'data/TOI_2025.10.04_07.50.21.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, comment='#', low_memory=False)
                df.columns = df.columns.str.strip()
                dataset_name = os.path.basename(file_path).split('_')[0]
                datasets[dataset_name] = df
            except Exception as e:
                st.warning(f"Could not load {file_path}: {str(e)}")
    
    return datasets

def prepare_training_data(datasets):
    """Prepare combined training data from all datasets with improved column mapping"""
    all_data = []
    
    # Column mapping for each dataset type
    column_mappings = {
        'period': ['period', 'pl_orbper', 'orbital_period'],
        'duration': ['duration', 'pl_trandur', 'transit_duration', 'tran_dur'],
        'depth': ['depth', 'pl_trandep', 'transit_depth', 'tran_depth'],
        'radius': ['radius', 'pl_rade', 'prad', 'pl_radj', 'planetary_radius', 'planet_radius']
    }
    
    for name, df in datasets.items():
        st.write(f"Processing {name} dataset...")
        
        # Find disposition column
        disp_col = find_column(df, ['disposition', 'disp', 'status', 'tfopwg_disp'])
        
        if disp_col is None:
            st.warning(f"⚠️ {name}: No disposition column found")
            continue
        
        # Find feature columns using flexible matching
        found_cols = {}
        for feat, possible_names in column_mappings.items():
            col = find_column(df, possible_names)
            if col:
                found_cols[feat] = col
        
        # Check if we have all required columns
        missing = set(['period', 'duration', 'depth', 'radius']) - set(found_cols.keys())
        
        if missing:
            st.warning(f"⚠️ {name}: Missing columns: {missing}")
            st.write(f"Available columns: {list(df.columns[:10])}...")
            continue
        
        # Extract data
        try:
            temp_df = df[[found_cols['period'], found_cols['duration'], 
                         found_cols['depth'], found_cols['radius'], disp_col]].copy()
            temp_df.columns = ['period', 'duration', 'depth', 'radius', 'disposition']
            
            # Remove invalid values
            temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
            temp_df = temp_df.dropna()
            
            # Create binary label with more flexible matching
            temp_df['label'] = temp_df['disposition'].apply(
                lambda x: 1 if any(word in str(x).upper() for word in 
                                  ['CONFIRMED', 'CANDIDATE', 'PC', 'CP']) else 0
            )
            
            all_data.append(temp_df[['period', 'duration', 'depth', 'radius', 'label']])
            st.success(f"✅ {name}: Added {len(temp_df)} samples")
            
        except Exception as e:
            st.error(f"❌ {name}: Error processing - {str(e)}")
            continue
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return None

def train_model(df):
    """Train Random Forest model"""
    X = df[['period', 'duration', 'depth', 'radius']]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'test_size': len(y_test),
        'train_size': len(y_train)
    }
    
    return model, scaler, metrics, X_test_scaled, y_test

def explain_prediction_shap(model, scaler, input_data, feature_names):
    """Generate SHAP explanation for prediction - FIXED VERSION"""
    try:
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Get prediction first
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        
        # Handle SHAP values correctly for binary classification
        # For RandomForest binary classification, shap_values can be:
        # - A list of 2 arrays (one per class)
        # - Or a single array for class 1
        if isinstance(shap_values, list):
            # Use class 1 (exoplanet) SHAP values
            shap_vals_class1 = shap_values[1][0]
        else:
            # Single array case
            shap_vals_class1 = shap_values[0]
        
        return shap_vals_class1, prediction, proba, explainer.expected_value
        
    except Exception as e:
        st.error(f"SHAP Error: {str(e)}")
        # Return dummy values if SHAP fails
        return np.zeros(len(feature_names)), prediction, proba, 0

def read_pdf(file):
    """Extract text from PDF (basic implementation)"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return None

def parse_uploaded_file(uploaded_file):
    """Parse CSV, Excel, or PDF file"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file, comment='#', low_memory=False)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'pdf':
            st.warning("PDF parsing is basic - CSV or Excel recommended for best results")
            return None
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def find_required_columns(df):
    """Find required columns in uploaded file with flexible matching"""
    column_mappings = {
        'period': ['period', 'pl_orbper', 'orbital_period'],
        'duration': ['duration', 'pl_trandur', 'transit_duration', 'tran_dur'],
        'depth': ['depth', 'pl_trandep', 'transit_depth', 'tran_depth'],
        'radius': ['radius', 'pl_rade', 'prad', 'pl_radj', 'planetary_radius', 'planet_radius']
    }
    
    found_cols = {}
    for feat, possible_names in column_mappings.items():
        col = find_column(df, possible_names)
        if col:
            found_cols[feat] = col
    
    return found_cols

# Main UI
st.markdown('<div class="main-title">🌌 EXOPLANET AI DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Machine Learning Detection System with Explainable AI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/200px-NASA_logo.svg.png", width=150)
    st.markdown("### 🚀 Control Panel")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🤖 Train Model", "🔮 Predict", "📊 Batch Analysis", "📈 Model Info"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")

    datasets = load_datasets()
    for name, df in datasets.items():
        st.metric(name.upper(), f"{len(df):,} entries")
    
    st.markdown("---")
    if st.session_state.model is not None:
        st.success("✅ Model Trained")
        if st.session_state.model_metrics:
            st.metric("Accuracy", f"{st.session_state.model_metrics['accuracy']:.1%}")
    else:
        st.warning("⚠️ Model Not Trained")

# Page: Home
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Mission</h3>
            <p>Detect exoplanets using advanced machine learning trained on NASA's Kepler, K2, and TESS datasets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🧠 AI Model</h3>
            <p>Random Forest Classifier with SHAP explainability for transparent decision-making</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>📊 Datasets</h3>
            <p>Trained on 3 comprehensive NASA exoplanet datasets with thousands of observations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🌟 Key Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Manual Input**: Enter 4 orbital parameters for instant prediction
        - **Batch Processing**: Upload CSV/Excel files for bulk analysis
        - **Real-time Training**: Train custom models on NASA datasets
        """)
    
    with col2:
        st.markdown("""
        - **SHAP Explanations**: Understand why AI made each classification
        - **Accuracy Metrics**: Precision, Recall, F1-Score, Confusion Matrix
        - **Beautiful Visualizations**: Interactive charts and graphs
        """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to **Train Model** first, then start making predictions!")

# Page: Train Model
elif page == "🤖 Train Model":
    st.markdown("## 🤖 Train Detection Model")
    st.write("Train a Random Forest classifier on combined NASA datasets")
    
    if st.button("🚀 Start Training", use_container_width=True, type="primary"):
        with st.spinner("Loading datasets..."):
            datasets = load_datasets()
            
            if not datasets:
                st.error("No datasets found! Please ensure data files are in the 'data' folder")
                st.stop()
            
            training_data = prepare_training_data(datasets)
            
            if training_data is None or len(training_data) == 0:
                st.error("Could not prepare training data")
                st.stop()
        
        st.success(f"✅ Loaded {len(training_data):,} training samples")
        
        with st.spinner("Training Random Forest model... This may take a minute"):
            model, scaler, metrics, X_test, y_test = train_model(training_data)
            
            # Save to session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_names = ['period', 'duration', 'depth', 'radius']
            st.session_state.model_metrics = metrics
        
        st.success("🎉 Model trained successfully!")
        
        # Display metrics
        st.markdown("### 📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #667eea; margin: 0;">{metrics['accuracy']:.1%}</h2>
                <p style="margin: 0.5rem 0 0 0;">Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #10b981; margin: 0;">{metrics['precision']:.1%}</h2>
                <p style="margin: 0.5rem 0 0 0;">Precision</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #f59e0b; margin: 0;">{metrics['recall']:.1%}</h2>
                <p style="margin: 0.5rem 0 0 0;">Recall</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #8b5cf6; margin: 0;">{metrics['f1']:.1%}</h2>
                <p style="margin: 0.5rem 0 0 0;">F1-Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Confusion Matrix
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📈 Confusion Matrix")
            cm = metrics['confusion_matrix']
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Not Exoplanet', 'Exoplanet'],
                y=['Not Exoplanet', 'Exoplanet'],
                colorscale='Purples',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
            ))
            
            fig.update_layout(
                title="Prediction vs Actual",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Training Details")
            st.write(f"**Training Samples**: {metrics['train_size']:,}")
            st.write(f"**Test Samples**: {metrics['test_size']:,}")
            st.write(f"**Model Type**: Random Forest")
            st.write(f"**Features**: 4 (Period, Duration, Depth, Radius)")
            st.write(f"**Trees**: 200")
            
            st.markdown("### 💡 Model Insights")
            st.write(f"✅ True Positives: {cm[1,1]}")
            st.write(f"✅ True Negatives: {cm[0,0]}")
            st.write(f"❌ False Positives: {cm[0,1]}")
            st.write(f"❌ False Negatives: {cm[1,0]}")

# Page: Predict
elif page == "🔮 Predict":
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first!")
        st.info("Go to **Train Model** page to train the AI")
        st.stop()
    
    st.markdown("## 🔮 Make Predictions")
    
    tab1, tab2 = st.tabs(["📝 Manual Input", "📁 File Upload"])
    
    with tab1:
        st.markdown("### Enter Orbital Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            period = st.number_input(
                "🌍 Orbital Period (days)",
                min_value=0.0,
                value=10.5,
                step=0.1,
                help="Time for one complete orbit around the star"
            )
            
            duration = st.number_input(
                "⏱️ Transit Duration (hours)",
                min_value=0.0,
                value=3.2,
                step=0.1,
                help="Duration of the transit event"
            )
        
        with col2:
            depth = st.number_input(
                "📉 Transit Depth (ppm)",
                min_value=0.0,
                value=120.0,
                step=1.0,
                help="Depth of the light curve dip"
            )
            
            radius = st.number_input(
                "🪐 Planetary Radius (Earth radii)",
                min_value=0.0,
                value=2.0,
                step=0.1,
                help="Size relative to Earth"
            )
        
        if st.button("🔍 Predict with AI", use_container_width=True, type="primary"):
            input_data = np.array([[period, duration, depth, radius]])
            
            with st.spinner("Analyzing with AI..."):
                shap_vals, prediction, proba, expected_value = explain_prediction_shap(
                    st.session_state.model,
                    st.session_state.scaler,
                    input_data,
                    st.session_state.feature_names
                )
            
            # Display result
            is_exoplanet = prediction == 1
            confidence = proba[1] if is_exoplanet else proba[0]
            
            if is_exoplanet:
                st.markdown(f"""
                <div class="result-exoplanet">
                    <h1 style="margin: 0; color: #10b981;">✅ EXOPLANET DETECTED!</h1>
                    <h2 style="margin: 1rem 0 0 0; color: #6ee7b7;">Confidence: {confidence:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-not-exoplanet">
                    <h1 style="margin: 0; color: #ef4444;">❌ NOT AN EXOPLANET</h1>
                    <h2 style="margin: 1rem 0 0 0; color: #fca5a5;">Confidence: {confidence:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # SHAP Explanation
            st.markdown("### 🧠 AI Explanation (SHAP Analysis)")
            st.write("**Why did the AI make this classification?**")
            
            feature_names = st.session_state.feature_names
            
            # Create waterfall-style explanation
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # SHAP bar plot
                fig = go.Figure()
                
                colors = ['#ef4444' if v < 0 else '#10b981' for v in shap_vals]
                
                fig.add_trace(go.Bar(
                    y=feature_names,
                    x=shap_vals,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in shap_vals],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Feature Impact on Prediction",
                    xaxis_title="SHAP Value (Impact)",
                    yaxis_title="Feature",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Feature Values")
                st.write(f"**Period**: {period:.2f} days")
                st.write(f"**Duration**: {duration:.2f} hours")
                st.write(f"**Depth**: {depth:.0f} ppm")
                st.write(f"**Radius**: {radius:.2f} R⊕")
                
                st.markdown("### 🎯 Prediction Details")
                st.write(f"**Exoplanet Prob**: {proba[1]:.1%}")
                st.write(f"**Not Exoplanet Prob**: {proba[0]:.1%}")
            
            st.markdown("---")
            st.markdown("### 💡 Interpretation Guide")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("""
                **Positive SHAP values** (green bars) push the prediction towards EXOPLANET
                """)
            with col2:
                st.info("""
                **Negative SHAP values** (red bars) push the prediction towards NOT EXOPLANET
                """)
    
    with tab2:
        st.markdown("### Upload Data File")
        st.write("Supported formats: CSV, Excel (.xlsx, .xls)")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain columns: period, duration, depth, radius"
        )
        
        if uploaded_file:
            df = parse_uploaded_file(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Loaded file with {len(df)} rows")
                
                # Find required columns
                found_cols = find_required_columns(df)
                
                st.write("**Column Detection:**")
                if len(found_cols) == 4:
                    for feat, col in found_cols.items():
                        st.write(f"✅ {feat.title()}: `{col}`")
                else:
                    st.error(f"❌ Could not find all required columns!")
                    st.write(f"**Found ({len(found_cols)}/4):**")
                    for feat, col in found_cols.items():
                        st.write(f"- {feat.title()}: `{col}`")
                    st.write(f"\n**Missing:** {set(['period', 'duration', 'depth', 'radius']) - set(found_cols.keys())}")
                    st.write(f"\n**Available columns in file:**")
                    for col in df.columns[:20]:
                        st.write(f"- {col}")
                    if len(df.columns) > 20:
                        st.write(f"... and {len(df.columns) - 20} more columns")
                
                if len(found_cols) == 4:
                    
                    if st.button("🚀 Predict All", use_container_width=True, type="primary"):
                        # Extract features
                        X = df[[found_cols[f] for f in ['period', 'duration', 'depth', 'radius']]].copy()
                        X.columns = ['period', 'duration', 'depth', 'radius']
                        
                        # Remove NaN and infinite values
                        X = X.replace([np.inf, -np.inf], np.nan)
                        X = X.dropna()
                        
                        if len(X) == 0:
                            st.error("No valid data after cleaning!")
                            st.stop()
                        
                        with st.spinner(f"Predicting {len(X)} entries..."):
                            # Preserve original indices and any name/ID columns
                            original_indices = X.index
                            
                            # Try to find a name/ID column in the original dataframe
                            name_col = None
                            possible_name_cols = ['name', 'pl_name', 'planet_name', 'kepid', 'koi', 'toi', 'epic', 'tic_id', 'hostname']
                            for col_name in possible_name_cols:
                                name_col = find_column(df, [col_name], exclude_keywords=[])
                                if name_col:
                                    break
                            
                            # Scale and predict
                            X_scaled = st.session_state.scaler.transform(X)
                            predictions = st.session_state.model.predict(X_scaled)
                            probas = st.session_state.model.predict_proba(X_scaled)
                            
                            # Add results to dataframe with index reset
                            result_df = X.reset_index(drop=True).copy()
                            
                            # Add identifier column
                            if name_col and name_col in df.columns:
                                identifiers = df.loc[original_indices, name_col].reset_index(drop=True)
                                result_df.insert(0, 'Identifier', identifiers)
                            else:
                                result_df.insert(0, 'Entry_ID', [f"Entry-{i+1}" for i in range(len(result_df))])
                            
                            result_df['Classification'] = ['Exoplanet' if p == 1 else 'Not Exoplanet' for p in predictions]
                            result_df['Confidence'] = [f"{probas[i][predictions[i]]:.1%}" for i in range(len(predictions))]
                            result_df['Exoplanet_Prob'] = [f"{probas[i][1]:.1%}" for i in range(len(predictions))]
                            result_df['Non_Exoplanet_Prob'] = [f"{probas[i][0]:.1%}" for i in range(len(predictions))]
                        
                        st.success("✅ Predictions complete!")
                        
                        # Get lists of identifiers
                        id_col = 'Identifier' if 'Identifier' in result_df.columns else 'Entry_ID'
                        exoplanet_list = result_df[result_df['Classification'] == 'Exoplanet'][id_col].tolist()
                        non_exoplanet_list = result_df[result_df['Classification'] == 'Not Exoplanet'][id_col].tolist()
                        
                        # Summary
                        col1, col2, col3 = st.columns(3)
                        exo_count = (predictions == 1).sum()
                        non_exo_count = (predictions == 0).sum()
                        
                        with col1:
                            st.metric("Total Analyzed", len(predictions))
                        with col2:
                            st.metric("🟢 Exoplanets Found", exo_count)
                        with col3:
                            st.metric("🔴 Not Exoplanets", non_exo_count)
                        
                        st.markdown("---")
                        
                        # Display Lists of Names
                        st.markdown("### 📝 Classification Lists")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"#### 🟢 Exoplanets ({len(exoplanet_list)})")
                            if len(exoplanet_list) > 0:
                                st.markdown("""
                                <div style="background: linear-gradient(135deg, #6BCB7720, #10b98120); 
                                            border-left: 4px solid #10b981; 
                                            border-radius: 8px; 
                                            padding: 1rem; 
                                            max-height: 400px; 
                                            overflow-y: auto;">
                                """, unsafe_allow_html=True)
                                for i, name in enumerate(exoplanet_list, 1):
                                    st.write(f"{i}. **{name}**")
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Download button for exoplanet list
                                exo_text = "\n".join([f"{i}. {name}" for i, name in enumerate(exoplanet_list, 1)])
                                st.download_button(
                                    label="📥 Download Exoplanet List",
                                    data=exo_text,
                                    file_name="exoplanets_list.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            else:
                                st.info("No exoplanets detected")
                        
                        with col2:
                            st.markdown(f"#### 🔴 Not Exoplanets ({len(non_exoplanet_list)})")
                            if len(non_exoplanet_list) > 0:
                                st.markdown("""
                                <div style="background: linear-gradient(135deg, #FF6B6B20, #ef444420); 
                                            border-left: 4px solid #ef4444; 
                                            border-radius: 8px; 
                                            padding: 1rem; 
                                            max-height: 400px; 
                                            overflow-y: auto;">
                                """, unsafe_allow_html=True)
                                for i, name in enumerate(non_exoplanet_list, 1):
                                    st.write(f"{i}. **{name}**")
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Download button for non-exoplanet list
                                non_exo_text = "\n".join([f"{i}. {name}" for i, name in enumerate(non_exoplanet_list, 1)])
                                st.download_button(
                                    label="📥 Download Not Exoplanet List",
                                    data=non_exo_text,
                                    file_name="not_exoplanets_list.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            else:
                                st.info("All entries classified as exoplanets")
                        
                        # Visualization
                        st.markdown("---")
                        st.markdown("### 📊 Results Distribution")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            pred_counts = pd.Series(predictions).value_counts()
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=['Not Exoplanet', 'Exoplanet'],
                                values=[pred_counts.get(0, 0), pred_counts.get(1, 0)],
                                marker_colors=['#ef4444', '#10b981'],
                                hole=0.4
                            )])
                            
                            fig.update_layout(title="Classification Results", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 📈 Detection Statistics")
                            st.write(f"**Detection Rate**: {exo_count/len(predictions):.1%}")
                            st.write(f"**Total Entries**: {len(predictions)}")
                            st.write(f"**Exoplanets**: {exo_count} ({exo_count/len(predictions):.1%})")
                            st.write(f"**Not Exoplanets**: {non_exo_count} ({non_exo_count/len(predictions):.1%})")
                            
                            st.markdown("---")
                            
                            avg_conf_exo = probas[predictions == 1, 1].mean() if exo_count > 0 else 0
                            avg_conf_non = probas[predictions == 0, 0].mean() if non_exo_count > 0 else 0
                            
                            st.write(f"**Avg Confidence (Exoplanets)**: {avg_conf_exo:.1%}")
                            st.write(f"**Avg Confidence (Not Exoplanets)**: {avg_conf_non:.1%}")
                        
                        st.markdown("---")
                        
                        # Tabbed view for detailed results
                        tab1, tab2, tab3 = st.tabs(["📋 All Results", "🟢 Exoplanets Details", "🔴 Not Exoplanets Details"])
                        
                        with tab1:
                            st.markdown("### All Predictions")
                            st.dataframe(result_df, use_container_width=True, height=400)
                        
                        with tab2:
                            exoplanet_df = result_df[result_df['Classification'] == 'Exoplanet']
                            st.markdown(f"### Detected Exoplanets - Full Details ({len(exoplanet_df)} entries)")
                            if len(exoplanet_df) > 0:
                                st.dataframe(exoplanet_df, use_container_width=True, height=400)
                            else:
                                st.info("No exoplanets detected in this dataset")
                        
                        with tab3:
                            non_exoplanet_df = result_df[result_df['Classification'] == 'Not Exoplanet']
                            st.markdown(f"### Not Exoplanets - Full Details ({len(non_exoplanet_df)} entries)")
                            if len(non_exoplanet_df) > 0:
                                st.dataframe(non_exoplanet_df, use_container_width=True, height=400)
                            else:
                                st.info("All entries classified as exoplanets")
                        
                        # Download button
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name="exoplanet_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

# Page: Batch Analysis
elif page == "📊 Batch Analysis":
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first!")
        st.info("Go to **Train Model** page to train the AI")
        st.stop()
    
    st.markdown("## 📊 Batch Analysis")
    st.write("Upload multiple files for comprehensive analysis")
    
    uploaded_files = st.file_uploader(
        "Upload multiple CSV/Excel files (max 3)",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("Maximum 3 files. Processing first 3 only.")
            uploaded_files = uploaded_files[:3]
        
        if st.button("🚀 Analyze All Files", use_container_width=True, type="primary"):
            results = []
            
            progress_bar = st.progress(0)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                st.write(f"Processing: **{uploaded_file.name}**")
                
                df = parse_uploaded_file(uploaded_file)
                
                if df is not None:
                    # Find columns using improved function
                    found_cols = find_required_columns(df)
                    
                    if len(found_cols) == 4:
                        try:
                            # Preserve original indices
                            original_indices = df.index
                            
                            # Try to find a name/ID column
                            name_col = None
                            possible_name_cols = ['name', 'pl_name', 'planet_name', 'kepid', 'koi', 'toi', 'epic', 'tic_id', 'hostname']
                            for col_name in possible_name_cols:
                                name_col = find_column(df, [col_name], exclude_keywords=[])
                                if name_col:
                                    break
                            
                            X = df[[found_cols[f] for f in ['period', 'duration', 'depth', 'radius']]].copy()
                            X.columns = ['period', 'duration', 'depth', 'radius']
                            
                            # Remove NaN and infinite values
                            X = X.replace([np.inf, -np.inf], np.nan)
                            valid_indices = X.dropna().index
                            X = X.loc[valid_indices]
                            
                            if len(X) > 0:
                                # Get identifiers for valid entries
                                if name_col and name_col in df.columns:
                                    identifiers = df.loc[valid_indices, name_col].tolist()
                                else:
                                    identifiers = [f"Entry-{i+1}" for i in range(len(X))]
                                
                                # Predict
                                X_scaled = st.session_state.scaler.transform(X)
                                predictions = st.session_state.model.predict(X_scaled)
                                probas = st.session_state.model.predict_proba(X_scaled)
                                
                                # Create lists
                                exoplanet_names = [identifiers[i] for i in range(len(predictions)) if predictions[i] == 1]
                                non_exoplanet_names = [identifiers[i] for i in range(len(predictions)) if predictions[i] == 0]
                                
                                results.append({
                                    'filename': uploaded_file.name,
                                    'total': len(predictions),
                                    'exoplanets': (predictions == 1).sum(),
                                    'not_exoplanets': (predictions == 0).sum(),
                                    'predictions': predictions,
                                    'probas': probas,
                                    'exoplanet_names': exoplanet_names,
                                    'non_exoplanet_names': non_exoplanet_names,
                                    'identifiers': identifiers
                                })
                                st.success(f"✅ {uploaded_file.name}: Processed {len(predictions)} entries")
                            else:
                                st.warning(f"⚠️ {uploaded_file.name}: No valid data after cleaning")
                        except Exception as e:
                            st.error(f"❌ {uploaded_file.name}: Error during prediction - {str(e)}")
                    else:
                        st.error(f"❌ {uploaded_file.name}: Missing required columns")
                        st.write(f"Found: {list(found_cols.keys())}")
                        st.write(f"Available columns: {list(df.columns[:10])}...")
                else:
                    st.error(f"❌ {uploaded_file.name}: Could not read file")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            progress_bar.empty()
            
            if results:
                st.success(f"✅ Processed {len(results)} files successfully!")
                st.markdown("---")
                
                # Display classified names for each file
                st.markdown("### 📝 Classified Objects by File")
                
                for result in results:
                    with st.expander(f"📁 {result['filename']} - {result['exoplanets']} Exoplanets, {result['not_exoplanets']} Non-Exoplanets", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"#### 🟢 Exoplanets ({len(result['exoplanet_names'])})")
                            if len(result['exoplanet_names']) > 0:
                                st.markdown("""
                                <div style="background: linear-gradient(135deg, #6BCB7720, #10b98120); 
                                            border-left: 4px solid #10b981; 
                                            border-radius: 8px; 
                                            padding: 1rem; 
                                            max-height: 300px; 
                                            overflow-y: auto;">
                                """, unsafe_allow_html=True)
                                for i, name in enumerate(result['exoplanet_names'], 1):
                                    st.write(f"{i}. **{name}**")
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Download button
                                exo_text = "\n".join([f"{i}. {name}" for i, name in enumerate(result['exoplanet_names'], 1)])
                                st.download_button(
                                    label=f"📥 Download Exoplanets",
                                    data=exo_text,
                                    file_name=f"{result['filename']}_exoplanets.txt",
                                    mime="text/plain",
                                    key=f"exo_{result['filename']}",
                                    use_container_width=True
                                )
                            else:
                                st.info("No exoplanets detected")
                        
                        with col2:
                            st.markdown(f"#### 🔴 Not Exoplanets ({len(result['non_exoplanet_names'])})")
                            if len(result['non_exoplanet_names']) > 0:
                                st.markdown("""
                                <div style="background: linear-gradient(135deg, #FF6B6B20, #ef444420); 
                                            border-left: 4px solid #ef4444; 
                                            border-radius: 8px; 
                                            padding: 1rem; 
                                            max-height: 300px; 
                                            overflow-y: auto;">
                                """, unsafe_allow_html=True)
                                for i, name in enumerate(result['non_exoplanet_names'], 1):
                                    st.write(f"{i}. **{name}**")
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Download button
                                non_exo_text = "\n".join([f"{i}. {name}" for i, name in enumerate(result['non_exoplanet_names'], 1)])
                                st.download_button(
                                    label=f"📥 Download Non-Exoplanets",
                                    data=non_exo_text,
                                    file_name=f"{result['filename']}_non_exoplanets.txt",
                                    mime="text/plain",
                                    key=f"non_exo_{result['filename']}",
                                    use_container_width=True
                                )
                            else:
                                st.info("All entries classified as exoplanets")
                
                st.markdown("---")
                
                # Summary cards
                st.markdown("### 📊 Summary Statistics")
                cols = st.columns(len(results))
                
                for idx, result in enumerate(results):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin: 0; color: #667eea;">{result['filename'][:20]}...</h3>
                            <p style="margin: 0.5rem 0;"><strong>Total:</strong> {result['total']}</p>
                            <p style="margin: 0.5rem 0; color: #10b981;"><strong>Exoplanets:</strong> {result['exoplanets']}</p>
                            <p style="margin: 0.5rem 0; color: #ef4444;"><strong>Not Exoplanets:</strong> {result['not_exoplanets']}</p>
                            <p style="margin: 0.5rem 0;"><strong>Rate:</strong> {result['exoplanets']/result['total']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Comparison charts
                st.markdown("### 📈 Comparison Analysis")
                
                tab1, tab2 = st.tabs(["Distribution", "Detection Rates"])
                
                with tab1:
                    fig_cols = st.columns(len(results))
                    
                    for idx, result in enumerate(results):
                        with fig_cols[idx]:
                            fig = go.Figure(data=[go.Pie(
                                labels=['Not Exoplanet', 'Exoplanet'],
                                values=[result['not_exoplanets'], result['exoplanets']],
                                marker_colors=['#ef4444', '#10b981'],
                                hole=0.3
                            )])
                            
                            fig.update_layout(
                                title=result['filename'][:20],
                                height=300,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Bar chart comparison
                    fig = go.Figure()
                    
                    filenames = [r['filename'][:15] for r in results]
                    exo_counts = [r['exoplanets'] for r in results]
                    not_exo_counts = [r['not_exoplanets'] for r in results]
                    
                    fig.add_trace(go.Bar(
                        name='Exoplanets',
                        x=filenames,
                        y=exo_counts,
                        marker_color='#10b981'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Not Exoplanets',
                        x=filenames,
                        y=not_exo_counts,
                        marker_color='#ef4444'
                    ))
                    
                    fig.update_layout(
                        title="Detection Comparison Across Files",
                        xaxis_title="File",
                        yaxis_title="Count",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No files could be processed successfully")

# Page: Model Info
elif page == "📈 Model Info":
    st.markdown("## 📈 Model Information")
    
    if st.session_state.model is None:
        st.warning("⚠️ No model trained yet")
        st.info("Go to **Train Model** page to train the AI")
        st.stop()
    
    metrics = st.session_state.model_metrics
    
    # Performance Overview
    st.markdown("### 🎯 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #667eea; margin: 0;">{metrics['accuracy']:.1%}</h2>
            <p style="margin: 0.5rem 0 0 0;">Accuracy</p>
            <small>Overall correctness</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #10b981; margin: 0;">{metrics['precision']:.1%}</h2>
            <p style="margin: 0.5rem 0 0 0;">Precision</p>
            <small>True positives accuracy</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #f59e0b; margin: 0;">{metrics['recall']:.1%}</h2>
            <p style="margin: 0.5rem 0 0 0;">Recall</p>
            <small>Detection rate</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #8b5cf6; margin: 0;">{metrics['f1']:.1%}</h2>
            <p style="margin: 0.5rem 0 0 0;">F1-Score</p>
            <small>Harmonic mean</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Confusion Matrix")
        cm = metrics['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Not Exoplanet', 'Exoplanet'],
            y=['Not Exoplanet', 'Exoplanet'],
            colorscale='Purples',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
        ))
        
        fig.update_layout(
            title="Prediction vs Actual",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📋 Model Specifications")
        st.write(f"**Algorithm**: Random Forest Classifier")
        st.write(f"**Number of Trees**: 200")
        st.write(f"**Max Depth**: 15")
        st.write(f"**Features**: 4 (Period, Duration, Depth, Radius)")
        st.write(f"**Training Samples**: {metrics['train_size']:,}")
        st.write(f"**Test Samples**: {metrics['test_size']:,}")
    
    with col2:
        st.markdown("### 🎯 Classification Breakdown")
        
        cm = metrics['confusion_matrix']
        
        st.write(f"**True Negatives (TN)**: {cm[0,0]}")
        st.caption("Correctly identified non-exoplanets")
        
        st.write(f"**False Positives (FP)**: {cm[0,1]}")
        st.caption("Non-exoplanets incorrectly classified as exoplanets")
        
        st.write(f"**False Negatives (FN)**: {cm[1,0]}")
        st.caption("Exoplanets incorrectly classified as non-exoplanets")
        
        st.write(f"**True Positives (TP)**: {cm[1,1]}")
        st.caption("Correctly identified exoplanets")
        
        st.markdown("---")
        
        st.markdown("### 💡 What These Metrics Mean")
        
        st.info(f"""
        **Accuracy ({metrics['accuracy']:.1%})**: Of all predictions, {metrics['accuracy']:.1%} were correct.
        
        **Precision ({metrics['precision']:.1%})**: When the model predicts "exoplanet", it's correct {metrics['precision']:.1%} of the time.
        
        **Recall ({metrics['recall']:.1%})**: The model detects {metrics['recall']:.1%} of all actual exoplanets.
        
        **F1-Score ({metrics['f1']:.1%})**: Balanced measure between precision and recall.
        """)
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### 🌟 Feature Importance")
    st.write("Which features are most important for detection?")
    
    feature_importance = st.session_state.model.feature_importances_
    feature_names = st.session_state.feature_names
    
    fig = go.Figure()
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    
    fig.add_trace(go.Bar(
        x=feature_names,
        y=feature_importance,
        marker_color=colors,
        text=[f"{v:.1%}" for v in feature_importance],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Feature Importance in Random Forest Model",
        xaxis_title="Feature",
        yaxis_title="Importance",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🧠 About the Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Random Forest** is an ensemble learning method that:
        - Uses multiple decision trees for better accuracy
        - Reduces overfitting through averaging
        - Handles non-linear relationships well
        - Provides feature importance naturally
        """)
    
    with col2:
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)**:
        - Explains individual predictions
        - Shows feature contribution
        - Based on game theory
        - Provides transparent AI decisions
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; padding: 2rem 0;">
    <p>🌌 Exoplanet AI Detector | Powered by Random Forest & SHAP</p>
    <p>Data sources: NASA Kepler, K2, TESS Missions</p>
</div>
""", unsafe_allow_html=True)