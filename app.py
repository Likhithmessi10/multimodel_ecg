import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import joblib
import wfdb

from zero_leakage_loader import PTBXLZeroLeakageLoader
from multimodal_fusion_net import LateFusionNet, SignalOnlyNet

# Page configuration
st.set_page_config(
    page_title="AI Cardiac Diagnostics Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium design
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Main page styling */
        .main-title {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #FF4B4B, #8B0000);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 0.2rem;
            text-align: center;
        }
        
        .sub-title {
            font-family: 'Outfit', sans-serif;
            color: #555555;
            font-weight: 400;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        /* Glassmorphic cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 75, 75, 0.15);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(255, 75, 75, 0.15);
            border-color: rgba(255, 75, 75, 0.4);
        }
        
        .metric-label {
            font-weight: 600;
            font-size: 1.1rem;
            color: #333333;
            margin-bottom: 0.5rem;
        }
        
        /* Custom buttons */
        .stButton>button {
            background: linear-gradient(135deg, #FF4B4B, #CC1111);
            color: white;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 2rem;
            box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #CC1111, #8B0000);
            box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5);
            transform: scale(1.02);
            color: white;
        }
        
        /* Section headers */
        h2 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: #1E1E1E;
            border-bottom: 2px solid #FF4B4B;
            padding-bottom: 0.3rem;
            margin-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Cache data loading for faster updates
@st.cache_resource
def get_loader():
    return PTBXLZeroLeakageLoader(data_dir="./data")

@st.cache_data
def load_metadata_df():
    loader = get_loader()
    return loader.load_raw_metadata()

# Load models safely
def load_trained_models():
    models_dir = "./models"
    cache_dir = "./data/cache"
    
    scaler_path = os.path.join(cache_dir, "metadata_scaler.pkl")
    impute_path = os.path.join(cache_dir, "metadata_impute_values.pkl")
    
    if not (os.path.exists(scaler_path) and os.path.exists(impute_path)):
        return None, None, None, None, None
        
    scaler = joblib.load(scaler_path)
    impute_values = joblib.load(impute_path)
    
    # Load XGBoost Tabular Model
    xgb_path = os.path.join(models_dir, "tabular_xgb_model.joblib")
    xgb_model = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    
    # Load PyTorch Models
    sig_model_path = os.path.join(models_dir, "signal_only_model.pth")
    sig_model = SignalOnlyNet(in_channels=12, num_classes=5)
    if os.path.exists(sig_model_path):
        # map_location='cpu' ensures load on CPU if running Streamlit locally
        sig_model.load_state_dict(torch.load(sig_model_path, map_location='cpu'))
        sig_model.eval()
    else:
        sig_model = None
        
    fus_model_path = os.path.join(models_dir, "late_fusion_model.pth")
    fus_model = LateFusionNet(in_channels=12, meta_features=4, num_classes=5)
    if os.path.exists(fus_model_path):
        fus_model.load_state_dict(torch.load(fus_model_path, map_location='cpu'))
        fus_model.eval()
    else:
        fus_model = None
        
    return scaler, impute_values, xgb_model, sig_model, fus_model

def get_patient_data(ecg_id, df_raw, loader):
    row = df_raw.loc[ecg_id]
    file_path = os.path.join("./data", row['filename_lr'])
    
    # Read signal
    signal_data, _ = wfdb.rdsamp(file_path)
    # Handle NaNs in signal
    signal_data = np.nan_to_num(signal_data, nan=0.0)
    
    # Apply FIR Filter
    filtered_signal = loader._apply_filter(signal_data)
    
    return signal_data.T, filtered_signal.T, row

def run_predictions(raw_meta, filtered_wave, scaler, impute_values, xgb_model, sig_model, fus_model):
    # Prepare metadata: [age, sex, height, weight]
    # Handle imputation and scaling
    meta_df = pd.DataFrame([raw_meta], columns=['age', 'sex', 'height', 'weight'])
    for col in ['age', 'height', 'weight']:
        meta_df[col] = meta_df[col].fillna(impute_values[col])
    
    meta_scaled = scaler.transform(meta_df)[0]
    
    # Convert inputs to torch tensors
    t_meta = torch.tensor([meta_scaled], dtype=torch.float32)
    
    # Waveform shape: (1, 12, 1000)
    t_wave = torch.tensor([filtered_wave], dtype=torch.float32)
    
    results = {}
    
    # 1. Tabular Only (XGBoost)
    if xgb_model is not None:
        xgb_input = meta_df.values
        xgb_prob_list = xgb_model.predict_proba(xgb_input)
        y_prob_xgb = [p[0][1] for p in xgb_prob_list]
        results['Tabular-Only'] = y_prob_xgb
        
    # 2. Signal Only (1D-ResNet)
    if sig_model is not None:
        with torch.no_grad():
            probs = sig_model(t_wave)[0].numpy()
            results['Signal-Only'] = probs
            
    # 3. Late Fusion Model
    if fus_model is not None:
        with torch.no_grad():
            probs = fus_model(t_wave, t_meta)[0].numpy()
            results['Late-Fusion'] = probs
            
    return results

def main():
    st.markdown('<div class="main-title">❤️ Multi-Modal Late-Fusion ECG Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">1D-ResNet Waveform Feature Extraction + Optimized Tabular Metadata Fusion</div>', unsafe_allow_html=True)
    
    # Check if models exist
    scaler, impute_values, xgb_model, sig_model, fus_model = load_trained_models()
    
    if scaler is None or fus_model is None:
        st.error("⚠️ Model checkpoints, scalar configurations, or training weights were not found.")
        st.info("💡 Please execute the training script (`python trainer.py`) first. This will train the baselines and generate the required models and paper artifacts.")
        return
        
    # Data Loading
    loader = get_loader()
    df_raw = load_metadata_df()
    
    # Filter for Fold 10 (Independent Test Split)
    df_test = df_raw[df_raw['strat_fold'] == 10]
    
    # Sidebar: Patient Selection & Metadata Parameters
    st.sidebar.header("🎯 Patient Record Select")
    
    # Dropdown for ECG IDs from Fold 10
    test_ecg_ids = df_test.index.tolist()
    selected_ecg_id = st.sidebar.selectbox("Choose patient record from Test Set (Fold 10):", test_ecg_ids)
    
    # Fetch original data for the patient
    raw_wave, filtered_wave, patient_row = get_patient_data(selected_ecg_id, df_raw, loader)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Clinical Metadata")
    
    # Prefill from selected patient, but let user change
    age = st.sidebar.number_input("Age (years):", min_value=1, max_value=120, value=int(patient_row['age']) if pd.notna(patient_row['age']) else 50)
    sex_str = st.sidebar.selectbox("Sex:", ["Male", "Female"], index=0 if patient_row['sex'] == 0 else 1)
    sex = 0 if sex_str == "Male" else 1
    
    weight = st.sidebar.number_input("Weight (kg):", min_value=1.0, max_value=250.0, value=float(patient_row['weight']) if pd.notna(patient_row['weight']) else 75.0)
    height = st.sidebar.number_input("Height (cm):", min_value=30.0, max_value=250.0, value=float(patient_row['height']) if pd.notna(patient_row['height']) else 170.0)
    
    # Button to execute diagnostic analysis
    run_analysis = st.sidebar.button("Run Diagnostic Fusion Analysis", use_container_width=True)
    
    # Display details of selected patient
    st.markdown("## 🔍 Active Record Information")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("ECG ID", f"#{selected_ecg_id}")
    with col2:
        st.metric("Patient ID", f"#{int(patient_row['patient_id'])}")
    with col3:
        original_diagnoses = ", ".join(patient_row['diagnostic_superclass'])
        st.metric("Ground Truth Labels", original_diagnoses if original_diagnoses else "NORM")
    with col4:
        st.metric("Sampling Rate", f"{loader.fs} Hz")
    with col5:
        st.metric("Signal Duration", "10.0 seconds")
        
    # Main Section tabs
    tab1, tab2, tab3 = st.tabs(["📊 12-Lead Time-Series Waveforms", "🩺 Diagnostic Predictions", "📄 Paper Artifacts Preview"])
    
    # TAB 1: ECG Waveforms Visualizer
    with tab1:
        st.markdown("### 📈 Interactive 12-Lead Subplots")
        st.write("This plot shows the 12 ECG leads. **Light grey dashed lines** represent the raw signal containing noise and baseline wander. The **crimson solid lines** represent the filtered output processed by our 0.5Hz - 45Hz Bandpass FIR Filter.")
        
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Subplot structure: 6 rows, 2 columns (height is 900px)
        fig = make_subplots(
            rows=6, cols=2,
            subplot_titles=lead_names,
            shared_xaxes=True,
            vertical_spacing=0.06,
            horizontal_spacing=0.08
        )
        
        time_axis = np.arange(1000) / loader.fs # 10 seconds x-axis
        
        for i, lead in enumerate(lead_names):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Raw signal trace
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=raw_wave[i],
                    name=f"Lead {lead} (Raw)",
                    line=dict(color='rgba(150, 150, 150, 0.45)', width=1, dash='dash'),
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            # Filtered signal trace
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=filtered_wave[i],
                    name=f"Lead {lead} (Filtered)",
                    line=dict(color='rgba(220, 20, 60, 0.95)', width=1.5),
                    hoverinfo='x+y'
                ),
                row=row, col=col
            )
            
        fig.update_layout(
            height=1000,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes styles to display grid
        for idx in range(1, 13):
            fig.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.25)', row=(idx-1)//2+1, col=(idx-1)%2+1)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.25)', row=(idx-1)//2+1, col=(idx-1)%2+1)
            
        st.plotly_chart(fig, use_container_width=True)
        
    # TAB 2: Predictions and Ablation Comparisons
    with tab2:
        if not run_analysis:
            st.info("👈 Click the **Run Diagnostic Fusion Analysis** button in the sidebar to feed the waveforms and metadata into the networks and calculate predictions.")
        else:
            raw_meta = [age, sex, height, weight]
            
            with st.spinner("Processing modalities and executing model inference..."):
                results = run_predictions(
                    raw_meta, 
                    filtered_wave, 
                    scaler, 
                    impute_values, 
                    xgb_model, 
                    sig_model, 
                    fus_model
                )
                
            st.success("✅ Diagnostic Inference complete.")
            
            # Layout predictions
            col_chart, col_details = st.columns([3, 2])
            
            with col_chart:
                st.markdown("### 📊 Diagnostic Probabilities (Late-Fusion Model)")
                
                superclasses = loader.superclasses
                probs = results['Late-Fusion']
                
                # Plot probabilities
                fig_bar = go.Figure()
                
                # Threshold line at 0.5
                fig_bar.add_vline(x=0.5, line_dash="dash", line_color="red", 
                                  annotation_text="Decision Threshold (0.50)", annotation_position="top right")
                
                # Custom coloring for bars based on threshold crossing
                bar_colors = ['#FF4B4B' if p >= 0.5 else '#1f77b4' for p in probs]
                
                fig_bar.add_trace(go.Bar(
                    y=superclasses,
                    x=probs,
                    orientation='h',
                    marker_color=bar_colors,
                    text=[f"{p:.2%}" for p in probs],
                    textposition='auto',
                    hoverinfo='x+y'
                ))
                
                fig_bar.update_layout(
                    xaxis=dict(title="Probability", range=[0.0, 1.05], tickformat="%"),
                    yaxis=dict(title="Cardiac Diagnostic Superclass", categoryorder='array', categoryarray=superclasses[::-1]),
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor='white',
                )
                fig_bar.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.25)')
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with col_details:
                st.markdown("### 📋 Multi-Model Diagnostic Comparison")
                
                # Make a DataFrame to compare models
                comparison_df = pd.DataFrame(index=superclasses)
                if 'Tabular-Only' in results:
                    comparison_df['Tabular-Only (XGBoost)'] = [f"{p:.2%}" for p in results['Tabular-Only']]
                if 'Signal-Only' in results:
                    comparison_df['Signal-Only (1D-ResNet)'] = [f"{p:.2%}" for p in results['Signal-Only']]
                if 'Late-Fusion' in results:
                    comparison_df['Proposed Fused Model'] = [f"{p:.2%}" for p in results['Late-Fusion']]
                    
                st.table(comparison_df)
                
                st.markdown("""
                    * **NORM**: Normal ECG
                    * **MI**: Myocardial Infarction
                    * **STTC**: ST/T Change
                    * **CD**: Conduction Disturbance
                    * **HYP**: Hypertrophy
                """)
                
            # Full diagnostic report card
            st.markdown("### 🫀 Clinical Diagnostic Diagnostic Report Summary")
            
            diagnostic_alarms = []
            for i, cls in enumerate(superclasses):
                if probs[i] >= 0.5:
                    diagnostic_alarms.append(cls)
                    
            if len(diagnostic_alarms) == 0:
                st.markdown("""
                    <div style="background-color: #d4edda; color: #155724; padding: 1.2rem; border-radius: 8px; border: 1px solid #c3e6cb;">
                        <h4 style="margin-top:0;">🟢 NORMAL CARDIAC DIAGNOSTIC REPORT</h4>
                        No abnormal ECG diagnostic superclass exceeds the 50% probability threshold. The signal features and metadata characteristics align within expected standard limits. Please correlate with clinical signs.
                    </div>
                """, unsafe_allow_html=True)
            else:
                alarms_str = ", ".join(diagnostic_alarms)
                st.markdown(f"""
                    <div style="background-color: #f8d7da; color: #721c24; padding: 1.2rem; border-radius: 8px; border: 1px solid #f5c6cb;">
                        <h4 style="margin-top:0;">🔴 WARNING: DETECTED CARDIAC ABNORMALITIES</h4>
                        The multi-modal model predicts a high probability of the following diagnostic superclasses: <strong>{alarms_str}</strong> (probability &ge; 50%). 
                        Further diagnostic investigation (e.g. echocardiogram, troponin, cardiology referral) is recommended.
                    </div>
                """, unsafe_allow_html=True)

    # TAB 3: Paper Artifacts Preview
    with tab3:
        st.markdown("### 📄 Manuscript Preparation Artifacts")
        st.write("Below are the exact empirical results and high-resolution charts generated during the model training pipeline. These are exported directly under `./paper_artifacts/` for manuscript preparation.")
        
        art_dir = "./paper_artifacts"
        if not os.path.exists(art_dir):
            st.warning("⚠️ Paper artifacts folder not found. Please run the training script (`python trainer.py`) to generate the artifacts.")
        else:
            subcol1, subcol2 = st.columns(2)
            
            with subcol1:
                st.markdown("#### Table 2: Model Performance Ablation Comparison")
                t2_path = os.path.join(art_dir, "table2_ablation_results.csv")
                if os.path.exists(t2_path):
                    t2_df = pd.read_csv(t2_path)
                    st.dataframe(t2_df, hide_index=True)
                else:
                    st.info("Table 2 not found.")
                    
                st.markdown("#### Table 3: Statistical Significance Validation")
                t3_path = os.path.join(art_dir, "table3_statistical_significance.csv")
                if os.path.exists(t3_path):
                    t3_df = pd.read_csv(t3_path)
                    st.dataframe(t3_df, hide_index=True)
                else:
                    st.info("Table 3 not found.")
                    
            with subcol2:
                st.markdown("#### Table 1: Model Hyperparameters & Environment")
                t1_path = os.path.join(art_dir, "table1_hyperparameters.csv")
                if os.path.exists(t1_path):
                    t1_df = pd.read_csv(t1_path)
                    st.dataframe(t1_df, hide_index=True)
                else:
                    st.info("Table 1 not found.")
            
            st.markdown("---")
            st.markdown("#### High-Resolution Figures")
            
            fig_losses_path = os.path.join(art_dir, "figure1_loss_curves.png")
            fig_roc_path = os.path.join(art_dir, "figure2_roc_curves.png")
            fig_pr_path = os.path.join(art_dir, "figure3_pr_curves.png")
            
            fcol1, fcol2, fcol3 = st.columns(3)
            
            with fcol1:
                st.markdown("**Figure 1: Convergence Curves**")
                if os.path.exists(fig_losses_path):
                    st.image(fig_losses_path, use_container_width=True)
                else:
                    st.info("Figure 1 not found.")
                    
            with fcol2:
                st.markdown("**Figure 2: ROC Sensitivities**")
                if os.path.exists(fig_roc_path):
                    st.image(fig_roc_path, use_container_width=True)
                else:
                    st.info("Figure 2 not found.")
                    
            with fcol3:
                st.markdown("**Figure 3: Precision-Recall curves**")
                if os.path.exists(fig_pr_path):
                    st.image(fig_pr_path, use_container_width=True)
                else:
                    st.info("Figure 3 not found.")

if __name__ == "__main__":
    main()
