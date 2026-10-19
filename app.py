"""
app.py - CropGuardians AI
Primary Streamlit Web Application for AI-Powered Crop Disease Diagnostics.
Engineered for Streamlit Community Cloud and mobile-responsive browser access.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import time
from pathlib import Path
from PIL import Image
import streamlit as st
import numpy as np

# Core internal modules
from src.disease_info import (
    get_disease_info,
    get_all_supported_classes,
    DISEASE_REGISTRY
)
from src.model_utils import (
    load_model_and_classes,
    preprocess_image,
    validate_image_file,
    predict_leaf_disease,
    get_model_file_size_mb,
    CONFIDENCE_THRESHOLD,
    IMG_SIZE
)
from src.evaluation import load_evaluation_report

# Page Configuration
st.set_page_config(
    page_title="CropGuardians AI — Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global font & background accents */
    .main {
        color: #e8f5e9;
    }
    .main-header {
        background: linear-gradient(135deg, #0d1710 0%, #152b1b 100%);
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #22c55e;
        margin-bottom: 4px;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #a7f3d0;
        margin-bottom: 12px;
    }
    .header-desc {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .result-card {
        background: #111d14;
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .healthy-badge {
        display: inline-block;
        background-color: #166534;
        color: #bbf7d0;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .disease-badge {
        display: inline-block;
        background-color: #991b1b;
        color: #fecaca;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .warning-badge {
        display: inline-block;
        background-color: #92400e;
        color: #fde68a;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .metric-box {
        background: #16241a;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid rgba(34, 197, 94, 0.15);
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #4ade80;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .treatment-step {
        background: #132318;
        border-left: 3px solid #22c55e;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)


# Resource Caching
@st.cache_resource(show_spinner="Loading lightweight MobileNetV2 model...")
def get_cached_model():
    """Caches the trained model in memory to prevent reloading across user sessions."""
    model_path = Path("models/mobilenetv2_cropguardians.keras")
    classes_path = Path("models/class_names.json")

    if not model_path.exists() or not classes_path.exists():
        return None, []

    model, class_names = load_model_and_classes(str(model_path), str(classes_path))
    return model, class_names


@st.cache_data(show_spinner=False)
def get_cached_evaluation_report():
    """Caches evaluation metrics loaded from disk."""
    return load_evaluation_report("models/evaluation_metrics.json")


def render_header():
    """Displays product branding and farmer guidance."""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🌿 CropGuardians AI</div>
        <div class="main-subtitle">AI-Powered Crop Disease Detection · Zelvora Technologies Project</div>
        <p class="header-desc">
            Empowering agricultural producers with instant, lightweight plant disease diagnostics.
            Upload or photograph a crop leaf to identify pathogens, evaluate confidence, and receive verified cultural management guidelines.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders navigation, system metadata, and environmental parameters."""
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1592417817098-8f3d69106883?auto=format&fit=crop&w=400&q=80", use_column_width=True)
        st.markdown("### 🌾 CropGuardians Navigation")
        selected_tab = st.radio(
            "Go to section:",
            ["🌱 Disease Detector", "📊 Model Performance", "📖 About Project", "🔬 Dataset & Limitations"],
            index=0
        )
        st.markdown("---")
        st.markdown("#### ⚡ System Specs")
        model_path = Path("models/mobilenetv2_cropguardians.keras")
        if model_path.exists():
            size_mb = get_model_file_size_mb(str(model_path))
            st.success(f"**Model Active:** MobileNetV2 (`{size_mb} MB`)")
        else:
            st.warning("**Model Status:** Not yet trained / loaded")

        st.info(
            "ℹ️ **Inference Notice:** Inference executes on the Streamlit cloud server "
            "to minimize mobile client battery and data consumption."
        )

        st.markdown("---")
        st.caption("CropGuardians AI v2.0 · Zelvora AI Internship Week 2")
        return selected_tab


def render_detector_tab(model, class_names):
    """Primary detector workflow: Image upload, validation, prediction, and recommendations."""
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Upload Leaf Image")
        st.caption("Supported crops: Apple, Corn, Grape, Potato, Tomato. Formats: JPG, JPEG, PNG.")

        # Preset sample picker for rapid testing
        sample_choice = st.selectbox(
            "Or choose a verification sample:",
            ["None (Upload my own image)", "Tomato - Late Blight Sample", "Apple - Scab Sample", "Corn - Common Rust Sample", "Potato - Early Blight Sample", "Grape - Healthy Sample"]
        )

        uploaded_file = st.file_uploader(
            "Choose a leaf image file:",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photograph of a single leaf showing visible symptoms or healthy foliage."
        )

        pil_img = None
        source_name = ""

        # Handle sample image selection
        if sample_choice != "None (Upload my own image)":
            sample_dir = Path("data/sample_leaves")
            sample_map = {
                "Tomato - Late Blight Sample": "Tomato___Late_blight",
                "Apple - Scab Sample": "Apple___Apple_scab",
                "Corn - Common Rust Sample": "Corn_(maize)___Common_rust_",
                "Potato - Early Blight Sample": "Potato___Early_blight",
                "Grape - Healthy Sample": "Grape___healthy"
            }
            folder_name = sample_map.get(sample_choice)
            target_folder = sample_dir / folder_name
            if target_folder.exists() and list(target_folder.glob("*.jpg")):
                first_img = list(target_folder.glob("*.jpg"))[0]
                pil_img = Image.open(first_img).convert("RGB")
                source_name = f"Sample: {sample_choice}"
            else:
                st.info(f"Sample folder `{folder_name}` will be available once sample data is generated.")

        if uploaded_file is not None:
            source_name = uploaded_file.name
            is_valid, err_msg, loaded_img = validate_image_file(uploaded_file)
            if not is_valid:
                st.error(f"❌ Upload Error: {err_msg}")
                pil_img = None
            else:
                pil_img = loaded_img

        if pil_img is not None:
            st.image(pil_img, caption=f"Selected Leaf: {source_name}", use_column_width=True)
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                detect_btn = st.button("🔍 Detect Disease", type="primary", use_container_width=True)
            with col_btn2:
                reset_btn = st.button("🔄 Reset", use_container_width=True)
                if reset_btn:
                    st.rerun()
        else:
            detect_btn = False
            st.info("👆 Please upload an image or select a verification sample to begin.")

    with col_result:
        st.subheader("2. Diagnosis & Guidance")

        if model is None:
            st.warning("⚠️ **Trained Model Not Found**")
            st.write(
                "A trained `.keras` model was not detected in `models/mobilenetv2_cropguardians.keras`.\n\n"
                "**Quick Resolution:**\n"
                "1. Run `python train_model.py --epochs 5` to train the lightweight MobileNetV2 classifier.\n"
                "2. Once training completes, refresh this page."
            )
            return

        if pil_img is not None and detect_btn:
            with st.spinner("Analyzing leaf pathology with MobileNetV2..."):
                try:
                    preprocessed_tensor = preprocess_image(pil_img)
                    pred_data = predict_leaf_disease(model, preprocessed_tensor, class_names)
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    return

            class_id = pred_data["class_name"]
            confidence = pred_data["confidence"]
            confidence_pct = pred_data["confidence_pct"]
            is_low_conf = pred_data["is_low_confidence"]
            latency_ms = pred_data["latency_ms"]

            info = get_disease_info(class_id)

            # Diagnosis Header Banner
            status_cls = "healthy-badge" if info["is_healthy"] else "disease-badge"
            status_text = "✓ Healthy Plant Foliage" if info["is_healthy"] else f"⚠️ {info['severity'].upper()} Severity Condition"

            st.markdown(f"""
            <div class="result-card">
                <span class="{status_cls}">{status_text}</span>
                <h3 style="margin-top: 10px; color: #ffffff;">{info['emoji']} {info['crop']} — {info['disease']}</h3>
                <p style="color: #94a3b8; font-style: italic; margin-bottom: 8px;">
                    Botanical: {info['crop_scientific']} · Pathogen: {info['scientific_name']} ({info['pathogen_type']})
                </p>
                <p style="color: #cbd5e1; font-size: 0.95rem;">{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Latency and Confidence Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{confidence_pct}%</div>
                    <div class="metric-label">Model Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{latency_ms} ms</div>
                    <div class="metric-label">Inference Latency</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{info['severity'].capitalize()}</div>
                    <div class="metric-label">Impact Level</div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence Bar
            st.markdown("#### Confidence Distribution")
            st.progress(float(min(max(confidence, 0.0), 1.0)))

            if is_low_conf:
                st.warning(
                    f"⚠️ **Low Confidence Alert ({confidence_pct}%):** The model probability is below the {int(CONFIDENCE_THRESHOLD*100)}% reliability threshold. "
                    "This image may feature background clutter, poor lighting, or an unsupported plant species. "
                    "We strongly recommend having the plant inspected by a certified agronomist or local agricultural extension."
                )

            # Top candidate breakdown
            with st.expander("📊 View Top Prediction Candidates", expanded=False):
                for cand in pred_data["top_candidates"]:
                    cand_info = get_disease_info(cand["class_name"])
                    st.write(f"- **{cand_info['crop']} — {cand_info['disease']}**: `{cand['percentage']}%`")

            # Detailed Advice Tabs
            tab_sym, tab_treat, tab_prev, tab_care = st.tabs(["🔍 Symptoms", "💊 Treatment Steps", "🛡️ Prevention", "🌱 Growth & Care"])

            with tab_sym:
                st.markdown("##### Visible Symptoms:")
                for sym in info["symptoms"]:
                    st.write(f"• {sym}")

            with tab_treat:
                st.markdown("##### Recommended Practical Treatments:")
                for idx, t in enumerate(info["treatment"]):
                    st.markdown(f"""
                    <div class="treatment-step">
                        <strong>{idx+1}. {t['title']}</strong><br/>
                        <span style="font-size: 0.9rem; color: #cbd5e1;">{t['desc']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.caption("⚠️ *Treatment disclaimer: Consult local agricultural extension officers before chemical application.*")

            with tab_prev:
                st.markdown("##### Long-term Cultural Prevention:")
                for p in info["prevention"]:
                    st.write(f"✓ {p}")

            with tab_care:
                st.markdown("##### Optimal Agronomic Conditions:")
                care = info["growth_care"]
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.write(f"🌡️ **Temperature:** {care.get('temperature', 'N/A')}")
                    st.write(f"💧 **Water:** {care.get('water', 'N/A')}")
                with col_c2:
                    st.write(f"☀️ **Sunlight:** {care.get('sunlight', 'N/A')}")
                    st.write(f"🧪 **Soil pH:** {care.get('soil_ph', 'N/A')}")

        elif pil_img is None:
            st.info("Awaiting image upload on the left panel...")


def render_performance_tab():
    """Renders genuine, un-fabricated model performance metrics and confusion matrix."""
    st.header("📊 Model Performance & Mobile Efficiency")
    st.write(
        "CropGuardians AI adheres to strict academic and engineering integrity. "
        "Only genuine, measured evaluation metrics from independent test splits are displayed below."
    )

    report = get_cached_evaluation_report()

    if report is None:
        st.info(
            "ℹ️ No saved evaluation report found in `models/evaluation_metrics.json`.\n\n"
            "To generate genuine evaluation metrics on your local machine or server, execute:\n"
            "```bash\n"
            "python train_model.py --epochs 10 --batch-size 16\n"
            "```"
        )
        return

    # High-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{report.get('accuracy_pct', 0.0)}%")
    with col2:
        st.metric("Macro Precision", f"{report.get('macro_precision', 0.0)}")
    with col3:
        st.metric("Macro Recall", f"{report.get('macro_recall', 0.0)}")
    with col4:
        st.metric("Macro F1-Score", f"{report.get('macro_f1', 0.0)}")

    st.markdown("---")
    col_arch, col_cm = st.columns([1, 1], gap="large")

    with col_arch:
        st.subheader("Mobile Optimization Metrics")
        st.write(f"• **Architecture:** `{report.get('model_architecture', 'MobileNetV2')}`")
        st.write(f"• **Model File Size:** `{report.get('model_file_size_mb', 0.0)} MB`")
        st.write(f"• **Total Parameters:** `{report.get('total_parameters', 0):,}`")
        st.write(f"• **Trainable Parameters:** `{report.get('trainable_parameters', 0):,}`")
        st.write(f"• **Independent Test Samples:** `{report.get('total_test_samples', 0)}`")
        st.write(f"• **Target Resolution:** `{IMG_SIZE[0]}x{IMG_SIZE[1]} px`")

        st.markdown("##### Per-Class Performance Breakdown")
        per_class = report.get("per_class", {})
        if per_class:
            class_table = []
            for cname, metrics in per_class.items():
                short_name = cname.replace("___", " - ").replace("_", " ")
                class_table.append({
                    "Class": short_name,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1_score"],
                    "Test Samples": metrics["support"]
                })
            st.dataframe(class_table, use_container_width=True)

    with col_cm:
        st.subheader("Confusion Matrix")
        cm_path = Path("models/confusion_matrix.png")
        if cm_path.exists():
            st.image(str(cm_path), caption="Evaluated on independent test partition (Zero data leakage)", use_column_width=True)
        else:
            st.write("Confusion matrix plot not found at `models/confusion_matrix.png`.")


def render_about_tab():
    """Project documentation, background, and Zelvora Week 2 requirements."""
    st.header("📖 About CropGuardians AI")
    st.markdown("""
    ### Project Overview
    **CropGuardians AI** is an intelligent agricultural decision-support tool engineered for the **Zelvora Technologies AI Internship (Week 2)**.
    It leverages state-of-the-art Convolutional Neural Networks (CNN) with **MobileNetV2 transfer learning** to assist farmers, agronomists, and home growers in rapidly detecting crop diseases from leaf photographs.

    ### Core Deliverables Achieved
    1. **MobileNetV2 Transfer Learning CNN:** High accuracy, lightweight footprint (~7 MB), optimized input preprocessing, and `.keras` format.
    2. **Stratified Splitting & Leakage Prevention:** Perceptual file hashing to eliminate duplicate images across train/val/test splits.
    3. **Responsive Streamlit Cloud Interface:** Mobile-first responsive UI, instant image validation, confidence metrics, and clear warnings for low-confidence or non-leaf uploads.
    4. **Actionable Agricultural Guidance:** Verified symptoms, biological pathogen classifications, cultural prevention, and evidence-based treatment steps without unsupported chemical claims.

    ### System Architecture & Workflow
    ```
    +-------------------+      +-----------------------+      +------------------------+
    | Farmer / Browser  | ---> | Streamlit Server (UI) | ---> | Preprocessing Pipeline |
    +-------------------+      +-----------------------+      +------------------------+
                                                                          |
                                                                          v
    +-------------------+      +-----------------------+      +------------------------+
    | Verified Guidance | <--- | Structured Disease DB | <--- | MobileNetV2 Classifier |
    +-------------------+      +-----------------------+      +------------------------+
    ```
    """)


def render_dataset_tab():
    """Dataset information, class catalog, and responsible use limitations."""
    st.header("🔬 Dataset & Responsible-Use Limitations")
    st.markdown("""
    ### PlantVillage Dataset Source
    The training dataset is derived from the open-source **PlantVillage** benchmark (Hughes & Salathé, 2015), comprising thousands of curated crop leaf images categorized into healthy foliage and specific pathogen infections.

    ### Supported Classes (17 Classes across 5 Key Crops)
    """)
    supported = get_all_supported_classes()
    cols = st.columns(2)
    for idx, cname in enumerate(supported):
        info = get_disease_info(cname)
        with cols[idx % 2]:
            st.write(f"- {info['emoji']} **{info['crop']}**: {info['disease']} *({cname})*")

    st.markdown("""
    ---
    ### ⚠️ Real-World Limitations & Disclaimers
    - **Lab vs. Field Imagery:** PlantVillage images were captured under controlled laboratory lighting with uniform backgrounds. Real-world field images with harsh sunlight, shadows, or multiple overlapping leaves may yield lower prediction confidence.
    - **Non-Leaf Detection:** The classifier is trained specifically on crop foliage. Uploading images of pets, furniture, or unsupported plants may cause erroneous high-confidence predictions. The app includes low-confidence alerts to caution users.
    - **Pesticide & Chemical Advisory:** All treatment steps are cultural or general preventative recommendations. Never apply restricted agrochemicals without consulting your regional agricultural extension service or Krishi Vigyan Kendra (KVK).
    """)


def main():
    render_header()
    model, class_names = get_cached_model()
    selected_tab = render_sidebar()

    if selected_tab == "🌱 Disease Detector":
        render_detector_tab(model, class_names)
    elif selected_tab == "📊 Model Performance":
        render_performance_tab()
    elif selected_tab == "📖 About Project":
        render_about_tab()
    elif selected_tab == "🔬 Dataset & Limitations":
        render_dataset_tab()


if __name__ == "__main__":
    main()
