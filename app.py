import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chest X-Ray AI Classifier",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASSES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]

MODEL_PATH = "model/best_densenet121_finetuned.pth"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# IMPORTANT:
# This MUST match the preprocessing used during training.
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    # Create DenseNet121 architecture
    model = models.densenet121(
        weights=None
    )

    # DenseNet121 classifier input features
    num_features = model.classifier.in_features

    # EXACT classifier used during training
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(
            num_features,
            len(CLASSES)
        )
    )

    # Load trained weights
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    # Move model to device
    model = model.to(DEVICE)

    # Evaluation mode
    model.eval()

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Failed to load the trained model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🩻 About the Model")

    st.write(
        """
        This application uses a fine-tuned
        **DenseNet121** model for chest X-ray
        image classification.
        """
    )

    st.divider()

    st.subheader("Classes")

    for class_name in CLASSES:
        st.write(f"• {class_name}")

    st.divider()

    st.subheader("Model Information")

    st.write("Architecture: DenseNet121")
    st.write("Input Size: 224 × 224")
    st.write("Pretrained: ImageNet")
    st.write(f"Device: {DEVICE}")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🩻 Chest X-Ray AI Classifier")

st.markdown(
    """
    ### Upload a chest X-ray
    The fine-tuned DenseNet121 model will analyze the image
    and provide predictions across four classes.
    """
)

st.warning(
    "⚠️ This is an experimental machine-learning project "
    "and is **not a medical diagnostic tool**. "
    "Do not use its predictions for medical decisions."
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Upload a PNG, JPG or JPEG chest X-ray image."
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # Load image
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.divider()

        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.subheader("Uploaded X-Ray")

            st.image(
                image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        input_tensor = transform(image)

        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        # Move to GPU/CPU
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

        # ----------------------------------------------------
        # GET PREDICTION
        # ----------------------------------------------------

        predicted_index = torch.argmax(
            probabilities
        ).item()

        predicted_class = CLASSES[
            predicted_index
        ]

        confidence = (
            probabilities[
                predicted_index
            ].item() * 100
        )

        # ----------------------------------------------------
        # PROBABILITY DATAFRAME
        # ----------------------------------------------------

        probability_data = pd.DataFrame({
            "Class": CLASSES,
            "Probability": [
                p.item() * 100
                for p in probabilities
            ]
        })

        probability_data = (
            probability_data
            .sort_values(
                "Probability",
                ascending=False
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with col2:

            st.subheader("Prediction")

            st.success(
                f"Prediction: {predicted_class}"
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.write("")

            st.subheader(
                "Class Probabilities"
            )

            # Display dataframe
            display_df = probability_data.copy()

            display_df["Probability"] = (
                display_df["Probability"]
                .map(lambda x: f"{x:.2f}%")
            )

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True
            )

        # ----------------------------------------------------
        # PROBABILITY CHART
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction Probability Distribution"
        )

        chart_data = probability_data.copy()

        chart_data = chart_data.set_index(
            "Class"
        )

        st.bar_chart(
            chart_data[
                ["Probability"]
            ]
        )

        # ----------------------------------------------------
        # TOP PREDICTION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Model Interpretation"
        )

        st.write(
            f"""
            The model assigned the highest probability
            to **{predicted_class}** with a confidence of
            **{confidence:.2f}%**.
            """
        )

        st.caption(
            "The displayed probabilities represent the model's "
            "output distribution and should not be interpreted "
            "as clinical probabilities."
        )

    except Exception as e:

        st.error(
            "❌ Unable to process this image."
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DenseNet121 • ImageNet Transfer Learning • Fine-Tuned "
    "on COVID-19 Radiography Dataset"
)

st.caption(
    "For educational and research purposes only."
)