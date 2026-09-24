import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import matplotlib.cm as cm

IMG_SIZE    = (512, 512)
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
MODEL_PATH  = "ordinal_model_final_Etapa3_2.keras"
LAST_CONV_LAYER = "top_conv"

st.set_page_config(page_title="Clasificare Retinopatie Diabetica", layout="wide")

@st.cache_resource
def load_dr_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_dr_model()
except Exception as e:
    st.error(f"Eroare la incarcarea modelului: {e}")
    st.stop()

def aplica_auto_crop_patrat(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20:
            center_x, center_y = x + w // 2, y + h // 2
            half = max(w, h) // 2
            x0 = max(0, center_x - half)
            y0 = max(0, center_y - half)
            x1 = min(img.shape[1], center_x + half)
            y1 = min(img.shape[0], center_y + half)
            crop = img[y0:y1, x0:x1]
            if crop.size > 0:
                return crop
    return img

def aplica_clahe_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_eq = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def aplica_masca_circulara(img):
    mask = np.zeros(IMG_SIZE[:2], dtype=np.uint8)
    cv2.circle(mask, (IMG_SIZE[0] // 2, IMG_SIZE[1] // 2), IMG_SIZE[0] // 2, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def preproceseaza_imagine(image_pil):
    img = np.array(image_pil.convert('RGB'))
    img = aplica_auto_crop_patrat(img)
    if img.size == 0:
        img = np.array(image_pil.convert('RGB'))
    img = cv2.resize(img, IMG_SIZE)
    img = aplica_clahe_lab(img)
    img = aplica_masca_circulara(img)
    tensor = preprocess_input(np.expand_dims(img.astype(np.float32), axis=0))
    return img, tensor

def decodeaza_ordinal(probabilitati):
    stadiu = 0
    for prob in probabilitati:
        if prob >= 0.5:
            stadiu += 1
        else:
            break
    return stadiu

def gradcam_global(tensor, model, conv_layer):
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tensor)
        scor = tf.reduce_sum(preds[0])
    grads = tape.gradient(scor, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    return (tf.maximum(heatmap, 0) / max_val).numpy()

def gradcam_neuron(tensor, model, conv_layer, index_neuron):
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tensor)
        scor = preds[0][index_neuron]
    grads = tape.gradient(scor, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    return (tf.maximum(heatmap, 0) / max_val).numpy()

def suprapune_heatmap(img_rgb, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    colored = np.uint8(255 * jet_colors[heatmap_uint8])
    return cv2.addWeighted(img_rgb, 1 - alpha, colored, alpha, 0)

# Interfata
st.title("Clasificarea Retinopatiei Diabetice - Model Ordinal")
st.write(
    "Aplicatie demonstrativa bazata pe arhitectura EfficientNet-B3 cu regresie ordinala (4 neuroni sigmoid). "
    "Pre-procesarea aplicata este identica cu cea folosita la antrenare: "
    "crop automat, CLAHE pe canalul L, masca circulara."
)

uploaded_file = st.file_uploader("Incarca o imagine fundoscopica (JPG sau PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imaginea originala")
        st.image(image_pil, use_column_width=True)

    if st.button("Analizeaza imaginea"):
        with st.spinner("Reteaua proceseaza imaginea..."):
            img_procesat, tensor = preproceseaza_imagine(image_pil)
            probabilitati = model.predict(tensor, verbose=0)[0]
            stadiu = decodeaza_ordinal(probabilitati)
            diagnostic = CLASS_NAMES[stadiu]
            heatmap_global = gradcam_global(tensor, model, LAST_CONV_LAYER)
            overlay_global = suprapune_heatmap(img_procesat, heatmap_global)

        with col2:
            st.subheader(f"Diagnostic: {diagnostic}")
            st.image(img_procesat, caption="Imagine dupa pre-procesare CLAHE", use_column_width=True)
            st.image(overlay_global, caption="Grad-CAM global - zonele de decizie ale modelului", use_column_width=True)

        st.markdown("---")
        st.subheader("Probabilitatile celor 4 neuroni ordinali")

        etichete_neuroni = [
            "q1 - cel putin Mild?",
            "q2 - cel putin Moderate?",
            "q3 - cel putin Severe?",
            "q4 - este Proliferative?"
        ]

        cols_prob = st.columns(4)
        for i, (eticheta, prob) in enumerate(zip(etichete_neuroni, probabilitati)):
            with cols_prob[i]:
                raspuns = "DA" if prob >= 0.5 else "NU"
                st.write(f"**{eticheta}**")
                st.write(f"{raspuns} ({prob * 100:.1f}%)")
                st.progress(float(prob))

        st.markdown("---")
        st.subheader("Harti Grad-CAM per neuron ordinal")
        st.write(
            "Fiecare harta arata ce zone ale retinei au influentat "
            "raspunsul neuronului respectiv."
        )

        cols_cam = st.columns(4)
        for i in range(4):
            heatmap_n = gradcam_neuron(tensor, model, LAST_CONV_LAYER, i)
            overlay_n = suprapune_heatmap(img_procesat, heatmap_n)
            prob = probabilitati[i]
            raspuns = "DA" if prob >= 0.5 else "NU"
            with cols_cam[i]:
                st.write(f"**{etichete_neuroni[i]}**")
                st.write(f"prob={prob:.3f} [{raspuns}]")
                st.image(overlay_n, use_column_width=True)

        st.markdown("---")
        mesaje = [
            "Pacientul nu prezinta semne de retinopatie diabetica.",
            "Retinopatie diabetica usoara (Mild) - monitorizare recomandata.",
            "Retinopatie diabetica moderata (Moderate) - consult oftalmologic recomandat.",
            "Retinopatie diabetica severa (Severe) - consult oftalmologic urgent.",
            "Retinopatie diabetica proliferativa - interventie medicala urgenta."
        ]
        st.write(f"**Concluzie:** {mesaje[stadiu]}")