import streamlit as st
import pickle
import numpy as np

# Configuración de la página con un título e ícono de vino
st.set_page_config(page_title="WINe", page_icon="🍷", layout="centered")

st.title("WINe: La herramienta perfecta para tu bodega 🍷")
st.markdown("""
Bienvenido a **WINe**. La solución integral que llevará tus vinos al siguiente nivel. Ajusta las siguientes características del vino y descubre su calidad:
- **Baja**: calidad entre 3 y 4
- **Media**: calidad entre 5 y 6
- **Alta**: calidad entre 7 y 8

¡Salud y disfruta de un gran vino!
""")

# Función para cargar el modelo (cache para no recargar en cada interacción)
@st.cache_resource
def load_model():
    with open('finalmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

rf_model = load_model()

# Controles interactivos (barra lateral) para ajustar las variables predictoras
st.sidebar.header("Componentes del Vino")
alcohol = st.sidebar.slider("Alcohol (%)", min_value=5.0, max_value=15.0, value=10.0, step=0.1)
volatile_acidity = st.sidebar.slider("Acidez Volátil (g/L)", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
sulphates = st.sidebar.slider("Sulfatos (g/L)", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
citric_acid = st.sidebar.slider("Ácido Cítrico (g/L)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

# Crear el array de entrada para el modelo
input_data = np.array([[alcohol, volatile_acidity, sulphates, citric_acid]])

# Botón para predecir
if st.button("Obtener la Calidad del Vino"):
    pred_numeric = rf_model.predict(input_data)[0]
    # Mapeo: 0 → Baja, 1 → Media, 2 → Alta
    quality_map = {0: "Baja", 1: "Media", 2: "Alta"}
    quality_pred = quality_map[pred_numeric]
    
    st.success(f"La calidad del vino es la siguiente: **{quality_pred}**")
    
    # Recomendaciones según la calidad
    if quality_pred == "Alta":
        st.info("¡Excelente elección! Este vino es de alta calidad, listo para ser disfrutado 🍷")
    elif quality_pred == "Media":
        st.info("Un vino de calidad media, la realización de ciertos ajustes podría darle un salto de calidad 🍇")
    else:
        st.info("Este vino es de calidad baja. Considere mejorar algunos aspectos en la producción para obtener un vino de mayor calidad. 🚫")
