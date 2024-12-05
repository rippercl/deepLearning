import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def home():
    st.title("Página Principal")
    st.write("Bienvenido a la página principal de la aplicación.")
    st.title("Curso DeepLearning IPS-UAI")
    st.write("Aplicacion web de implementacion de Streamlit")
    st.write("Alumnos: Cesar Neculman / Chrsitian Fuentes")

def ConArchivo():
    st.title("Implementación de carga de datos")
    st.write("Esta pagina permite cargar un archivo con formato csv para poder realizar predicciones.")
    # Título de la aplicación
    

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    if uploaded_file is not None:
        # Leer el archivo CSV
        df = pd.read_csv(uploaded_file)
        
        # Mostrar el DataFrame
        st.write("Datos del CSV:")
        st.dataframe(df)

        # Mostrar estadísticas descriptivas
        st.write("Estadísticas Descriptivas:")
        st.write(df.describe())

        # Crear un gráfico simple (histograma)
        st.write("Histograma de una Columna:")
        column_to_plot = st.selectbox("Selecciona la columna para el histograma", df.columns)
        st.bar_chart(df[column_to_plot].value_counts())
    else:
        st.write("Por favor, sube un archivo CSV para empezar.")
    
    
    

def ConCaracter():
    st.title("Contacto")
    st.write("Esta es la página de Contacto. Aquí puedes añadir información de contacto.")
    # Título de la aplicación
    st.title("Aplicación de Predicción con KNN")

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    if uploaded_file is not None:
        # Leer el archivo CSV
        df = pd.read_csv(uploaded_file)
        
        # Mostrar el DataFrame
        st.write("Datos del CSV:")
        st.dataframe(df)

        # Selección de características y etiqueta
        features = st.multiselect("Selecciona las características", df.columns)
        label = st.selectbox("Selecciona la etiqueta", df.columns)
        
        if features and label:
            # Codificar etiquetas si es necesario
            if df[label].dtype == 'object':
                le = LabelEncoder()
                df[label] = le.fit_transform(df[label])
            
            X = df[features]
            y = df[label]

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Crear y entrenar el modelo KNN
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)

            # Hacer predicciones con el conjunto de prueba
            y_pred = knn.predict(X_test)

            # Mostrar la precisión del modelo
            accuracy = (y_test == y_pred).mean()
            st.write(f'Precisión del modelo: {accuracy:.2f}')

            # Hacer predicciones con nuevos datos
            st.write("Predicción con nuevos datos:")
            new_data = {feature: st.number_input(f"Ingrese {feature}", value=0) for feature in features}
            new_data_df = pd.DataFrame([new_data])
            new_pred = knn.predict(new_data_df)
            
            if df[label].dtype == 'object':
                new_pred = le.inverse_transform(new_pred)
            
            st.write(f'Predicción para los datos ingresados: {new_pred[0]}')

        else:
            st.write("Por favor, sube un archivo CSV para empezar.")
    


# Diccionario para mapear las páginas
pages = {
    "Home": home,
    "Uso con Archivo": ConArchivo,
    "Uso Con Archivo y caracteriscita": ConCaracter
}

# Menú de selección para cambiar entre las páginas
st.sidebar.title("Navegación")
selection = st.sidebar.radio("Ir a", list(pages.keys()))

# Mostrar la página seleccionada
page = pages[selection]
page()
