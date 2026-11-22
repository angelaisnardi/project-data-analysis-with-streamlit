import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import fetch_california_housing

st.title('Análisis Interactivo de Viviendas en California')
st.write('Datos del censo de 1990 de California, a nivel de grupo de bloques.')
st.markdown('---')

# ===== FASE 1: Carga y Preparación de Datos =====
st.header('1. Carga y Preparación de Datos')

# Cargar el dataset
datos = fetch_california_housing()

# DataFrame con las características y sus nombres
df_california = pd.DataFrame(datos.data, columns=datos.feature_names)

# Agregar la variable objetivo como una columna más
df_california['MedHouseVal'] = datos.target

# Verificar valores faltantes
faltantes = df_california.isnull().sum()

if faltantes.sum() == 0:
    st.success('El dataset no tiene valores faltantes.')
else:
    st.warning('El dataset tiene valores faltantes:')
    st.write(faltantes[faltantes > 0])

# Primeras 5 filas
st.markdown('**Primeras 5 filas**')
st.dataframe(df_california.head())

# Tipo de dato de cada columna
st.markdown('**Tipo de dato de cada columna**')
st.write(df_california.dtypes.astype(str))

st.markdown('---')

# ===== FASE 2: Análisis Descriptivo Interactivo =====

# Barra lateral: título y descripción
st.sidebar.markdown('## Filtros')
st.sidebar.markdown('Elegí qué viviendas analizar. Todo se actualiza automáticamente.')

# Filtro 1: edad de la vivienda (slider de rango, del mínimo al máximo)
edad_min = int(df_california['HouseAge'].min())
edad_max = int(df_california['HouseAge'].max())

rango_edad = st.sidebar.slider('Edad mediana de la casa (HouseAge)',
                               min_value=edad_min,
                               max_value=edad_max,
                               value=(edad_min, edad_max),
                               step=1)

# Filtro 2: latitud mínima
latitud_minima = st.sidebar.number_input('Latitud mínima (Latitude)',
                                         min_value=float(df_california['Latitude'].min()),
                                         max_value=float(df_california['Latitude'].max()),
                                         value=float(df_california['Latitude'].min()),
                                         step=0.5)

# Aplicar los filtros
df_filtrado = df_california[(df_california['HouseAge'] >= rango_edad[0]) &
                            (df_california['HouseAge'] <= rango_edad[1]) &
                            (df_california['Latitude'] >= latitud_minima)]

st.sidebar.info(f'Grupos de bloques seleccionados: {len(df_filtrado)}')

# Resumen descriptivo
st.header('2. Resumen Descriptivo')

if df_filtrado.empty:
    st.warning('Ninguna vivienda cumple los filtros. Probá ampliar el rango.')
else:
    mediana_valor = df_filtrado['MedHouseVal'].median()
    rango_valor = df_filtrado['MedHouseVal'].max() - df_filtrado['MedHouseVal'].min()

    resumen = pd.DataFrame({
        'Medida': ['Grupos de bloques', 'Mediana de MedHouseVal', 'Rango de MedHouseVal (máx - mín)'],
        'Valor': [str(len(df_filtrado)), f'{mediana_valor:.2f}', f'{rango_valor:.2f}']
    })
    st.table(resumen)
    st.info('MedHouseVal está en cientos de miles de dólares (1.00 = USD 100.000).')
    st.markdown('---')

    # ===== FASE 3: Visualización Dinámica =====
    st.header('3. Visualizaciones')

    # Histograma con Matplotlib
    st.subheader('Distribución del valor de la vivienda')

    fig, ax = plt.subplots()
    ax.hist(df_filtrado['MedHouseVal'], bins=30, color='skyblue', edgecolor='black')
    ax.axvline(mediana_valor, color='red', linestyle='--',
               label=f'Mediana = {mediana_valor:.2f}')
    ax.set_title('Distribución de MedHouseVal (datos filtrados)')
    ax.set_xlabel('Valor mediano de la vivienda (cientos de miles de USD)')
    ax.set_ylabel('Cantidad de grupos de bloques')
    ax.legend()
    st.pyplot(fig)

    # Gráfico de dispersión con Plotly
    st.subheader('Relación entre ingreso y valor de la vivienda')

    fig_scatter = px.scatter(df_filtrado,
                             x='MedInc',
                             y='MedHouseVal',
                             opacity=0.4,
                             title='Ingreso mediano vs. valor mediano de la vivienda',
                             labels={'MedInc': 'Ingreso mediano (decenas de miles de USD)',
                                     'MedHouseVal': 'Valor de la vivienda (cientos de miles de USD)'})
    st.plotly_chart(fig_scatter)

    correlacion = df_filtrado['MedInc'].corr(df_filtrado['MedHouseVal'])
    st.write(f'Correlación entre ingreso y valor de la vivienda: **{correlacion:.2f}**')

    # Mapa geográfico (opcional)
    st.subheader('Mapa de las viviendas filtradas')
    st.map(df_filtrado, latitude='Latitude', longitude='Longitude')