import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="Análisis Poblacional - Cajamarca",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Función para cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("poblacion_cajamarca_estandarizado.csv")
    
    # Estandarizar nombres de columnas
    df = df.rename(columns={
        'Departamento': 'Location',
        'Año': 'Time',
        'RangoEdad': 'AgeRange',
        'EdadInicio': 'AgeStart',
        'EdadFin': 'AgeEnd',
        'Sexo': 'Sex',
        'Población': 'Value'
    })
    
    # Definir grupos quinquenales exactos
    age_groups = [
        '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', 
        '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
        '60-64', '65-69', '70-74', '75-79', '80 y más'
    ]
    
    # Mapear AgeEnd a los grupos quinquenales
    age_mapping = {
        4: '0-4',
        9: '5-9',
        14: '10-14',
        19: '15-19',
        24: '20-24',
        29: '25-29',
        34: '30-34',
        39: '35-39',
        44: '40-44',
        49: '45-49',
        54: '50-54',
        59: '55-59',
        64: '60-64',
        69: '65-69',
        74: '70-74',
        79: '75-79',
        120: '80 y más'
    }
    
    df['AgeGroup'] = df['AgeEnd'].map(age_mapping)
    df['AgeGroup'] = pd.Categorical(df['AgeGroup'], categories=age_groups, ordered=True)
    
    return df

# Cargar datos
df = load_data()
years = sorted(df['Time'].unique())

# Función para generar pirámide poblacional quinquenal
def generate_population_pyramid(data, year):
    df_year = data[data['Time'] == year]
    
    # Preparar datos manteniendo el orden quinquenal
    df_male = df_year[df_year['Sex'] == 'Hombre'].groupby('AgeGroup', observed=True)['Value'].sum().reset_index()
    df_female = df_year[df_year['Sex'] == 'Mujer'].groupby('AgeGroup', observed=True)['Value'].sum().reset_index()
    
    # Crear figura
    fig = go.Figure()
    
    # Barras para hombres (valores positivos)
    fig.add_trace(go.Bar(
        y=df_male['AgeGroup'],
        x=df_male['Value'],
        name='Hombres',
        orientation='h',
        marker_color='#1f77b4',
        hovertemplate='%{y}: %{x:,} hombres<extra></extra>',
        text=df_male['Value'],
        texttemplate='%{text:,}',
        textposition='inside'
    ))
    
    # Barras para mujeres (valores negativos)
    fig.add_trace(go.Bar(
        y=df_female['AgeGroup'],
        x=-df_female['Value'],
        name='Mujeres',
        orientation='h',
        marker_color='#ff7f0e',
        hovertemplate='%{y}: %{x:,} mujeres<extra></extra>',
        text=df_female['Value'],
        texttemplate='%{text:,}',
        textposition='inside'
    ))
    
    # Valores fijos para el eje x
    tickvals = [-90000, -70000, -50000, -30000, -10000, 0, 10000, 30000, 50000, 70000, 90000]
    ticktext = ['90k', '70k', '50k', '30k', '10k', '0', '10k', '30k', '50k', '70k', '90k']
    
    # Actualizar diseño
    fig.update_layout(
        title=f'Pirámide Poblacional Quinquenal de Cajamarca - {year}',
        barmode='relative',
        bargap=0.1,
        xaxis_title="Población",
        yaxis_title="Grupo Quinquenal de Edad",
        xaxis=dict(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            range=[-100000, 100000]  # Rango fijo para asegurar que todos los valores sean visibles
        ),
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Función para gráfico de evolución quinquenal
def generate_population_trend(data):
    df_trend = data.groupby(['Time', 'AgeGroup'], observed=True)['Value'].sum().unstack()
    
    fig = px.area(
        df_trend,
        x=df_trend.index,
        y=df_trend.columns,
        title="Evolución Poblacional por Grupos Quinquenales (1981-2017)",
        labels={'value': 'Población', 'Time': 'Año', 'variable': 'Grupo de Edad'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            title='Grupos Quinquenales',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    return fig

# Función para gráfico de comparación quinquenal
def generate_comparison_chart(data, year1, year2):
    df_year1 = data[data['Time'] == year1].groupby('AgeGroup', observed=True)['Value'].sum().reset_index()
    df_year2 = data[data['Time'] == year2].groupby('AgeGroup', observed=True)['Value'].sum().reset_index()
    
    df_compare = pd.merge(df_year1, df_year2, on='AgeGroup', suffixes=(f'_{year1}', f'_{year2}'))
    df_compare['Change'] = (df_compare[f'Value_{year2}'] - df_compare[f'Value_{year1}']) / df_compare[f'Value_{year1}'] * 100
    
    fig = px.bar(
        df_compare,
        x='AgeGroup',
        y='Change',
        title=f"Cambio Poblacional por Grupo Quinquenal ({year1} vs {year2})",
        labels={'Change': 'Cambio (%)', 'AgeGroup': 'Grupo Quinquenal'},
        color='Change',
        color_continuous_scale='RdYlGn',
        range_color=[-50, 50],
        text=df_compare['Change'].apply(lambda x: f"{x:.1f}%")
    )
    
    fig.update_layout(
        yaxis_title="Cambio Porcentual (%)",
        coloraxis_colorbar=dict(title="Cambio %"),
        xaxis={'categoryorder': 'array', 'categoryarray': df['AgeGroup'].cat.categories},
        height=600
    )
    
    fig.update_traces(
        textposition='outside',
        textfont_size=12
    )
    
    return fig

# Interfaz principal
st.title("Análisis Demográfico Quinquenal del Departamento de Cajamarca")
st.markdown("""
Visualización interactiva de la estructura poblacional quinquenal de Cajamarca entre 1981 y 2017.
Los datos muestran la distribución exacta por grupos de 5 años y sexo.
""")

# Sidebar con controles
with st.sidebar:
    st.header("Controles")
    selected_year = st.selectbox(
        "Seleccione año para pirámide:",
        options=years,
        index=len(years)-1
    )
    
    st.markdown("---")
    st.header("Comparación entre años")
    year1, year2 = st.select_slider(
        "Seleccione rango para comparación:",
        options=years,
        value=(years[0], years[-1])
    )
    st.markdown("---")
    st.markdown(f"""
    **Datos disponibles:**  
    Años: {years[0]} - {years[-1]}  
    Grupos quinquenales: 0-4 hasta 80 y más  
    Última actualización: {datetime.now().strftime('%Y-%m-%d')}
    """)

# Mostrar métricas principales
col1, col2, col3 = st.columns(3)
with col1:
    total_pop = df[df['Time'] == selected_year]['Value'].sum()
    st.metric("Población Total", f"{total_pop:,}")
with col2:
    male_pop = df[(df['Time'] == selected_year) & (df['Sex'] == 'Hombre')]['Value'].sum()
    st.metric("Población Masculina", f"{male_pop:,}")
with col3:
    female_pop = df[(df['Time'] == selected_year) & (df['Sex'] == 'Mujer')]['Value'].sum()
    st.metric("Población Femenina", f"{female_pop:,}")

# Gráficos principales
tab1, tab2, tab3 = st.tabs(["Pirámide Poblacional", "Evolución Temporal", "Comparación"])

with tab1:
    st.plotly_chart(
        generate_population_pyramid(df, selected_year),
        use_container_width=True
    )

with tab2:
    st.plotly_chart(
        generate_population_trend(df),
        use_container_width=True
    )

with tab3:
    if year1 == year2:
        st.warning("Seleccione dos años diferentes para comparar")
    else:
        st.plotly_chart(
            generate_comparison_chart(df, year1, year2),
            use_container_width=True
        )

# Datos detallados
with st.expander("Ver datos detallados por grupo quinquenal"):
    st.dataframe(
        df[df['Time'] == selected_year].sort_values(['AgeStart', 'Sex']),
        hide_index=True,
        column_config={
            "Location": None,
            "AgeStart": st.column_config.NumberColumn("Edad Inicio"),
            "AgeEnd": st.column_config.NumberColumn("Edad Fin"),
            "AgeGroup": "Grupo Quinquenal",
            "Value": st.column_config.NumberColumn("Población", format="%,d")
        },
        height=600
    )

# Notas al pie
st.caption("""
**Fuente:** INEI, *Censos Nacionales*.  
**Nota:** Grupos etarios en intervalos quinquenales.  
[inei.gob.pe/estadisticas/censos](https://www.inei.gob.pe/estadisticas/censos/)
""")