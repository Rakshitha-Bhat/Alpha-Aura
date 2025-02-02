pip install streamlit plotly folium streamlit-folium pandas numpy torch

import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import zipfile
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader

# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AISDataset(Dataset):
    def _init_(self, features, labels):
        self.features = torch.FloatTensor(features).to(device)
        self.labels = torch.FloatTensor(labels).to(device)
        
    def _len_(self):
        return len(self.features)
    
    def _getitem_(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(uploaded_file):
    """Load and process the uploaded AIS data"""
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file) as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                dfs = []
                for file in csv_files:
                    with zip_ref.open(file) as f:
                        df = pd.read_csv(f)
                        dfs.append(df)
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    return df
        else:
            df = pd.read_csv(uploaded_file)
            return df
    return None

def preprocess_data(df):
    """Preprocess data and move to GPU if available"""
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert coordinates to torch tensors for GPU processing
    if all(col in df.columns for col in ['latitude', 'longitude']):
        coords = torch.tensor(df[['latitude', 'longitude']].values, device=device)
        return df, coords
    return df, None

def create_map(df):
    """Create an interactive map with vessel positions"""
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add vessel markers with clustering
    marker_cluster = folium.MarkerCluster()
    
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=f"MMSI: {row['mmsi']}<br>Time: {row['timestamp']}",
            color='blue',
            fill=True
        ).add_to(marker_cluster)
    
    marker_cluster.add_to(m)
    return m

def calculate_vessel_density(coords, resolution=100):
    """Calculate vessel density using GPU acceleration"""
    if coords is None:
        return None
    
    x_min, x_max = coords[:, 1].min().item(), coords[:, 1].max().item()
    y_min, y_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    
    x = torch.linspace(x_min, x_max, resolution, device=device)
    y = torch.linspace(y_min, y_max, resolution, device=device)
    
    xx, yy = torch.meshgrid(x, y)
    grid_points = torch.stack([yy.flatten(), xx.flatten()], dim=1)
    
    density = torch.zeros(resolution * resolution, device=device)
    
    # Calculate density using GPU
    batch_size = 1000
    for i in range(0, len(grid_points), batch_size):
        batch = grid_points[i:i+batch_size]
        for coord in coords:
            dist = torch.norm(batch - coord, dim=1)
            density[i:i+batch_size] += torch.exp(-dist / 0.1)
    
    return density.reshape(resolution, resolution).cpu().numpy()

def main():
    st.title('GPU-Accelerated AIS Vessel Data Analysis Dashboard')
    
    # File uploader
    uploaded_file = st.file_uploader("Upload AIS data (CSV or ZIP)", type=['csv', 'zip'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            df, coords = preprocess_data(df)
            
            # Display basic statistics
            st.header('Dataset Overview')
            st.write(f"Total records: {len(df)}")
            st.write(f"Unique vessels (MMSI): {df['mmsi'].nunique()}")
            st.write(f"Running on: {device}")
            
            # Time range selector
            st.header('Time Range Filter')
            if 'timestamp' in df.columns:
                min_date = df['timestamp'].min()
                max_date = df['timestamp'].max()
                
                date_range = st.date_input(
                    "Select date range",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
                    filtered_df = df.loc[mask]
                    filtered_coords = coords[mask] if coords is not None else None
                    
                    # Display interactive map
                    st.header('Vessel Positions Map')
                    m = create_map(filtered_df)
                    folium_static(m)
                    
                    # GPU-accelerated vessel density heatmap
                    st.header('GPU-Accelerated Vessel Density Analysis')
                    if filtered_coords is not None:
                        density = calculate_vessel_density(filtered_coords)
                        fig = px.imshow(density, 
                                      title='Vessel Density Heatmap (GPU-Accelerated)',
                                      labels={'color': 'Density'})
                        st.plotly_chart(fig)
                    
                    # Vessel movement over time
                    st.header('Vessel Movement Timeline')
                    timeline_fig = px.scatter(
                        filtered_df,
                        x='timestamp',
                        y='mmsi',
                        title='Vessel Positions Over Time'
                    )
                    st.plotly_chart(timeline_fig)
                    
                    # MMSI Distribution
                    st.header('MMSI Distribution Analysis')
                    mmsi_counts = filtered_df['mmsi'].value_counts()
                    fig = px.bar(x=mmsi_counts.index, 
                                y=mmsi_counts.values,
                                title='Vessel Frequency Distribution')
                    st.plotly_chart(fig)
                    
                    # Data table with pagination
                    st.header('Raw Data')
                    page_size = st.number_input('Rows per page', min_value=10, value=50)
                    page = st.number_input('Page', min_value=1, value=1)
                    start_idx = (page - 1) * page_size
                    end_idx = start_idx + page_size
                    st.dataframe(filtered_df.iloc[start_idx:end_idx])
                    
            else:
                st.error("The uploaded data doesn't contain the expected 'timestamp' column")
    else:
        st.info("Please upload an AIS data file (CSV or ZIP) to begin analysis")

if _name_ == "_main_":
    main()

streamlit run app.py 
