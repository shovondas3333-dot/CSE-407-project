import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Smart Energy Consumption Dashboard")

st.write("""
## Dataset Brief
This dashboard visualizes an energy dataset, including timestamped records of wattage, current, voltage, and cumulative kWh usage for energy monitoring.
""")

uploaded_file = st.file_uploader("Upload your energy dataset CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Raw Data Sample")
    st.dataframe(df.head())

    # Date handling
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        st.write("Data range:", df['Timestamp'].min(), "to", df['Timestamp'].max())

    # Basic stats
    st.write("### Summary Statistics")
    st.write(df.describe())

    # Time-series plot selection
    option = st.selectbox("Select column to plot over time:",
                          ('Watt', 'Current', 'Voltage', 'kWh'))

    if 'Timestamp' in df.columns and option in df.columns:
        st.write(f"#### {option} Over Time")
        fig, ax = plt.subplots()
        ax.plot(df['Timestamp'], df[option], label=option)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel(option)
        plt.xticks(rotation=45)
        ax.legend()
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")

