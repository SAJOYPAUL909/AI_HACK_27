import streamlit as st
import requests, io, pandas as pd, os, plotly.express as px

API = os.getenv("API_BASE", "http://localhost:5000")

st.set_page_config(page_title="Energy Consumption Optimization Advisor", layout="wide")
st.title("Energy Consumption Optimization Advisor — Full MVP")
st.markdown("Supports two data types: household (appliance-level) and industrial (multi-energy). No external APIs required by default.")

dtype = st.radio("Select data type", ["household", "industrial"])

st.markdown("You can upload a CSV in the selected format, or generate a synthetic sample and download it.")

if st.button("Show packaged sample CSV"):
    sample = f"data/generated_{dtype}_sample.csv"
    if os.path.exists(sample):
        df = pd.read_csv(sample)
        st.download_button("Download synthetic sample", df.to_csv(index=False).encode("utf-8"), file_name=os.path.basename(sample), mime="text/csv")
        st.dataframe(df.head())
    else:
        st.error("Sample not found in packaged data folder.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
    with st.spinner("Uploading..."):
        r = requests.post(f"{API}/upload", files=files, data={"data_type": dtype})
    if not r.ok:
        st.error("Upload failed: " + r.text)
    else:
        j = r.json()
        if not j.get("ok"):
            st.error(j.get("error", "unknown"))
        else:
            cleaned = j["cleaned_filename"]
            st.success("Uploaded and cleaned: " + cleaned)
            preview = pd.DataFrame(j["preview"])
            st.dataframe(preview.head(50))

            if st.button("Download cleaned CSV"):
                rr = requests.get(f"{API}/download/{cleaned}")
                if rr.ok:
                    st.download_button("Download cleaned file", rr.content, file_name=cleaned, mime="text/csv")
                else:
                    st.error("Download failed.")

            st.subheader("Visualize cleaned data")
            rr = requests.get(f"{API}/download/{cleaned}")
            if rr.ok:
                dfc = pd.read_csv(io.StringIO(rr.content.decode("utf-8")), parse_dates=["timestamp"])
                if dtype == "household":
                    if "appliance" in dfc.columns:
                        agg = dfc.groupby(["timestamp", "appliance"])["energy_kwh"].sum().reset_index()
                        fig = px.line(agg, x="timestamp", y="energy_kwh", color="appliance", title="Appliance-level consumption")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.line(dfc, x="timestamp", y="energy_kwh", title="Household consumption")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    if "energy_type" in dfc.columns:
                        agg = dfc.groupby(["timestamp", "energy_type"])["energy_value"].sum().reset_index()
                        fig = px.line(agg, x="timestamp", y="energy_value", color="energy_type", title="Industrial energy types")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        col = "energy_value" if "energy_value" in dfc.columns else dfc.columns[1]
                        fig = px.line(dfc, x="timestamp", y=col, title="Industrial consumption")
                        st.plotly_chart(fig, use_container_width=True)

                if st.button("Detect anomalies"):
                    with st.spinner("Detecting anomalies..."):
                        r2 = requests.post(f"{API}/anomalies", json={"cleaned_filename": cleaned})
                    if r2.ok:
                        j2 = r2.json()
                        anoms = j2.get("anomalies", [])
                        st.write("Anomalies found:", len(anoms))
                        if len(anoms) > 0:
                            st.dataframe(pd.DataFrame(anoms))
                    else:
                        st.error("Anomaly API error")

                st.subheader("Forecast & recommendations")
                horizon = st.slider("Forecast horizon (hours)", 6, 168, 24)
                if st.button("Run forecast"):
                    with st.spinner("Forecasting..."):
                        r3 = requests.post(f"{API}/forecast", json={"cleaned_filename": cleaned, "horizon": horizon})
                    if r3.ok:
                        j3 = r3.json()
                        st.json(j3)
                    else:
                        st.error("Forecast API error")

                price = st.number_input("Price per kWh (USD)", value=0.15, step=0.01)
                context = st.text_input("Context (optional): e.g. 'residential, EV owner'")

                if st.button("Get recommendations"):
                    with st.spinner("Generating recommendations..."):
                        r4 = requests.post(f"{API}/recommend", json={"cleaned_filename": cleaned, "price_per_kwh": price, "horizon": horizon, "context": context})
                    if r4.ok:
                        j4 = r4.json()
                        if "recommendations" in j4:
                            st.json(j4["recommendations"])
                        else:
                            st.text(j4.get("recommendations_text", "No response"))
                    else:
                        st.error("Recommendation API error")
