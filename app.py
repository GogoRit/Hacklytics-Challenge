import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ✅ Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("❌ Gemini API key not found. Please check your .env file.")

# ✅ Streamlit UI
st.set_page_config(layout="wide", page_title="Multi-Agent AI System")

# ✅ Load Datasets
@st.cache_data
def load_data():
    financial_df = pd.read_csv('/Users/gaurankmaheshwari/Documents/Projects/hacklytics/data/agent_dashboards/Financial_Performance_Dataset.csv')
    location_df = pd.read_csv('/Users/gaurankmaheshwari/Documents/Projects/hacklytics/data/agent_dashboards/Location_and_Disaster_Information_Dataset.csv')
    outcomes_df = pd.read_csv('/Users/gaurankmaheshwari/Documents/Projects/hacklytics/data/agent_dashboards/Project_Outcomes_Dataset.csv')

    # Process dates
    outcomes_df['dateInitiallyApproved'] = pd.to_datetime(outcomes_df['dateInitiallyApproved'], errors='coerce')
    outcomes_df['dateClosed'] = pd.to_datetime(outcomes_df['dateClosed'], errors='coerce')
    outcomes_df['ProjectTimelineMonths'] = np.floor(((outcomes_df['dateClosed'] - outcomes_df['dateInitiallyApproved']).dt.days / 30).fillna(0))

    # State name to abbreviation mapping
    state_abbreviations = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
        'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    location_df['state_code'] = location_df['state'].map(state_abbreviations)

    return financial_df, location_df, outcomes_df

financial_df, location_df, outcomes_df = load_data()

# ✅ Multi-Agent System
class LocationRiskAgent:
    def define_risk_zones(self, df):
        frequency_counts = df['state'].value_counts().reset_index()
        frequency_counts.columns = ['state', 'Frequency']
        frequency_counts['RiskZone'] = pd.cut(
            frequency_counts['Frequency'],
            bins=[0, frequency_counts['Frequency'].quantile(0.33),
                  frequency_counts['Frequency'].quantile(0.66),
                  frequency_counts['Frequency'].max()],
            labels=['Low Risk Zone', 'Moderate Risk Zone', 'High Risk Zone'],
            include_lowest=True
        ).astype(str)
        df = df.merge(frequency_counts[['state', 'RiskZone']], on='state', how='left')
        return df

class FinancialAgent:
    def visualize_financial_performance(self, df):
        scaling_factor = 1e6  # Scale to millions
        df['scaledProjectAmount'] = df['projectAmount'] / scaling_factor
        df['scaledFederalShare'] = df['federalShareObligated'] / scaling_factor
        
        fig = px.scatter(df, 
                         x='scaledFederalShare', 
                         y='scaledProjectAmount', 
                         color='benefitCostRatio',
                         title='Financial Performance of Disaster Recovery Projects',
                         color_continuous_scale='Viridis',
                         range_color=[0, 1.5],
                         labels={
                             'scaledFederalShare': 'Federal Share Obligated (Millions)',
                             'scaledProjectAmount': 'Project Amount (Millions)',
                             'benefitCostRatio': 'Benefit-Cost Ratio'
                         })
        fig.update_layout(
            xaxis=dict(range=[0, 50]),
            yaxis=dict(range=[0, 100]),
            height=600
        )
        return fig
    
class OutcomesAgent:
    def visualize_project_timeline(self, df):
        fig = px.histogram(df, x='ProjectTimelineMonths', nbins=30,
                           title='Distribution of Project Timelines (Months)')
        return fig

# ✅ Initialize Agents
location_risk_agent = LocationRiskAgent()
financial_agent = FinancialAgent()
outcomes_agent = OutcomesAgent()

# ✅ Process Data using Agents
location_df = location_risk_agent.define_risk_zones(location_df)

# ✅ Gemini LLM Prediction
def predict_insurance_scenario(location, federal_funding, cost_benefit_ratio, timeline_months):
    prompt = f"""
    Predict the insurance zone (HIGH, MODERATE, LOW) based on:
    - Location: {location}
    - Federal Funding: ${federal_funding}
    - Cost-Benefit Ratio: {cost_benefit_ratio}
    - Project Timeline: {timeline_months} months

    Output only one word: HIGH, MODERATE, or LOW.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip().upper()
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# ✅ Streamlit UI
st.title('Multi-Agent AI System Dashboard')

# ✅ Sidebar options
agent_option = st.sidebar.selectbox('Choose a Dashboard', ['Location Risk Zones', 'Financial Performance', 'Project Timeline', 'Insurance Coverage Prediction'])

# ✅ Location Risk Zones Dashboard
if agent_option == 'Location Risk Zones':
    st.header('Location Risk Zones (USA Map)')
    st.write('Visualizes geographical disaster risk levels across the US.')
    location_df['text'] = location_df['state'] + '<br>' + location_df['RiskZone']
    fig = px.scatter_geo(location_df,
                         locations='state_code',
                         locationmode='USA-states',
                         color='RiskZone',
                         hover_name='text',
                         scope='usa',
                         title='Geographical Clusters of Risk Zones')
    st.plotly_chart(fig)

# ✅ Financial Performance Dashboard
elif agent_option == 'Financial Performance':
    st.header('Financial Performance Dashboard')
    st.write('Scatter plot of project amounts and federal funding, colored by the benefit-cost ratio.')
    fig = financial_agent.visualize_financial_performance(financial_df)
    st.plotly_chart(fig)

# ✅ Project Timeline Dashboard
elif agent_option == 'Project Timeline':
    st.header('Project Timeline Dashboard')
    st.write('Distribution of project durations in months.')
    fig = outcomes_agent.visualize_project_timeline(outcomes_df)
    st.plotly_chart(fig)

# ✅ Insurance Coverage Prediction
elif agent_option == 'Insurance Coverage Prediction':
    st.header('Insurance Coverage Prediction Using Gemini LLM')
    st.write('Predict whether the insurance coverage will be HIGH, MODERATE, or LOW.')

    with st.form("insurance_prediction_form"):
        location = st.selectbox('Select Location', location_df['state'].unique())
        federal_funding = st.number_input('Federal Funding ($)', min_value=10000, max_value=10000000, value=500000)
        cost_benefit_ratio = st.number_input('Benefit-Cost Ratio', min_value=0.1, max_value=10.0, value=1.5)
        timeline_months = st.number_input('Project Timeline (Months)', min_value=1, max_value=60, value=12)
        submitted = st.form_submit_button("Predict Insurance Zone")

    if submitted:
        with st.spinner("Predicting..."):
            prediction = predict_insurance_scenario(location, federal_funding, cost_benefit_ratio, timeline_months)
            st.success(f"Predicted Insurance Zone: **{prediction}**")

# ✅ Footer
st.markdown("---")
st.markdown("**Multi-Agent AI System with Gemini LLM - Insurance Cost Prediction**")