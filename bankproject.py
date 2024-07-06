import pandas as pd 
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pickle
from PIL import Image
import re

df = pd.read_csv("C:/Users/LENOVO/Desktop/Files/Datae2.csv")
df1=pd.read_csv("C:/Users/LENOVO/Desktop/Files/mydat1.csv")
df2=pd.read_csv("C:/Users/LENOVO/Desktop/Files/finaldat1.csv")
#dfg=pd.read_csv("C:/Users/LENOVO/Desktop/Files/Datt.csv")

# --------------------------------------------------Logo & details on top

icon = Image.open("pe.png")
st.set_page_config(page_title= "Bank Risk Controller System | By Dhanalakshmi S",
                   page_icon= icon,
                   layout= "wide",
                   initial_sidebar_state= "expanded")


with st.sidebar:
    st.image("dnx.jpeg")
   


    opt = option_menu("Menu",
                    ["Home",'Data Showcase','Data Visualization','ML Prediction','ML Recommendation system',"About"],
                    icons=["house","table","bar-chart-line","graph-up-arrow","search", "exclamation-circle"],
                    menu_icon="cast",
                    default_index=0,
                    styles={"icon": {"color": "orange", "font-size": "20px"},
                            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "-2px", "--hover-color": "#F2D2BD"},
                            "nav-link-selected": {"background-color": "#FBCEB1"}})

def home():
    if opt=="Home":
        
            col,coll = st.columns([1,4],gap="small")
            with col:
              st.write(" ")
            with coll:
                st.markdown("# :orange[*Bank*] *Risk* :orange[*Controller*] *System*")
                st.markdown(
                        """
                        <hr style="border: none; height: 5px; background-color: #FFFFFF;" />
                        """,
                        unsafe_allow_html=True
                    )   
                st.write(" ")     
            st.markdown("### :orange[*OVERVIEW* ]")
            st.markdown("### *The expected outcome of this project is a robust predictive model that can accurately identify customers who are likely to default on their loans. This will enable the financial institution to proactively manage their credit portfolio, implement targeted interventions, and ultimately reduce the risk of loan defaults.*")
            col1,col2=st.columns([3,2],gap="large")
            with col1:
                st.markdown("### :orange[*DOMAIN* ] ")
                st.markdown(" ### *Banking* ")
                st.markdown("""
                            ### :orange[*TECHNOLOGIES USED*]     

                            ### *PYTHON*
                            ### *DATA PREPROCESSING*
                            ### *EDA*
                            ### *PANDAS*
                            ### *NUMPY*
                            ### *VISUALIZATION*
                            ### *MACHINE LEARNING*
                            ### *STREAMLIT GUI*
                            
                            """)
            with col2:
                    st.write(" ")
                    st.image("bank.jpg",caption=' ', use_column_width=True)

home()




def data():
    if opt=="Data Showcase":
                    st.header("Data Used")
                    st.dataframe(df)

                    st.header("Model Performance")
                    data = {
                                "Algorithm": ["Decision Tree","Random Forest","KNN","XGradientBoost"],
                                "Accuracy": [89,88,97,93],
                                "Precision": [90,90,96,94],
                                "Recall": [89,89,96,94],
                                "F1 Score": [89,89,97,94]
                                
                                }
                    dff = pd.DataFrame(data)
                    st.dataframe(dff)
                    st.markdown(f"## The Selected Algorithm is :orange[*XGradient Boost*] and its Accuracy is   :orange[*93%*]")

data()

def pred():
    if opt=="ML Prediction":
           
           
            # Function to safely convert to sqrt
            def safe_sqrt(value):
                try:
                    return np.sqrt(float(value))  # Conversion to float
                except (ValueError, TypeError):
                    return np.nan  

            # Streamlit form for user inputs
            st.markdown(f'## :violet[*Predicting Customers Default on Loans*]')
            st.write(" ")
            st.write( f'<h5 style="color:#FBCEB1;"><i>NOTE: Min & Max given for reference, you can enter any value</i></h5>', unsafe_allow_html=True )

            with st.form("my_form"):
                col1, col2 = st.columns([5, 5])
                
                with col1:
                    TOTAL_INCOME = st.text_input("TOTAL INCOME (Min: 25650.0 & Max: 117000000.0)", key='TOTAL_INCOME')
                    AMOUNT_CREDIT = st.text_input("CREDIT AMOUNT (Min: 45000.0 & Max: 4050000.0)", key='AMOUNT_CREDIT')
                    AMOUNT_ANNUITY = st.text_input("ANNUITY AMOUNT (Min: 1980.0 & Max: 225000.0)", key='AMOUNT_ANNUITY')
                    OCCUPATION_TYPE = st.text_input("OCCUPATION TYPE (Min: 0 & Max: 17)", key='OCCUPATION_TYPE')

                with col2:
                    EDUCATION_TYPE = st.text_input("EDUCATION TYPE (Min: 0 & Max: 4)", key='EDUCATION_TYPE')
                    FAMILY_STATUS = st.text_input("FAMILY STATUS (Min: 0 & Max: 4)", key='FAMILY_STATUS')
                    OBS_30_COUNT = st.text_input("OBS_30 COUNT (Min: 0 & Max: 348.0)", key='OBS_30_COUNT')
                    DEF_30_COUNT = st.text_input("DEF_30 COUNT (Min: 0 & Max: 34.0)", key='DEF_30_COUNT')

                submit_button = st.form_submit_button(label="PREDICT STATUS")

                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #FBCEB1;
                        color: purple;
                        width: 50%;
                        display: block;
                        margin: auto;
                    }
                    </style>
                """, unsafe_allow_html=True)

            # Validate input
            flag = 0 
            pattern = "^(?:\d+|\d*\.\d+)$"

            for i in [TOTAL_INCOME, AMOUNT_CREDIT, AMOUNT_ANNUITY, OCCUPATION_TYPE, EDUCATION_TYPE, FAMILY_STATUS, OBS_30_COUNT, DEF_30_COUNT]:             
                if re.match(pattern, i):
                    pass
                else:                    
                    flag = 1  
                    break

            if submit_button and flag == 1:
                if len(i) == 0:
                    st.write("Please enter a valid number, space not allowed")
                else:
                    st.write("You have entered an invalid value: ", i)  

            if submit_button and flag == 0:
                with open(r"xgbmodel.pkl", 'rb') as file:
                   xgb = pickle.load(file)

                #with open(r'xgbscaler.pkl', 'rb') as f:
                   #scaler_loaded = pickle.load(f)
                    
                sample = np.array([
                    [
                        safe_sqrt(TOTAL_INCOME),          # Convert TOTAL_INCOME to float and take sqrt
                        safe_sqrt(AMOUNT_CREDIT),         # Convert AMOUNT_CREDIT to float and take sqrt
                        safe_sqrt(AMOUNT_ANNUITY),        # Convert AMOUNT_ANNUITY to float and take sqrt
                        int(OCCUPATION_TYPE),             # Convert OCCUPATION_TYPE to integer
                        int(EDUCATION_TYPE),              # Convert EDUCATION_TYPE to integer
                        int(FAMILY_STATUS),               # Convert FAMILY_STATUS to integer
                        safe_sqrt(OBS_30_COUNT),          # Convert OBS_30_COUNT to float and take sqrt
                        safe_sqrt(DEF_30_COUNT)           # Convert DEF_30_COUNT to float and take sqrt
                    ]
                ])

                #sample = scaler_loaded.transform(sample)
                pred = xgb.predict(sample)
                if pred == 1:
                    st.markdown(f' ## :grey[The status is :] :orange[Repay]')
                else:
                    st.write(f' ## :orange[The status is ] :grey[Won\'t Repay]')

pred()


def recomm():
        if opt=="ML Recommendation system":
                st.write(" ")
                col,coll,col2 = st.columns([2,4,1],gap="small")
                with col:
                    st.write(" ")
                with coll:
                    st.markdown("# :red[*Movie*] *Recommendation* :red[*System*] ")
                with col2:
                    st.write(" ")

                # Load the movies and similarity data
                movies = pickle.load(open("recmnd.pkl", 'rb'))
                similarity = pickle.load(open("similarity.pkl", 'rb'))
                recmnd = movies['title'].values

                # Select movie from dropdown
                selectvalue = st.selectbox("Select movie from dropdown", recmnd)

                # Function to recommend movies
                def recommend(movie):
                    index = movies[movies['title'] == movie].index[0]
                    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
                    recommend_movie = []
                    for i in distance[1:6]:
                        recommend_movie.append(movies.iloc[i[0]].title)
                    return recommend_movie

                # Show recommendations when button is clicked
                if st.button("Show Recommend"):
                    movie_name = recommend(selectvalue)
                    
                    # Get the details of the selected movie
                    selected_movie = movies[movies['title'] == selectvalue].iloc[0]
                    selected_genre = selected_movie['genres']
                    
                    # Display the selected movie details
                    st.markdown(f"## :violet[*The Movie you choosen is*]   {selectvalue} ") 
                    st.markdown(f"## :violet[*Genre of that is*]  {selected_genre} ")

                    st.write(" ")
                    st.markdown(f"### :rainbow[*Here is your recommendation !*]")  
                    st.write(" ")
                    # Display recommended movies in columns
                    for name in movie_name:
                        recommended_movie = movies[movies['title'] == name].iloc[0]
                        recommended_genre = recommended_movie['genres']
                        st.markdown(f"#### Title: :orange[{name}]")
                        st.markdown(f"#### Genre: :blue[{recommended_genre}]")
                        st.write("") 

recomm()

def visual():
    if opt=="Data Visualization":
                            st.subheader("Insights of Bank Risk Controller System")
             
                

                #tab1,tab2 = st.tabs(["Rough Analysis", "Target Analysis"])
                #with tab1:

                    #--------------------------------------------------------------1

                            # Assuming df is your DataFrame and 'AMT_INCOME_TOTAL' is your column of interest
                            fig = px.histogram(df, x='AMT_INCOME_TOTAL_sqrt', nbins=50, marginal='box', histnorm='density')

                            # Add KDE line
                            fig.update_traces(marker_color='blue', opacity=0.7)
                            fig.add_scatter(x=df['AMT_INCOME_TOTAL_sqrt'], y=df['AMT_INCOME_TOTAL_sqrt'].value_counts(normalize=True).sort_index(),
                                            mode='lines', name='KDE', line=dict(color='red'))

                            # Update layout for better visualization
                            fig.update_layout(title='AMT_INCOME_TOTAL Distribution with KDE',
                                            xaxis_title='AMT_INCOME_TOTAL',
                                            yaxis_title='Density',
                                            showlegend=True)

                            st.plotly_chart(fig,use_container_width=True)

                            #--------------------------------------------------------------4

                            #  Bar Plot: Top 10 Occupation Types
                            occupation_counts = df1['OCCUPATION_TYPE'].value_counts().reset_index()
                            occupation_counts.columns = ['OCCUPATION_TYPE', 'COUNT']

                            # Create a bar chart
                            fig = px.bar(occupation_counts, y='OCCUPATION_TYPE', x='COUNT',color="COUNT", title='Occupation Type Counts',color_continuous_scale='PiYG')
                            st.plotly_chart(fig,use_container_width=True)
                            

                            #--------------------------------------------------------------5

                                        
                            INCOME_counts = df1['NAME_INCOME_TYPE'].value_counts().reset_index()
                            INCOME_counts.columns = ['NAME_INCOME_TYPE', 'COUNT']

                            # Create a bar chart
                            fig = px.line(INCOME_counts, x='NAME_INCOME_TYPE', y='COUNT', title='Income Type Counts')
                            st.plotly_chart(fig,use_container_width=True)

                            #--------------------------------------------------------------6

                            family = df1['NAME_FAMILY_STATUS'].value_counts().reset_index()
                            family.columns = ['NAME_FAMILY_STATUS','COUNT']

                            # Create a bar chart
                            fig = px.pie(family, names='NAME_FAMILY_STATUS', values='COUNT', title='Family Status Distribution')
                            st.plotly_chart(fig,use_container_width=True)

                            #--------------------------------------------------------------7

                            EDUCATION_counts = df1['NAME_EDUCATION_TYPE'].value_counts().reset_index()
                            EDUCATION_counts.columns = ['NAME_EDUCATION_TYPE', 'COUNT']

                            # Create a bar chart
                            fig = px.bar(EDUCATION_counts, x='NAME_EDUCATION_TYPE', y='COUNT',color='COUNT',
                                            color_continuous_scale='Viridis', title='Occupation Type Counts')
                            fig.update_layout(legend_title_text='Education Type')
                            st.plotly_chart(fig,use_container_width=True)

                            #--------------------------------------------------------------3
                            
                            fig2 = px.pie(df1, names='NAME_CONTRACT_TYPE_x', title='Distribution of Contract Types')
                            fig2.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig2,use_container_width=True)


                #with tab2:

                            #--------------------------------------------------------------2

                            dff = df2[['AMT_INCOME_TOTAL_sqrt',
                                    'AMT_CREDIT_x_sqrt', 'AMT_ANNUITY_x_sqrt',
                                    'OCCUPATION_TYPE_sqrt', 'NAME_EDUCATION_TYPE_sqrt',
                                    'AMT_GOODS_PRICE_x_sqrt',
                                    'OBS_30_CNT_SOCIAL_CIRCLE_sqrt',"TARGET"]]

                            # Calculate the correlation matrix
                            corr = dff.corr().round(2)

                            # Plot the heatmap using Plotly Express
                            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu",
                                            title="Correlation Matrix Heatmap")
                            st.plotly_chart(fig,use_container_width=True)

                        

                            #--------------------------------------------------------------8

                            fig1 = px.histogram(df1, x='OCCUPATION_TYPE', color='TARGET', barmode='group')
                            fig1.update_layout(title='Countplot of TARGET by OCCUPATION_TYPE', xaxis_title='OCCUPATION_TYPE', yaxis_title='Count')
                            st.plotly_chart(fig1,use_container_width=True)



visual()





            


    










