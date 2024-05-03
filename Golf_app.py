import streamlit as st
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
import plotly.express as px

st.sidebar.markdown(
    "My Logo (sidebar) should be on top of the Navigation within the sidebar"
)

st.markdown("# Home")




# Export the data needed from the db to run analysis
conn = sqlite3.connect('Golf_data.db')

query = "SELECT * FROM 'Course Stats'"
df = pd.read_sql_query(query, conn)
cluster_df = df.copy()
course_names = df['course']
cluster_df.dropna(inplace=True)
drop_columns = ['id', 'course']
cluster_df.drop(columns=drop_columns, inplace=True)
columns = cluster_df.columns.tolist()
scaler = StandardScaler()
cluster_df[columns] = scaler.fit_transform(cluster_df[columns])
pca = PCA(n_components=2)
principal_components = pca.fit_transform(cluster_df[columns])
conn.close()
kmeans = KMeans(n_clusters=2, random_state=20)
kmeans.fit(cluster_df[columns])
cluster_df['kmeans_2'] = kmeans.labels_
# Function to create an elbow plot for kmeans
def optimise_k_means(data, max_k):
    means = []
    inertias = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        means.append(k)
        inertias.append(kmeans.inertia_)

    # Generate the elbow plot using Plotly Express
    fig = px.line(x=means, y=inertias, title='Elbow Plot', labels={'x': 'Number of Clusters', 'y': 'Inertia'})
    fig.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig)

#  function to create a graph plotting the course clusters
def visualize_clusters(data, clusters, title):

    
    
    # Create a Plotly figure
    fig_plotly = px.scatter(
        x=data[:, 0],
        y=data[:, 1],
        color=clusters,
        color_continuous_scale='viridis',
        hover_name=course_names,
        title=title,
        labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'}
    )
    fig_plotly.update_layout(plot_bgcolor='white')
    # Display Plotly figure in Streamlit
    st.plotly_chart(fig_plotly)

# function to create a box plot to visualize how the courses were clustered
def box_plot_cluster():
    # Group the DataFrame by cluster label
    cluster_groups = cluster_df.groupby('kmeans_2')

    # Calculate the mean or median value of each column within each cluster
    cluster_means = cluster_groups.mean()  # You can also use .median() if preferred

    # Plot the bar graph
    fig, ax = plt.subplots(figsize=(12, 8))
    cluster_means.plot(kind='bar', ax=ax)
    plt.title('Mean Value of Each Stat by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=0)
    plt.legend(title='Stat')
    st.pyplot(fig)

# Convert the hole proximity stat into inches
def parse_distance(distance_str):
    # Check if distance_str is empty
    if not distance_str:
        return np.nan
    
    # Check if the distance_str contains a space
    if "'" in distance_str:
        # Split the string by space
        distance_list = distance_str.split("'")
    else:
        # Split the string by non-digit characters
        distance_list = re.split(r'\D+', distance_str)
    
    # Extract the feet and inches parts
    if len(distance_list) >= 2:
        feet = int(distance_list[0])
        inches_str = distance_list[1].strip('"')  # Remove the trailing double quote
        inches = int(inches_str)
        return feet * 12 + inches  # Convert feet and inches to total inches
    elif len(distance_list) == 1:
        return int(distance_list[0]) * 12  # Only feet provided, convert to inches
    else:
        return np.nan  # Invalid format, return NaN
        
# Function to run the Multiple Linear Regression
def run_mlr(year, cluster):
    conn = sqlite3.connect('Golf_data.db')




    player_stats_query = "SELECT * FROM 'Player Stats'"
    winner_tourney_query = "SELECT * FROM 'Tournament & Course Winners'"
    stats = pd.read_sql_query(player_stats_query, conn)
    wins = pd.read_sql_query(winner_tourney_query,conn)


    # Filter winners dataframe for each cluster
    cluster_winners = wins[(wins['Course Cluster'] == cluster) & (wins['Year'] == year)]

    # Get unique player names or winners
    cluster_players = cluster_winners['Winner']
    #print(cluster_players)
    players = stats[(stats['year'] == str(year))]
    cluster_players_df = pd.DataFrame(cluster_players, columns=['Winner'])

    # List to store player data
    player_data_list = []

    for player_name in  players['name']:
        # Filter player_stats for the desired year and player
        player_stats = stats[(stats['name'] == player_name) & (stats['year'] == str(year))]
        #player_stats['hole_proximity_avg'] = player_stats['hole_proximity_avg'].apply(parse_distance)
        player_stats.loc[:, 'hole_proximity_avg'] = player_stats['hole_proximity_avg'].apply(parse_distance)

        # Count the number of wins for each cluster
        

        # Now you can perform operations on the DataFrame
        cluster_wins = len(cluster_players_df[cluster_players_df['Winner'] == player_name])
        # Append player's data to the list
        player_data_list.append({
            'Name': player_name,
            'Drive_Avg': player_stats['drive_avg'].iloc[0],
            'Drive_Acc': player_stats['drive_acc'].iloc[0],
            'Gir_Pct': player_stats['gir_pct'].iloc[0],
            'Putt_Avg': player_stats['putt_avg'].iloc[0],
            'Sand_Saves_Pct': player_stats['sand_saves_pct'].iloc[0],
            'Birdies_Per_Round': player_stats['birdies_per_round'].iloc[0],
            'Holes_Per_Eagle': player_stats['holes_per_eagle'].iloc[0],
            'Strokes_Gained': player_stats['strokes_gained'].iloc[0],
            'Hole_Proximity_Avg': player_stats['hole_proximity_avg'].iloc[0],
            'Scrambling_Pct': player_stats['scrambling_pct'].iloc[0],
            'Scoring_Avg': player_stats['scoring_avg'].iloc[0],
            'Strokes_Gained_Tee_Green': player_stats['strokes_gained_tee_green'].iloc[0],
            'Strokes_Gained_Total': player_stats['strokes_gained_total'].iloc[0],
            'Total_Driving': player_stats['total_driving'].iloc[0],
            'cluster_wins': cluster_wins
        })

    # Create a DataFrame from the list of player data
    player_stats_with_wins = pd.DataFrame(player_data_list)

    # Display the resulting DataFrame

    conn.close()

    #sns.lmplot(x='Hole_Proximity_Avg', y='Cluster_0_Wins', data=player_stats_with_wins)
    #plt.show()

    mlr_df = player_stats_with_wins.drop(columns=['Name'], axis=1)
    mlr_df.dropna(inplace=True, axis=0)
    #print(mlr_df)

    X = mlr_df[['Drive_Avg', 'Drive_Acc', 'Gir_Pct', 'Putt_Avg', 'Sand_Saves_Pct',
        'Birdies_Per_Round', 'Holes_Per_Eagle', 'Strokes_Gained',
        'Hole_Proximity_Avg', 'Scrambling_Pct', 'Scoring_Avg',
        'Strokes_Gained_Tee_Green', 'Strokes_Gained_Total', 'Total_Driving']]
    y = mlr_df[['cluster_wins']]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=34)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    #print(lr.score(X_train, y_train))

    y_pred = lr.predict(X_test)
    abs_error = mean_absolute_error(y_test, y_pred)
    coef = lr.coef_[0]
   


    prediction = {'Drive_Avg': coef[0], 'Drive_Acc': coef[1] , 'Gir_Pct': coef[2], 'Putt_Avg': coef[3], 'Sand_Saves_Pct': coef[4],
        'Birdies_Per_Round':coef[5] , 'Holes_Per_Eagle': coef[6], 'Strokes_Gained': coef[7],
        'Hole_Proximity_Avg': coef[8], 'Scrambling_Pct': coef[9], 'Scoring_Avg': coef[10],
        'Strokes_Gained_Tee_Green': coef[11], 'Strokes_Gained_Total': coef[12], 'Total_Driving': coef[13], 'Abslolute Error': abs_error, 'R2 Score': r2_score(y_test, y_pred), 'Prediction Magnitude': int(lr.intercept_[0])}
    prediction_series = pd.Series(prediction)
    df = pd.DataFrame(prediction_series, columns=['Values'])
    return df
    
# Function to query Course Info
def get_course_info(tournament, year):
    conn = sqlite3.connect('Golf_data.db')
    cursor = conn.cursor()
    # Create Course stats data frame 
    course_stats = pd.read_sql_query("SELECT * FROM 'Course Stats'", conn)
    course_stats.drop(columns=['id'], axis=1)
    # Fetch Course name
    cursor.execute(f"SELECT Course_name FROM 'Tournament & Course Winners' WHERE tournament_name='{tournament}' AND year={year}")
    course_name_tuple = cursor.fetchone()
    # Fetch Course info, put in exception if course info is not found
    if course_name_tuple:
        course_name = course_name_tuple[0]
        course_info = course_stats[(course_stats['course'] == course_name)]
    else: 
        course_info = "No Data for this year"
    return course_info

# Function to query winner info 
def get_winner_info(tournament, year):
    conn = sqlite3.connect('Golf_data.db')
    cursor = conn.cursor()
    player_stats = pd.read_sql_query("SELECT * FROM 'Player Stats'", conn)
    player_stats.drop(columns=['id'], axis=1)
    cursor.execute(f"SELECT Winner FROM 'Tournament & Course Winners' WHERE tournament_name='{tournament}' AND year={year}")
    winner_info = cursor.fetchone()
    if winner_info:
        winner_name = winner_info[0]
        player_info = player_stats[(player_stats['name'] == winner_info[0]) & (player_stats['year'] == str(year))]
    else: 
        winner_name = None
        player_info = None
    conn.close()
    return winner_name, player_info

# Function to Query Winner stats
def get_winner_stats(tournament, year):
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM Winner_Stats WHERE tournament_name='{tournament}' AND year={year}")
    winner_stats = cursor.fetchall()
    conn.close()
    return winner_stats

# Function to get tournament names
def get_unique_values_from_db(column_name, table_name):
    conn = sqlite3.connect('Golf_data.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT {column_name} FROM '{table_name}'")
    unique_values = cursor.fetchall()
    conn.close()
    return [value[0] for value in unique_values]

# Create multiple pages
def main():
    st.title('PGA Data Exploration Project')

    # Add navigation to sidebar
    page = st.sidebar.selectbox("Choose a page", ["Main Page", "Statistical Analysis", "Data Exploration", "Data Set Description", "Questions"])

    # Render different pages based on user selection
    if page == "Main Page":
        render_main_page()
    elif page == "Statistical Analysis":
        render_stats_page()
    elif page == "Data Exploration":
        render_data_page()
    elif page == 'Data Set Description':
        render_description_page()
    elif page == 'Questions':
        render_conclusion_page()

# Create main page to answer questions 
def render_main_page():
    st.header('Main Page')
    st.write('Questions/Answers')
    st.markdown('''
        Cole Whitelaw
                
        2. An explanation of how to use your webapp: what interactivity there is, what the plots/charts mean, what your conclusions were, etc
        
        Answer:
        This web app is a presentation of PGA tour data for the years of 2013 through 2023. The data exploration page allows you to query through specific tournamnets and the year they were played. 
        After the tournament and year is selected, the app will present the course it was played on along with its statistics and the winner of that tournament along with the winners statistics for that year. 
        The statistical analysis was ran to find which player statistics have the largest impact on specific types of courses. The analysis uses a kmeans clustering algorithm to group the courses and a multiple linear regression
        analysis to find which player statistics have the largest impacts on course cluster wins. As of now it seems like there is not much correlation to player statistics and course cluster wins, this being shown on the statistical analysis page.
        The statistical analysis pgae has 4 graphs. The elbow plot is is used to decide which amount of clusters was used to run the analysis, while the second graph visulizes the clusters to show how they were grouped. This graph is also interactive so you can find out which courses were apart of which cluster by clicking the point on the graph.
        The third graph shows how the courses were clustered representing a bar graph of the mean of each course statistic of all the courses. The last part of this page allows you to choose a year and cluster to run a multiple linear regression on and creates a bar graph to see which player statistic had the largest impact in that year on the course cluster you chose.  A better description of the analysis and its specific components can be explained on the Statistical analysis page.  

        3. Any major “gotchas” (i.e. things that don’t work, go slowly, could be improved, etc.)
        
        Answer:
        Unfortunately, the analysis did not come to any major conclusions as to what statistics have the largest impact, even though there were some statistics that had a larger impact than others it was not significant. The analysis was ran for each year and each year had different results. 
        It may have been beneficial to run this analysis for all years, which may have presented a better conclusion but it also may have been worse. The data seems to be a lot better in the current
        years versus previous which may affect the analysis in the later years presented. Also, the clustering of the courses is by no means perfect, there is not a clear cut distinction between the groups of courses. It also could have been beneficial 
        to run a multiple linear regression on each course instead of using clusters but again the varaince could have been worse. 

        
    '''
    )
# Render stats page
def render_stats_page():
    st.header('Statistical Analysis')
    st.write('''This analysis uses a kmeans clusterig algorithm to group each course into a cluster, based on the analyis 2 clusters was the most reasonable (Cluster 0 & Cluster 1, see elbow plot). 
             Based on the statistics cluster 0 courses seemed to be made up of shorter courses but harder skill play while cluster 1 courses were longer courses but easier skill play. After the clustering was done and added into the data base 
             a multiple linear regression was used to compare the number of wins on a specific cluster against the player statistics. In the drop down menu below you can select a year and course cluster. This will show how much the course cluster wins 
             were explained by player statistics and the variance explained of each stat. The bar graph will show which specific statistics hold the most weight for that year and course cluster. 
             
             ''')
    st.write('''
             


    The elbow plot below helps in selecting the optimal number of clusters for K-means clustering. A good clustering solution strikes a balance between having enough clusters to capture the underlying structure of the data without overfitting. The elbow point provides a heuristic for choosing the number of clusters that best represents the inherent patterns in the data.
    ''')
    # Create elow graph 
    optimise_k_means(cluster_df[columns], 10)
    
    #kmeans = KMeans(n_clusters=2)
    #kmeans.fit(cluster_df[columns])
    #cluster_df['kmeans_2'] = kmeans.labels_
    st.write('''
    This graph below provides a visual representation of data points in a two-dimensional space, with clusters differentiated by color.
    The graph is also interactave so you can see which course falls within each cluster by putting your cursor over a point on the graph. 
             ''')
    # Show cluster graph
    visualize_clusters(principal_components, kmeans.labels_, 'KMeans Clustering with 2 clusters')

    st.write('''
    This graph below shows how the courses were clustered, the statistics labeled in the middle and the bar graph respresenting the mean of those statistics. 
             ''')
    # Show box plot
    box_plot_cluster()


    # Create interactivity to run Multiple Linear Regression on chosen years and clusters 
    st.write('''
    Here a multiple linear regression analysis will be ran based on the year and cluster you choose. The data will present with the variance explained of each statistic along with Absolute error, R2 Value, and Prediction Magnitude. 
    The bar graph plots each statistic and its variance explained showing which statistics had the largest impact on that year specific to the cluster you choose. 
             ''')
    year_options = [2023,2022,2021,2020,2019,2018,2017,2016]
    cluster_options = [0,1]
    selected_year = st.selectbox('Select Year', year_options)
    selected_cluster = st.selectbox('Select Cluster', cluster_options)
    data = run_mlr(selected_year,selected_cluster)
    df = pd.DataFrame(data)
    st.write(data)
    # Create a box plot for the 'Drive_Avg' key
    exclude_columns = ['Abslolute Error', 'R2 Score', 'Prediction Magnitude']
    data_filtered = data[~data.index.isin(exclude_columns)]

    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot positive values
    data_filtered[data_filtered >= 0].plot(kind='bar', stacked=True, color='green', ax=ax)

    # Plot negative values
    data_filtered[data_filtered < 0].plot(kind='bar', stacked=True, color='blue', ax=ax)

    plt.title('Level of Impact for each statistic')
    plt.xlabel('Statistic')
    plt.ylabel('Variance Explained')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend(['Positive', 'Negative'], loc='upper right')  # Add legend
    st.pyplot(fig)
    
# Create data exploration page 
def render_data_page():
    st.header('Data Exploration')
    st.write('Find course statistics and winner statistics for specific tournaments and years')
    # Create Tournament option 
    tournament_options = get_unique_values_from_db('tournament_name', 'Tournament & Course Winners' )
    selected_tournament = st.selectbox('Select Tournament', tournament_options)
    # Create Year Option
    year_options = get_unique_values_from_db('year', 'Years')
    selected_year = st.selectbox('Select Year', year_options)
    # Get Course info
    course_info = get_course_info(selected_tournament, selected_year)
    st.subheader('Course Information')
    st.write(course_info)
    winner_info, winner_stats = get_winner_info(selected_tournament, selected_year)
    st.subheader('Winner Information')
    st.write(winner_info)
    st.subheader('Winner Stats')
    st.write(winner_stats)

def render_description_page():
    st.header('Data Set Description')
    st.write('Description of Data Sets Used')
    st.markdown('''

        Data Set 1: Sportradar API: https://developer.sportradar.com/golf/reference/golf-overview
        This API returns all PGA tour golfers statistics per year, below is an example of one player in 2024 from the API endpoint. This is for Seung-Yul Noh in 2024. [{"id":"7d470d98-95a1-42a5-b5eb-39bd2259ac5f","first_name":"Seung-Yul","last_name":"Noh","country":"SOUTH KOREA","abbr_name":"S.Noh","statistics":{"events_played":1,"first_place":0,"second_place":0,"third_place":0,"top_10":0,"top_25":0,"cuts":0,"cuts_made":1,"withdrawals":0,"points":12.0,"points_rank":185,"earnings":21480.0,"earnings_rank":190,"drive_avg":293.6,"drive_acc":60.71,"gir_pct":72.22,"putt_avg":1.731,"sand_saves_pct":50.0,"birdies_per_round":4.25,"world_rank":538,"scrambling_pct":75.0,"scoring_avg":71.225,"total_driving":1998}}

        Data set 2: pgatour.com: https://www.pgatour.com/schedule/2024 
        For this data source I scraped all the past PGA tournament courses and winners. The data is stored in the HTML so I scraped the data using BeautifulSoup. 

        Data Set 3: https://datagolf.com/course-table
        The last data source was a CSV file exported from Datagolf.com. This csv file contains statistics of most of the golf courses on the PGA tour (some courses are missing). Examples of variables include: par, total yardage, total scoring average, etc. 


        Note: For the first two data sets the data was collected for the years of 2013-2023
                
        The data collected from these data sets was combined into a sqlite relational data base. The model for this data base is located in the github repository.

        
    '''
    )






def render_conclusion_page():
    st.header('Questions')
    st.write('Conlcuding Questions')
    st.markdown('''

        1. What did you set out to study?  (i.e. what was the point of your project?  This should be close to your Milestone 1 assignment, but if you switched gears or changed things, note it here.)
        
            Answer: I set out to study if specific types of golfers tend to do better on specific types of courses. Are there certain skills/statistics that enable players to win more on certain courses based on the course statistics? 

        2. What did you Discover/what were your conclusions (i.e. what were your findings?  Were your original assumptions confirmed, etc.?)

            Answer: Based on the data I had there was not a clear answer as to which players did well on what courses. 
            The correlation was weak and it changed every year, maybe pointing to the fact that there is no correlation. More data or a different analysis may change this conclusion but more research needs to be done. 
            I thought there would be a stornger correlation but this was not the case, the correlation was also very variable between each year and cluster.       

        3. What difficulties did you have in completing the project?
            
            Answer: Finding all the data and scraping it was the most difficult part of the project. Ensuring that the data was collected and inputted 
            correctly into the data based was a challenge because making a small mistake could corupt the entire data base. It was also not easy to scrape from PGA.com, the html was very messy and unorganized making it difficult to structure the data in order to properly input it into my database.  

        4. What skills did you wish you had while you were doing the project?
        
            Answer: It would have been extremely beneficial having a more complete understanding of building relational data bases and querying them in SQL. 
            I ran into some difficulty with joining tables and querying information, which I solved by just creating a pandas dataframe from the database tables.
            The web app would be more complete if the data was complied directly from the sqlite databse but I ran into some difficulties doing so. 

        5. What would you do “next” to expand or augment the project?
            
            Answer: I would try to find more data to use/input into the data base to make my analysis more robust. If there was more data or less missing values it may have been viable to run a multiple linear regression on all the years together instea dof doing it seperately. Also, I would try to run a multiple linear 
            regression for each course instead of clustering them. The clustering analysis may not be a perfect fit based on the variability of the courses. 
    ''')




if __name__ == "__main__":
    main()
