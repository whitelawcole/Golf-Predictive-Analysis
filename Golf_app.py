import streamlit as st
import sqlite3
#import matplotlib.pyplot as plt
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
    st.pyplot()


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
    

def query_database(query):
    conn = sqlite3.connect('Golf_data.db')
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data


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

def get_winner_stats(tournament, year):
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM Winner_Stats WHERE tournament_name='{tournament}' AND year={year}")
    winner_stats = cursor.fetchall()
    conn.close()
    return winner_stats


def get_unique_values_from_db(column_name, table_name):
    conn = sqlite3.connect('Golf_data.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT {column_name} FROM '{table_name}'")
    unique_values = cursor.fetchall()
    conn.close()
    return [value[0] for value in unique_values]


def main():
    st.title('PGA Data Exploration Project')

    # Add navigation to sidebar
    page = st.sidebar.selectbox("Choose a page", ["Main Page", "Statistical Analysis", "Data Exploration"])

    # Render different pages based on user selection
    if page == "Main Page":
        render_main_page()
    elif page == "Statistical Analysis":
        render_stats_page()
    elif page == "Data Exploration":
        render_data_page()

def render_main_page():
    st.header('Main Page')
    st.write('Questions/Answers')
    st.markdown('''
        Cole Whitelaw
        
        This web app allows you to query through specific PGA Tournaments and find the winner for that year, the course and course statistics for that year and winner statistics for that year. This will be on the
        Data Exploration page. The statistical analysis is explained on the 

        3. Any major “gotchas” (i.e. things that don’t work, go slowly, could be improved, etc.)

        In addition, you’ll need to answer the following questions (they’ll go in a specific page in your webapp.  Don’t worry, we’ll learn how to do that!)

        1. What did you set out to study?  (i.e. what was the point of your project?  This should be close to your Milestone 1 assignment, but if you switched gears or changed things, note it here.)

        2. What did you Discover/what were your conclusions (i.e. what were your findings?  Were your original assumptions confirmed, etc.?)

        3. What difficulties did you have in completing the project?

        4. What skills did you wish you had while you were doing the project?

        5. What would you do “next” to expand or augment the project?
    '''
    )

def render_stats_page():
    st.header('Statistical Analysis')
    st.write('''This analysis used a kmeans clusterig algorithm to group each course into a cluster, based on the analyis 2 clusters were the most accurate (Cluster 0 & Cluster 1). 
             Based on the statistics cluster 0 courses seemed to be made up of shorter courses but harder skill play while cluster 1 courses were longer courses but easier skill play. After the clustering was done and added into the data base 
             I ran a multiple linear regression against the number of wins on a specific cluster based on the player statistics. In the drop down menu below you can select a year and course cluster. This will show how much the course cluster wins 
             were explained by player statistics. The bar graph will show which specific statistics hold the most weight for that year and course cluster. 
             
             ''')
    
    optimise_k_means(cluster_df[columns], 10)

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(cluster_df[columns])
    cluster_df['kmeans_2'] = kmeans.labels_

    visualize_clusters(principal_components, kmeans.labels_, 'KMeans Clustering with 2 clusters')
    box_plot_cluster()




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
    st.pyplot()
    

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




if __name__ == "__main__":
    main()
