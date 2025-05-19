import pandas as pd
import numpy as np
import re
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from xgboost import plot_importance


# Load data

# Change the year based on what year you are back-testing for
years = list(range(2022,2014,-1)) # Change the year to the year before predict year
predict_year = 2023
stats = pd.read_excel('All_stats.xlsx')
wins = pd.read_excel('Top10_Final.xlsx')

import re
import numpy as np


# This function is to fix fomatting for Hole Proximity Average Statistic 
def parse_distance(distance_str):
    """
    Parse a distance string in the format "X' Y\"" into total inches.

    Args:
        distance_str (str): The distance string (e.g., "37' 10\"").
    
    Returns:
        float: The distance in inches or np.nan if invalid.
    """
    # Check if distance_str is empty or None
    if not distance_str:
        return np.nan
    
    # Use regex to capture feet and inches
    match = re.match(r"(\d+)' ?(\d+)\"", distance_str)
    
    if match:
        feet = int(match.group(1))  # Get feet
        inches = int(match.group(2))  # Get inches
        return feet * 12 + inches  # Convert to total inches
    else:
        return np.nan  # If format is invalid, return NaN
    



def get_prob(cluster):
    player_data_list = [] 
    for year in years:
        cluster_finishers = wins[(wins['cluster'] == cluster) & (wins['Year'] == year)]
        cluster_players = cluster_finishers['Player'].unique() 
        for player in cluster_players:
            player_stats = stats[(stats['name'] == player) & (stats['year'] == year)]
            
            # Skip if player_stats is empty
            if player_stats.empty:
                continue

            
            # Convert 'hole_proximity_avg' if parse_distance is defined
            if 'hole_proximity_avg' in player_stats.columns:
                player_stats = player_stats.copy()
                player_stats['hole_proximity_avg'] = player_stats['hole_proximity_avg'].apply(parse_distance)
                
            # Count the number of wins for each player in the cluster
            cluster_wins = len(cluster_finishers[cluster_finishers['Player'] == player])
            
            # Append aggregated stats to player_data_list
            player_data_list.append({
                'name': player,
                'drive_avg': player_stats['drive_avg'].iloc[0],
                'drive_acc': player_stats['drive_acc'].iloc[0],
                'gir_pct': player_stats['gir_pct'].iloc[0],
                'putt_avg': player_stats['putt_avg'].iloc[0],
                'sand_saves_pct': player_stats['sand_saves_pct'].iloc[0],
                'birdies_per_round': player_stats['birdies_per_round'].iloc[0],
                'holes_per_eagle': player_stats['holes_per_eagle'].iloc[0],
                'hole_proximity_avg': player_stats['hole_proximity_avg'].iloc[0],
                'scrambling_pct': player_stats['scrambling_pct'].iloc[0],
                'total_driving': player_stats['total_driving'].iloc[0],
                'strokes_gained_put': player_stats['strokes_gained_put'].iloc[0],
                'strokes_gained_tee_green': player_stats['strokes_gained_tee_green'].iloc[0],
                'scoring_avg': player_stats['scoring_avg'].iloc[0],
                'strokes_gained_total': player_stats['strokes_gained_total'].iloc[0],

                'cluster_wins': cluster_wins
            })
            
        # Create a DataFrame from the list of player data  
            

        
    df = pd.DataFrame(data=player_data_list)
    df = df.dropna()
    conditions = [
        df['cluster_wins'] >= 3,
        df['cluster_wins'] < 3
    ]

    # Define the corresponding classes
    choices = [0, 1]

    # Create a new column 'cluster_class' based on these conditions
    df['cluster_class'] = np.select(conditions, choices)

    X = df[['drive_avg', 'drive_acc', 'gir_pct', 'putt_avg', 'sand_saves_pct',
                    'birdies_per_round', 'strokes_gained_total', 'hole_proximity_avg', 'scrambling_pct', 'strokes_gained_put', 'holes_per_eagle', 'strokes_gained_tee_green', 'scoring_avg']]
    y = df[['cluster_class']] 


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set Model Hyperparameters

    model = XGBClassifier(
        eval_metric='mlogloss',
        subsample=0.9,
        n_estimators=2000,
        max_depth=3,
        learning_rate=0.01,
        colsample_bytree=0.9,
    )

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions and evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Final Model Accuracy:", accuracy)
    feature_important = model.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    import matplotlib.pyplot as plt

    plt.barh(keys, values)
    plt.title(f'Feature Importance For Cluster {cluster}')
    plt.show()
    # Plot feature importance
    plot_importance(model)
    plt.title(f'Feature Importance Cluster {cluster}')
    plt.show()
    # Check for missing values in features and target variable
    print("Missing values in features:\n", X.isnull().sum())
    print("Missing values in target:\n", y.isnull().sum())

    # Classification report for precision, recall, and F1-score
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix to see prediction distribution across classes
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    # Plotting the confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(f"Confusion Matrix: Course Cluster {cluster}")
    plt.show()


    # Generate the classification report as a dictionary and convert to DataFrame
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Plotting the classification report heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Classification Report Heatmap: Course Cluster {cluster}")
    plt.show()



    # Get predict year stats
    stats_df = pd.read_excel('All_stats.xlsx')
    stats_df = stats_df[stats_df['year'] == predict_year]
    Names = stats_df['name'].tolist()
    Roster_df = stats_df[stats_df['name'].isin(Names)]
    missed_stats = list(set(Names) - set(stats_df['name']))
    Roster_df = Roster_df.dropna()
    Roster_df['hole_proximity_avg'] = Roster_df['hole_proximity_avg'].apply(parse_distance)
    Roster_stats = Roster_df.copy()




    comp_stats = Roster_df[['drive_avg', 'drive_acc', 'gir_pct', 'putt_avg', 'sand_saves_pct',
                    'birdies_per_round', 'strokes_gained_total', 'hole_proximity_avg', 'scrambling_pct', 'strokes_gained_put', 'holes_per_eagle', 'strokes_gained_tee_green', 'scoring_avg']]

    probabilities = model.predict_proba(comp_stats)

    # Add the probability of class 0 to the DataFrame
    Roster_df[f'cluster{cluster}_prob'] = probabilities[:, 0]
    
            



    for name in missed_stats:
        print(f"no data was founf for {name}")

    top_25_probs = Roster_df[['name', f'cluster{cluster}_prob']].sort_values(by=f'cluster{cluster}_prob', ascending=False)
    return top_25_probs


cluster_0 = get_prob(0)
cluster_1 = get_prob(1)

# Merge both cluster probabilities and calculate average
merged = cluster_0.merge(cluster_1, on='name')
merged['final_probability'] = (merged['cluster0_prob'] + merged['cluster1_prob']) / 2

top_probs = merged[['name', 'final_probability']].sort_values(by='final_probability', ascending=False)
top_probs['final_probability'] = top_probs['final_probability'] * 100

print(top_probs.head(25))


wins = wins[wins['Year'] == predict_year]

# Get the name counts for wins
name_counts = wins['Player'].value_counts()
print(name_counts.head(15))

# Determine the cutoff for the top 10, including ties
top_10_threshold = name_counts.iloc[9]  # The count of the 10th most frequent name

# Filter names with counts greater than or equal to the threshold
top_names = name_counts[name_counts >= top_10_threshold]

# Display the results
print(top_names)

# Get the predicted top names
predicted_names = set(top_probs['name'].head(10))

# Get the actual top names from the "top_names" Series
actual_top_names = set(top_names.index)

# Calculate the intersection of predicted and actual top names
correct_predictions = predicted_names.intersection(actual_top_names)

# Calculate accuracy
accuracy = len(correct_predictions) / len(predicted_names)

# Display results
print(f"Correctly Predicted Names: {correct_predictions}")
print(f"Accuracy: {accuracy:.2%}")

# Prepare predicted DataFrame with actual and wins information
predicted_df = top_probs[['name', 'final_probability']].head(10)
predicted_df['actual'] = predicted_df['name'].apply(lambda x: 1 if x in actual_top_names else 0)
predicted_df['wins'] = predicted_df['name'].map(name_counts)  # Map the wins to each player

# Fill NaN wins with 0 for players not in the actual wins list
predicted_df['wins'] = predicted_df['wins'].fillna(0).astype(int)

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))

# Plot predicted probabilities
ax.bar(predicted_df['name'], predicted_df['final_probability'], label='Predicted Probability', alpha=0.7, color='blue')

# Plot actual winners as binary values
for i, (actual, wins) in enumerate(zip(predicted_df['actual'], predicted_df['wins'])):
    # Annotate if the player is an actual winner
    ax.text(i, predicted_df['final_probability'].iloc[i] + 0.02, 'Winner' if actual else '', ha='center', color='red')
    
    # Annotate the number of wins
    ax.text(i, predicted_df['final_probability'].iloc[i] / 2, f'{wins} Wins', ha='center', color='black', fontsize=10)

# Adding labels and legend
ax.set_title(f'Model Predictions vs Actual Winners and Number of Wins: {predict_year}')
ax.set_ylabel('Probability (%)')
ax.set_xlabel('Predicted Players with the Most Top 10 Finishes')
plt.xticks(rotation=45)
plt.legend(['Predicted Probability'], loc='upper right')

plt.tight_layout()
plt.show()
