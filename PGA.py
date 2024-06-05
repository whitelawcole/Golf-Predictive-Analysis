import requests
from bs4 import BeautifulSoup
import sqlite3
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



# Scraping Data from PGAtour.com, Tournaments, Courses & Winners
years = list(range(2023,2012, -1))

# Obtain the Data from each Page for a specific Year
def get_page_data(year):
    url = f'https://www.pgatour.com/schedule/{year}'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
# Obtain The Winners into a list
    def get_winner():
        winner_list = []
        for player_link in soup.find_all('a', class_='chakra-link css-1jfg7sy'):
            player_href = player_link.get('href')
            if player_href and '/player' in player_href:
                player_name = player_link.text.strip()
                winner_list.append(player_name)
        del winner_list[0]
        return winner_list
    
    # Have to account for multiple winners
    def combined_winners():
        winners = get_winner()
        true_winners = []
        i = 0
    
        while i < len(winners):
            current_winner = winners[i]
        
            if current_winner.endswith(','):
                combined = [current_winner]
            
                # Continue combining until the next winner doesn't end with a comma
                while i + 1 < len(winners) and winners[i].endswith(','):
                    combined.append(winners[i + 1])
                    i += 1
            
                # Strip any trailing commas and spaces from the last element of combined
                true_winners.append(combined)
            else:
                # If the current winner does not end with a comma, append it as a single-element list
                true_winners.append([current_winner])
        
            i += 1
    
        return true_winners

# Create a list of tournaments
    def get_Tournament():
        tourney_names = []
        Tournaments = soup.find_all(class_='chakra-text css-vgdvwe')
        for Tournament in Tournaments:
            tourney_names.append(Tournament.text.strip())
        return tourney_names
    
# Create a list of courses
    def get_course():
        Courses = soup.find_all(class_='chakra-text css-16dpohb')
        Course_names = []
        for Course in Courses:
            Course_names.append(Course.text.strip())
        return Course_names
    
    def get_date():
        dates = soup.find_all(class_='chakra-text css-mi4td0')
        date_text = [date.text.strip()[0:3] for date in dates]
        return date_text
    
    def months_to_years(dates):
        # The page is in seasons (ex: 2021-2020), determine which year the tournamnent was played based off the first occurences of the month 
        replacement_year = f'{int(year) - 1}'
        replaced = []
        Years_list = []

        for i, month in enumerate(dates):
            if month == 'SEP' and month not in replaced:
                Years_list.append(replacement_year)
            elif 'SEP' not in replaced:
                replaced.append('SEP')
                Years_list.append(replacement_year)
            elif month == 'OCT' and month not in replaced:
                Years_list.append(replacement_year)
            elif 'OCT' not in replaced:
                replaced.append('OCT')
                Years_list.append(replacement_year)
            elif month == 'NOV' and month not in replaced:
                Years_list.append(replacement_year)
            elif 'NOV' not in replaced:
                replaced.append('NOV')
                Years_list.append(replacement_year)
            elif month == 'DEC' and month not in replaced:
                Years_list.append(replacement_year)
            elif 'DEC' not in replaced:
                replaced.append('DEC')
                Years_list.append(replacement_year)
            else:
                Years_list.append(year)
        return Years_list

# Organize the Data into a list of Dictionaries 
    def organize_data():
        Player_stats = []
        All_data = {}
        winners = combined_winners()
        tournaments = get_Tournament()
        courses = get_course()
        dates = get_date()
        years = months_to_years(dates)
        # Puerto Rico Open Charity Day never has a winner so remove from the list including the other indexes that correspond
        if 'Puerto Rico Open Charity Day' in tournaments:
            index = tournaments.index('Puerto Rico Open Charity Day')
            tournaments.remove('Puerto Rico Open Charity Day')
            courses.remove(courses[index])
            years.remove(years[index])

        min_length = min(len(winners), len(tournaments), len(courses), len(dates))

        for i in range(min_length):
            tournament_name = tournaments[i]
            winner_name = winners[i]
            course_name = courses[i]
            date_info = years[i]


            Player_stats.append(
                {
                    'Tournament': tournament_name,
                    'Winner': winner_name,
                    'Course': course_name,
                    'Year': date_info
                })
            
        return Player_stats
     

    return organize_data()



# Add Winner Data in SQL DB, do this before you add the rest of the data to account for abbreviated names
def add_winner_names(year):
    url = f'https://www.pgatour.com/schedule/{year}'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
# Add The Winners into a list
    winner_list = []
    for player_link in soup.find_all('a', class_='chakra-link css-1jfg7sy'):
        player_href = player_link.get('href')
        if player_href and '/player' in player_href:
            player_name = player_link.text.strip()
            if player_name.endswith(','):
                player_name = player_name.rstrip(',')
            winner_list.append(player_name)
    del winner_list[0]
    conn = sqlite3.connect('Golf_data.db')
    cursor = conn.cursor()
    # Special cases that are missed by the fuzz score, find these by running the add_winner function and cross checking with the db
    special_cases = ['Seung-Yul Noh', 'Kyoung-Hoon Lee', 'Ludvig Aberg', 'Sang-Moon Bae', 'Angel Cabrera']
    for special_case in special_cases:
        cursor.execute('''
                        SELECT id FROM Players WHERE name = ?
                    ''', (special_case,))
        player_id = cursor.fetchone()
        if player_id is not None:
            player_id = player_id[0]
            cursor.execute('''
            INSERT OR IGNORE INTO WINNERS (player_id, name) VALUES (?, ?)
            ''', (player_id, special_case))
            conn.commit()

    for name in winner_list:
        cursor.execute('''
            SELECT name, id FROM Players
        ''')
        players = cursor.fetchall()
        matched_name, score = process.extractOne(name, [player[0] for player in players], scorer=fuzz.token_sort_ratio)
        # Find the name that matches the best and use the ID, score ratio is set to 75 
        if score>= 75:
                player_id = next(player[1] for player in players if player[0] == matched_name)
                cursor.execute('''
                INSERT OR IGNORE INTO WINNERS (player_id, name) VALUES (?, ?)
            ''', (player_id, name ))
                conn.commit()
        else:
            print(f"Player '{name}' not found in database or doesn't match any special cases. Skipping...")
    conn.close()


# Add Tournament Data in SQL DB, do this before add the rest of the data in order to relate keys 
def add_tourney_names(year):
    url = f'https://www.pgatour.com/schedule/{year}'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    conn = sqlite3.connect('Golf_data.db')
    cursor = conn.cursor()
    Tournaments = soup.find_all(class_='chakra-text css-vgdvwe')
    for Tournament in Tournaments:
        name = Tournament.text.strip()
        # Do not add Puerto Rico Open Charity Day
        if name != 'Puerto Rico Open Charity Day':
            cursor.execute('''
                INSERT OR IGNORE INTO Tournmanents (name) VALUES (?)
            ''', (name,))
            conn.commit()

# Add the rest of the data 
def add_course_winner_data(year):
    data = get_page_data(year)
    conn = sqlite3.connect('Golf_data.db')
    cursor = conn.cursor()
    try:
        for entry in data:
            # Extract tournament details from dictionary
            tournament_name = entry['Tournament']
            course_name = entry['Course']
            data = entry['Year']
            for winner in entry['Winner']:
                if winner.endswith(','):
                    winner = winner.rstrip(',')
                cursor.execute('''
                SELECT id FROM Tournmanents WHERE name = ? 
                ''', (tournament_name,))
                tournament_row = cursor.fetchone()
                if tournament_row:
                    tournament_id = tournament_row[0]
                else:
                    # Insert new tournament and get its ID, this should not be the case because we used the same webpage 
                    cursor.execute('''
                        INSERT INTO Tournmanents (name)
                        VALUES (?)
                    ''', (tournament_name,))
                    tournament_id = cursor.lastrowid

                # Check if course exists and get its ID
                cursor.execute('''
                    SELECT course, id FROM "Course Stats" 
                ''')
                courses = cursor.fetchall()
                
                matched_name, score = process.extractOne(course_name, [course[0] for course in courses], scorer=fuzz.token_set_ratio)
                # Accounts for course names not being exactly the same as the names for courses in course stats table
                # Also does not add the data if there is no data for the course in the course stats table 
                if score >= 70:
                    course_id = next(course[1] for course in courses if course[0] == matched_name)
                    course_name = matched_name
                    cursor.execute('''
                    SELECT player_id FROM Winners WHERE name = ?
                    ''', (winner,))
                    winner_row = cursor.fetchone()
                    if winner_row:
                        winner_id = winner_row[0]
                        cursor.execute('''
                            INSERT INTO "Tournament & Course Winners" (Tournament_name, Tournament_id, Course_name, Course_id, Winner, Winner_id, Year)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (tournament_name, tournament_id, course_name, course_id, winner, winner_id, year))
                    # These elif statements will hardcode theses special cases where there is no matching name, these are the same as
                    # the special cases but reversed 
                    elif winner == 'K.H. Lee':
                        winner = 'Kyoung-Hoon Lee'
                        cursor.execute('''
                        SELECT player_id FROM Winners WHERE name = ?
                        ''', (winner,))
                        winner_row = cursor.fetchone()
                        winner_id = winner_row[0]
                        cursor.execute('''
                            INSERT INTO "Tournament & Course Winners" (Tournament_name, Tournament_id, Course_name, Course_id, Winner, Winner_id, Year)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (tournament_name, tournament_id, course_name, course_id, winner, winner_id, year))
                    elif winner == 'Sangmoon Bae':
                        winner = 'Sang-Moon Bae'
                        cursor.execute('''
                        SELECT player_id FROM Winners WHERE name = ?
                        ''', (winner,))
                        winner_row = cursor.fetchone()
                        winner_id = winner_row[0]
                        cursor.execute('''
                            INSERT INTO "Tournament & Course Winners" (Tournament_name, Tournament_id, Course_name, Course_id, Winner, Winner_id, Year)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (tournament_name, tournament_id, course_name, course_id, winner, winner_id, year))
                    elif winner == 'S.Y. Noh':
                        winner = 'Seung-Yul Noh'
                        cursor.execute('''
                        SELECT player_id FROM Winners WHERE name = ?
                        ''', (winner,))
                        winner_row = cursor.fetchone()
                        winner_id = winner_row[0]
                        cursor.execute('''
                            INSERT INTO "Tournament & Course Winners" (Tournament_name, Tournament_id, Course_name, Course_id, Winner, Winner_id, Year)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (tournament_name, tournament_id, course_name, course_id, winner, winner_id, year))
                    elif winner == 'Ángel Cabrera':
                        winner = 'Angel Cabrera'
                        cursor.execute('''
                        SELECT player_id FROM Winners WHERE name = ?
                        ''', (winner,))
                        winner_row = cursor.fetchone()
                        winner_id = winner_row[0]
                        cursor.execute('''
                            INSERT INTO "Tournament & Course Winners" (Tournament_name, Tournament_id, Course_name, Course_id, Winner, Winner_id, Year)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (tournament_name, tournament_id, course_name, course_id, winner, winner_id, year))
                    elif winner == 'Ludvig Åberg':
                        winner = 'Ludvig Aberg'
                        cursor.execute('''
                        SELECT player_id FROM Winners WHERE name = ?
                        ''', (winner,))
                        winner_row = cursor.fetchone()
                        winner_id = winner_row[0]
                        cursor.execute('''
                            INSERT INTO "Tournament & Course Winners" (Tournament_name, Tournament_id, Course_name, Course_id, Winner, Winner_id, Year)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (tournament_name, tournament_id, course_name, course_id, winner, winner_id, year))

                    else:
                        # find out which courses we do not have stats for 
                        print(f"Player '{winner}' not found in database. Skipping...") 
                        
                else:
                    # find out which players were not found
                    print(f"Course '{course_name}' not found in database. Skipping ...")


        # Commit changes
        conn.commit()
        print("Data inserted successfully!")

    except sqlite3.Error as e:
        print(f"Error inserting data: {e}")
    conn.close()

# Run this function first to find any names that dont match, if they are abbreviated input the real name into the special cases list in the add winners function.
# Also create a elif statment in the add course winner data with the real name, this ensures all names are the same throughout the db
# Some of the names the fuzz library wont catch if is abbreviated, but decreasing the ratio may corrupt the db 
def add_winner_data():
    for year in years:
            add_winner_names(str(year))
            time.sleep(10)



def add_rest():
    for year in years:
        add_tourney_names(str(year))
        time.sleep(10)
    for year in years:
        add_course_winner_data(str(year))
        time.sleep(10)

#add_winner_data()

# Run this function after you have ran the add_winner_data and checked the names 
add_rest()


