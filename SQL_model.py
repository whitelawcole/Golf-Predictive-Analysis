import sqlite3

# SQL model for golf_project db

conn = sqlite3.connect('Golf_data.db')
conn.execute('PRAGMA foreign_keys = ON;')
cursor = conn.cursor()

# Table for players
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Players (
    id INTEGER PRIMARY KEY,
    name UNIQUE,
    country TEXT, API_Key INTEGER)
    ''')



# Table for Tournamnents 
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Tournmanents (
    id INTEGER PRIMARY KEY UNIQUE,
    name UNIQUE)
''')

# Table for Tournamnents & Winners 
cursor.execute('''
    CREATE TABLE IF NOT EXISTS "Tournament & Course Winners" (
        id INTEGER PRIMARY KEY,
        Tournament_name TEXT,
        Tournament_id INTEGER,
        Course_name TEXT,
        Course_id INTEGER,
        Winner TEXT,
        Winner_id INTEGER,
        Year INTEGER,
        FOREIGN KEY(Tournament_id) REFERENCES Tournaments(id),
        FOREIGN KEY(Course_id) REFERENCES "Course Stats"(course_id),
        FOREIGN KEY(Winner_id) REFERENCES Winners(id)
    )
''')
# Table for Winners with foreign key to player table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Winners (
    player_id INTEGER PRIMARY KEY UNIQUE,
    name UNIQUE, FOREIGN KEY(player_id) REFERENCES Players(id) )
''')

# Table for Years
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Years (
    year UNIQUE PRIMARY KEY )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS "Player Stats" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            year TEXT,
            player_id INTEGER,
            events_played INTEGER,
            first_place INTEGER,
            second_place INTEGER,
            third_place INTEGER,
            top_10 INTEGER,
            top_25 INTEGER,
            cuts INTEGER,
            cuts_made INTEGER,
            withdrawals INTEGER,
            points REAL,
            points_rank INTEGER,
            earnings REAL,
            earnings_rank INTEGER,
            drive_avg REAL,
            drive_acc REAL,
            gir_pct REAL,
            putt_avg REAL,
            sand_saves_pct REAL,
            birdies_per_round REAL,
            holes_per_eagle REAL,
            world_rank INTEGER,
            strokes_gained REAL,
            hole_proximity_avg TEXT,
            scrambling_pct REAL,
            scoring_avg REAL,
            strokes_gained_tee_green REAL,
            strokes_gained_total REAL,
            total_driving INTEGER,
            FOREIGN KEY(player_id) REFERENCES Players(id),
            FOREIGN KEY(year) REFERENCES Years
        )
    ''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS "Course Stats" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course UNIQUE,
        par INTEGER,
        yardage INTEGER,
        yardage_4_5 INTEGER,
        yardage_3 INTEGER,
        adj_score_to_par REAL,
        adj_par_3_score REAL,
        adj_par_4_score REAL,
        adj_par_5_score REAL,
        adj_driving_distance REAL,
        adj_sd_distance INTEGER,
        adj_driving_accuracy REAL,
        putt_sg REAL,
        arg_sg REAL,
        app_sg REAL,
        ott_sg REAL,
        fw_width REAL,
        miss_fw_pen_frac REAL,
        adj_gir REAL,
        less_150_sg REAL,
        greater_150_sg REAL,
        arg_fairway_sg REAL,
        arg_rough_sg REAL,
        arg_bunker_sg REAL,
        less_5_ft_sg REAL,
        greater_5_less_15_sg REAL,
        greater_15_sg REAL
    )
''')


# Insert Years Table, The year will be the primary key
years = list(range(2023,2012, -1))

for year in years:
    conn.execute(
        '''INSERT OR IGNORE INTO Years (year) VALUES (?)''', (year,)
    )


conn.commit()
conn.close()