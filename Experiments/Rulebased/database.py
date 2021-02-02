# layout will be
# | num_players | agent | turn | state | action | team |
# for the state action table and
# | num_players | turn | state | team |
# for the state table
# todo maybe we leave the state table out for now,
#  because we can deduce it from the complete state action table
import sqlite3
import pickle

def create_connection(path):
    """ create a database connection to the SQLite database
            specified by path
        :param path: database file
        :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(str(path))
        return conn
    except sqlite3.Error as e:
        print(e)
    return None


def insert_data(conn, replay_dictionary, with_obs_dict):
    if with_obs_dict:
        insert_state_dict_data(conn, replay_dictionary)
    else:
        insert_vectorized_data(conn, replay_dictionary)


def insert_state_dict_data(conn, replay_dictionary):
    """
        # current_player: 0
        # current_player_offset: 0
        # deck_size: 40
        # discard_pile: []
        # fireworks: {}
        # information_tokens: 8
        # legal_moves: [{}, ..., {}]
        # life_tokens: 3
        # observed_hands: [[{},...,{}], ..., [{},...,{}]]
        # card_knowledge BLUB
        # pyhanabi BLUB of the BLUBS

    """
    # todo pickle pyhanabi data as in
    #  https://stackoverflow.com/questions/198692/can-i-pickle-a-python-dictionary-into-a-sqlite3-text-field
    cursor = conn.cursor()

    # create table if it does not exist already
    # | num_players | agent | turn | state | action | team |
    cursor.execute(''' CREATE TABLE IF NOT EXISTS pool_of_state_dicts( 
        num_players INTEGER , 
        agent TEXT, 
        turn INTEGER,  
        int_action INTEGER,
        dict_action TEXT, 
        team TEXT, 
        current_player INTEGER, 
        current_player_offset INTEGER, 
        deck_size INTEGER, 
        discard_pile TEXT, 
        fireworks TEXT, 
        information_tokens INTEGER, 
        legal_moves TEXT, 
        life_tokens INTEGER, 
        observed_hands TEXT,
        card_knowledge TEXT,
        vectorized TEXT,
        pyhanabi BLOB
        )''')
    # conn.commit()

    num_players = replay_dictionary.pop('num_players')
    team = str(replay_dictionary.pop('team'))
    values = []
    for agent in replay_dictionary.keys():
      if agent not in ['states', 'actions', 'obs_dicts']:
        num_transitions = len(replay_dictionary[agent]['turns'])

        for i in range(num_transitions):
            # | num_players | agent | turn | state | action | team |
            # obs = replay_dictionary[agent]['obs_dict'][i]
            # pyhanabi = replay_dictionary[agent]['obs_dicts'][i]
            obs_pyhanabi = replay_dictionary[agent]['obs_dicts'][i]['pyhanabi']
            # pyhanabi = replay_dictionary[agent]['obs_dicts'][i]['pyhanabi']
            pyhanabi = pickle.dumps(obs_pyhanabi, pickle.HIGHEST_PROTOCOL)

            row = (num_players,
                   agent,
                   replay_dictionary[agent]['turns'][i],
                   int(replay_dictionary[agent]['int_actions'][i]),
                   str(replay_dictionary[agent]['dict_actions'][i]),
                   team,
                   # parse observation_dictionary
                   replay_dictionary[agent]['obs_dicts'][i]['current_player'],
                   replay_dictionary[agent]['obs_dicts'][i]['current_player_offset'],
                   replay_dictionary[agent]['obs_dicts'][i]['deck_size'],
                   str(replay_dictionary[agent]['obs_dicts'][i]['discard_pile']),
                   str(replay_dictionary[agent]['obs_dicts'][i]['fireworks']),
                   replay_dictionary[agent]['obs_dicts'][i]['information_tokens'],
                   str(replay_dictionary[agent]['obs_dicts'][i]['legal_moves']),
                   replay_dictionary[agent]['obs_dicts'][i]['life_tokens'],
                   str(replay_dictionary[agent]['obs_dicts'][i]['observed_hands']),
                   str(replay_dictionary[agent]['obs_dicts'][i]['card_knowledge']),
                   str(replay_dictionary[agent]['obs_dicts'][i]['vectorized']),
                   # sqlite3.Binary(b'pyhanabi_goes_here')  # sqlite3.Binary(pyhanabi)
                   sqlite3.Binary(pyhanabi)
                   )
            assert type(row[3]) == int, f'action was {row[3]} with type={type(row[3])}'


            values.append(row)

    cursor.executemany('INSERT INTO pool_of_state_dicts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', values)
    conn.commit()


def insert_vectorized_data(conn, replay_dictionary):
    """
    CREATE TABLE IF NOT EXISTS info (PRIMARY KEY id int, username text, password text)
    replay_dictionary example:
    dict_keys(['OuterAgent', 'FlawedAgent', 'IGGIAgent', 'team', 'num_players'])
    replay_dictionary['OuterAgent'] : dict_keys(['states', 'actions', 'turns'])
    - replay_dictionary['OuterAgent']['states']: List[List[int]]
    - replay_dictionary['OuterAgent']['actions']: List[int]
    - replay_dictionary['OuterAgent']['turns']: List[int]
    """
    raise NotImplementedError
    # cursor = conn.cursor()
    # # create table if it does not exist already
    # # | num_players | agent | turn | state | action | team |
    # cursor.execute(''' CREATE TABLE IF NOT EXISTS pool_of_states(
    # num_players INTEGER , agent TEXT, turn INTEGER, state TEXT, action INTEGER, team TEXT)''')
    # # conn.commit()
    #
    # num_players = replay_dictionary.pop('num_players')
    # team = str(replay_dictionary.pop('team'))
    # values = []
    # for agent in replay_dictionary.keys():
    #     num_transitions = len(replay_dictionary[agent]['turns'])
    #     for i in range(num_transitions):
    #         # | num_players | agent | turn | state | action | team |
    #         row = (num_players,
    #                agent,
    #                replay_dictionary[agent]['turns'][i],
    #                str(replay_dictionary[agent]['states'][i]),
    #                replay_dictionary[agent]['actions'][i],
    #                team
    #                )
    #         assert type(row[4]) == int, f'action was {row[4]}'
    #         assert 0 <= row[4] <= 50, f'action was {row[4]}'
    #         values.append(row)
    # cursor.executemany('INSERT INTO pool_of_states VALUES (?,?,?,?,?,?)', values)
    # conn.commit()
