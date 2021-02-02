from data import StateActionWriter
from cl2 import AGENT_CLASSES
import train_from_database as util
players = 3
env_config = {
            "colors":
                3,
            "ranks":
                5,
            "players":
                players,
            "hand_size":
                4 if players in [4, 5] else 5,
            # hand size is derived from number of players
            "max_information_tokens":
                8,
            "max_life_tokens":
                3,
            "observation_type":
                1  # pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        } 
# can remove stringify, if states from mixed game configs wanted in one database
path_to_db = f'./database_{util.stringify_env_config(env_config)}.db'  
num_rows_to_add = int(1e3)


if __name__ == '__main__':
    """ Create and Fill database """
    writer = StateActionWriter(AGENT_CLASSES, env_config, 3)
    writer.collect_and_write_to_database(path_to_db, num_rows_to_add)

    """ Get torch dataset with 'size' many elements, randomly drawn from database  """
    stateloader = util.MapStylePoolOfStatesFromDatabase(from_db_path=path_to_db, size=int(2e3)).get_eagerly(
           pyhanabi_as_bytes=True, batch_length=1, random_seed=42)
    for batch in stateloader:
        # batch of dictionaries, each dict contains all players information
        print(batch)
        # element in batch
        print(batch[0])
        # vectorized observation
        print(batch[0]['vectorized'])
        break