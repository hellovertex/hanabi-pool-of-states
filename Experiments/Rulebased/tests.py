import cl2
import rulebased_agent as ra
from internal_agent import InternalAgent
from outer_agent import OuterAgent
from iggi_agent import IGGIAgent
from legal_random_agent import LegalRandomAgent
from flawed_agent import FlawedAgent
from piers_agent import PiersAgent
from van_den_bergh_agent import VanDenBerghAgent

AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}


def test_save_load():
    gen = cl2.StateActionCollector(AGENT_CLASSES, 2)
    states=[1,2,3,4,5,6,7,8,9]
    gen.save(states=states)
    print(gen.load())

def test_generate():
    gen = cl2.StateActionCollector(AGENT_CLASSES, 3)
    states, actions = gen.collect(num_states_to_collect=10)
    print(len(states))
test_generate()

