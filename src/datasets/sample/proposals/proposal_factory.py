from .energy_detection import EnergyDetection
from .selective_search import SelectiveSearch


proposal_factory = {
    "ss": SelectiveSearch,
    "ed": EnergyDetection
}

def get_proposal(strategy, max_prop=30):
    proposal = None

    prop_type = strategy[:strategy.find('_')] if '_' in strategy else "ss"
    strategy_type = strategy[strategy.find('_') + 1:] if '_' in strategy else strategy

    if strategy_type == 'topk':
        proposal = proposal_factory[prop_type](max_prop)
    else:
        raise ValueError("No such strategy")

    return proposal