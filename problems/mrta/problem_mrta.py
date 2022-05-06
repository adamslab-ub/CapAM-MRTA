from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.mrta.state_mrta import StateMRTA
from utils.beam_search import beam_search


class MRTA(object):

    NAME = 'mrta'  # Capacitated Vehicle Routing Problem

    # VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size, loc_vec_size = dataset['loc'].size()
        # print(batch_size, graph_size, loc_vec_size)
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"


        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        cost = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

        return cost

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MRTADataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMRTA.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = MRTA.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args, n_agents):
    depot, loc, deadline, *args = args
    initial_size = 100
    # n_agents = 100
    max_capacity = 10
    max_range = 4
    max_speed = 10
    enable_capacity_constraint =  False
    enable_range_constraint = True
    min_n_agents = 50
    max_n_agents = 400
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'deadline': torch.tensor(deadline, dtype=torch.float),
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'initial_size':initial_size,
        'max_n_agents': torch.tensor([[max_n_agents]]),
        'min_n_agents': torch.tensor([[min_n_agents]]),
        'n_agents':torch.tensor([[[n_agents]]]),
        'max_range':max_range,
        'max_capacity':max_capacity,
        'max_speed':max_speed,
        'enable_capacity_constraint':enable_capacity_constraint,
        'enable_range_constraint':enable_range_constraint

    }


class MRTADataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0,
                 n_depot = 1,
                 initial_size = None,
                 deadline_min = None,
                 deadline_max=None,
                 max_n_agents = 20,
                 min_n_agents = 5,
                 max_range = 4,
                 max_capacity = 10,
                 max_speed = 10,
                 n_agents = 5,
                 distribution=None):
        super(MRTADataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args, n_agents = n_agents) for args in data[offset:offset+num_samples]]

        else:

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(n_depot,2).uniform_(0, 1),
                    'deadline':torch.FloatTensor(size).uniform_(deadline_min,deadline_max),
                    'initial_size':initial_size,
                    'n_agents': torch.tensor([[torch.randint(min_n_agents, max_n_agents+1, (1, 1))]]),
                    'max_n_agents': torch.tensor([[max_n_agents]]),
                    'min_n_agents': torch.tensor([[min_n_agents]]),
                    'max_range':max_range,
                    'max_capacity':max_capacity,
                    'max_speed':max_speed,
                }
                for i in range(num_samples)
            ]


        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
