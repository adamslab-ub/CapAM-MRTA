import torch
from typing import NamedTuple
import numpy as np
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateMRTA(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc (coordinates of all the locations including the depot)
    distance_matrix: torch.Tensor # distance matrix for all the coordinates
    time_matrix: torch.Tensor # time matrix between all the coordinates using the speed of the agents
    deadline: torch.Tensor # deadline for all the tasks (special case for the depot, keep a very large time)
    # demand: torch.Tensor # we do not need this, we can remove this or set the demand as 1 or something equal to the quantity for 1 time delivery
    tasks_finish_time: torch.Tensor
    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows (this is basically the ids for all the location, which are considered as integers)



    # robot specific
    robots_initial_decision_sequence: torch.Tensor # for timestep 1, all robots need decision, so we set a sequence for this
    robots_task_done_success: torch.Tensor # for each robot, this variable tracks the id of the task done successfully
    robots_task_missed_deadline: torch.Tensor # for each robot, this variable tracks the id of the task with missed deadline
    robots_task_visited: torch.Tensor # keeps track of all the nodes visited by all the robots (successful or not)
    robots_distance_travelled: torch.Tensor # keeps track of the total distance travelled by the robots (this is updated everytime a new decision is made and also during the end of the simulation)
    # robots_total_tasks_done # optional
    robots_next_decision_time: torch.Tensor # tracks the next decision time for all the robots
    robots_range_remaining: torch.Tensor # tracks of the range remaining for all the robots
    robots_capacity: torch.Tensor # keeps track of the capacity of the robot
    robots_current_destination: torch.Tensor
    robots_start_point: torch.Tensor
    robot_taking_decision_range: torch.Tensor


    #general - frequent changing variable
    current_time: torch.Tensor # stores the value for the current time
    robot_taking_decision: torch.Tensor  # stores the id of the robot which will take the next decision
    next_decision_time: torch.Tensor # time at which the next decision is made. (0 t begin with)
    previous_decision_time: torch.Tensor # time at which the previous decision was made


    # for performance tracking
    tasks_done_success: torch.Tensor # keeps track of all the task id which was done successfully
    tasks_missed_deadline: torch.Tensor # keeps track of all the tasks which are visited but the deadline was missed
    tasks_visited: torch.Tensor # keeps track of all the tasks which are visited (successful or not)
    depot: torch.Tensor



    #end
    is_done: torch.Tensor


    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    # VEHICLE_CAPACITY = 1.0  # Hardcoded

    n_agents : torch.Tensor
    max_range :torch.Tensor
    max_capacity : torch.Tensor
    max_speed : torch.Tensor
    n_nodes: torch.Tensor
    initial_size: torch.Tensor
    n_depot: torch.Tensor



    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_[:,:,self.n_depot:]
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StateMRTA, self).__getitem__(key)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input,
                   visited_dtype=torch.uint8):
        # input = input_data['data']
        depot = input['depot']
        loc = input['loc']
        max_speed = input['max_speed'][0].item()
        coords = torch.cat((depot[:, :], loc), -2).to(device=loc.device)
        distance_matrix = (coords[:, :, None, :] - coords[:, None, :, :]).norm(p=2, dim=-1).to(device=loc.device)
        time_matrix = torch.mul(distance_matrix, (1/max_speed)).to(device=loc.device)
        deadline = input['deadline'] #+ torch.tensor(np.random.normal(loc=0, scale=0, size=input['deadline'].size()))
        n_agents = input['n_agents'].reshape(-1)[:, None]
        max_n_agents = input['max_n_agents'][0, 0, 0].item()
        max_range = input['max_range'][0].item()
        max_capacity = input['max_capacity'][0].item()
        max_speed = input['max_speed'][0].item()
        initial_size = input['initial_size'][0].item()
        batch_size, n_loc, _ = loc.size()
        n_depot = input['depot'].size()[1]
        robots_initial_decision_sequence = torch.from_numpy(np.arange(0, max_n_agents)).expand(
            (batch_size, max_n_agents)).to(device=loc.device)


        # n_nodes = torch.tensor(loc.size()[1], dtype=torch.uint8)

        return StateMRTA(
            coords=coords,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            robots_initial_decision_sequence = torch.from_numpy(np.arange(0, max_n_agents)).to(device=loc.device),#torch.from_numpy(np.arange(0, n_agents)),
            robots_task_done_success = torch.zeros((batch_size, max_n_agents), dtype=torch.int64, device=loc.device),
            robots_task_missed_deadline = torch.zeros((batch_size, max_n_agents), dtype=torch.int64, device=loc.device),
            robots_task_visited = torch.zeros((batch_size, max_n_agents), dtype=torch.int64, device=loc.device),
            robots_distance_travelled  = torch.zeros((batch_size, max_n_agents), dtype=torch.float, device=loc.device),
            robots_next_decision_time =  ((robots_initial_decision_sequence > (n_agents - 1)).to(torch.float) * 10000).to(device=loc.device),
            robots_range_remaining = torch.mul(torch.ones((batch_size,max_n_agents), dtype=torch.float, device=loc.device), max_range),
            robots_capacity = torch.mul(torch.ones((batch_size,max_n_agents), dtype=torch.float, device=loc.device), max_capacity),
            current_time = torch.zeros((batch_size, 1), dtype=torch.float, device=loc.device),
            robot_taking_decision = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            next_decision_time = torch.zeros((batch_size, 1), dtype=torch.float, device=loc.device),
            previous_decision_time = torch.zeros((batch_size, 1), dtype=torch.float, device=loc.device),
            tasks_done_success = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            tasks_missed_deadline = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            tasks_visited = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            is_done = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            distance_matrix = distance_matrix,
            time_matrix = time_matrix,
            deadline = deadline,
            tasks_finish_time=torch.zeros((batch_size, n_loc), dtype=torch.float, device=loc.device),
            robots_current_destination = torch.zeros((batch_size, max_n_agents), dtype=torch.int64, device=loc.device),
            robots_start_point = torch.zeros((batch_size, max_n_agents), dtype=torch.int64, device=loc.device),
            robot_taking_decision_range = torch.mul(torch.ones(batch_size, 1, dtype=torch.float, device=loc.device), max_range),
            depot = torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            max_capacity = max_capacity,
            n_agents = n_agents,
            max_range = max_range,
            n_nodes = input['loc'].size()[1],
            initial_size = initial_size,
            n_depot=n_depot,
            max_speed = max_speed,
        )


    def update(self, selected):
        # print('************** New decision **************')
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        previous_time = self.current_time

        current_time = self.next_decision_time


        #update mileage
        robots_range_remaining = self.robots_range_remaining
        robot_taking_decision = self.robot_taking_decision



        cur_coords = self.coords[self.ids, self.robots_current_destination[self.ids, self.robot_taking_decision]]

        time = self.time_matrix[self.ids, self.robots_current_destination[self.ids,self.robot_taking_decision[:]], selected]
        self.robots_next_decision_time[self.ids, self.robot_taking_decision] += time
        self.robots_distance_travelled[self.ids, self.robot_taking_decision] += self.distance_matrix[
            self.ids, self.robots_current_destination[self.ids, robot_taking_decision], selected]

        zero_indices = torch.nonzero(selected[:,0] ==0)
        if zero_indices.size()[0] > 0:
            self.robots_capacity[zero_indices[:,0], self.robot_taking_decision[zero_indices[:,0]].view(-1)]= self.max_capacity
            robots_range_remaining[zero_indices[:, 0], robot_taking_decision[zero_indices[:, 0]].view(-1)] = self.max_range

        non_zero_indices = torch.nonzero(selected)
        if non_zero_indices.size()[0] > 0:
            deadlines = self.deadline[self.ids.view(-1), selected.view(-1) - 1]
            dest_time = self.robots_next_decision_time[self.ids.view(-1), self.robot_taking_decision[self.ids].view(-1)]
            self.tasks_finish_time[self.ids, selected - 1] = dest_time[:, None]
            feas_ids = (deadlines > dest_time).nonzero()
            combined = torch.cat((non_zero_indices[:,0], feas_ids[:,0]))
            uniques, counts = combined.unique(return_counts=True)
            # difference = uniques[counts == 1]
            intersection = uniques[counts > 1]
            distance_new = self.distance_matrix[non_zero_indices[:, 0], self.robots_current_destination[non_zero_indices[:, 0], robot_taking_decision[non_zero_indices[:, 0]].view(-1)].view(-1), selected[non_zero_indices[:, 0]].view(-1)]
            robots_range_remaining[non_zero_indices[:, 0], robot_taking_decision[non_zero_indices[:, 0]].view(-1)] -= distance_new
            if intersection.size()[0] > 0:
                self.tasks_done_success[intersection] +=1
            self.tasks_visited[non_zero_indices[:,0]] += 1

        self.robots_start_point[self.ids, self.robot_taking_decision] = self.robots_current_destination[
            self.ids, self.robot_taking_decision]
        self.robots_current_destination[self.ids, self.robot_taking_decision] = selected

        sorted_time, indices = torch.sort(self.robots_next_decision_time)

        robot_taking_decision_range = robots_range_remaining[self.ids, indices[self.ids, 0]]

        if self.visited_.dtype == torch.uint8:

            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        new_cur_coord = self.coords[self.ids, selected]

        lengths = self.lengths + (new_cur_coord - cur_coords).norm(p=2, dim=-1)
        visited_[:,:,0] = 0


        # print('visited: ', visited_[0])
        # print('***************** End of decision making process*******')
        return self._replace(
            prev_a=prev_a, previous_decision_time = previous_time, current_time = current_time,
            robots_range_remaining = robots_range_remaining, robot_taking_decision = indices[self.ids,0],
            next_decision_time = sorted_time[self.ids,0],
            robot_taking_decision_range = robot_taking_decision_range,
            visited_=visited_,
            lengths=lengths, cur_coord=new_cur_coord,
            i=self.i + 1
        )


    def all_finished(self):
        # return self.i.item() >= self.demand.size(-1) and self.visited.all()
        return self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.robots_current_destination[self.ids, self.robot_taking_decision] #self.prev ## this has been changed

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_)

        mask_loc = visited_loc.to(torch.bool)  # | exceeds_cap

        robot_taking_decision = self.robot_taking_decision

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.robots_current_destination[self.ids, robot_taking_decision] == 0) & (
                    (mask_loc == 0).int().sum(-1) > 0)
        full_mask = torch.cat((mask_depot[:, :, None], mask_loc), -1)
        robot_taking_decision = self.robot_taking_decision
        capacity = self.robots_capacity[self.ids, robot_taking_decision]
        zero_capacity_ind = (capacity[:, 0] < 1).nonzero()

        if zero_capacity_ind.size()[0] > 0:
            full_mask[zero_capacity_ind[:, 0], :, 1:] = True

        non_zero_capacity_ind = (capacity[:, 0] > 0).nonzero()
        if non_zero_capacity_ind.size()[0] > 0:
            robot_dest = self.robots_current_destination[self.ids[:, 0], robot_taking_decision[self.ids[:, 0]].view(-1)]
            non_zero_robot_dest = (robot_dest != 0).nonzero()
            combined = torch.cat((non_zero_capacity_ind[:, 0], non_zero_robot_dest[:, 0]))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            if intersection.size()[0] > 0:
                avail_range = self.robots_range_remaining[
                    intersection, robot_taking_decision[intersection].view(-1)]
                d1 = self.distance_matrix[intersection, robot_dest[intersection].view(-1)]
                d2 = self.distance_matrix[intersection, 0]
                avail_range_expand = avail_range.T.expand(self.n_nodes + 1, avail_range.size()[0]).T
                set_true = full_mask[intersection].squeeze(1) | (avail_range_expand < d1 + d2)
                full_mask[intersection, :, 1:] = set_true[:, None, 1:]
        return full_mask

    def construct_solutions(self, actions):
        return actions
