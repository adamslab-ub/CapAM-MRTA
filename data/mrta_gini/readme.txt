===================================================================
Test instances of the
Task Allocation Problem with Time and Capacity constraints (TAPTC)
January 2015.
===================================================================

1) Agent data :
---------------
In TAPTC, the agent data are seperate from the task data.
They are available in the "agent" folder, in text files (txt) with names
like "a2i01.txt", which is sample (or instance) number one of 2 agents.
The agent file header has the format: 
	n d d        % n: the number of agents, d: the map x-dimension and y-dimension 
eg. 2 100 100        % there 2 agents spread over 100 by 100 map.
The agent file data lines have the format:
	a x y p	     % a: the agent identifier number, (x,y): the agent initial location coordinate, p: the agent capacity.
eg. 0 57 56 2.0,
means that agent 0 is initially located at point (57,56) and has capacity 2.0 to perform a task.

2) Task data :
--------------
The task data files are divided into two groups, available in folders with their respective names:		  
Group1: 
tasks with deadlines drawn from a tight time range. 
This make agents have short-horizon schedules (small to moderate number of visits per tour).
Group2: 
tasks with deadlines drawn from a wider time range.
This make agents have long-horizon schedules (mederate to large number of visits per tour).
The task data files are named like: r12a3i00, 
where "r" means that task locations are randomly drawn,
 "1" is the group number, 
"2"/4*100% is the percentage of normally distributed deadlines,
 "3" is the number of agents and "00" is the instance (or sample) number.
The task file header has the following format:
	m d d	     % m: the total number of tasks.
eg. 100 100 100	     % 100 tasks should be done.
The task lines have the format:
	k x y w d    % k: task identifier, (x,k): task location, w: task workload, d: task deadline.
eg. 14 74 98 20 181 	 			
means that task 14 is located at (74,80), has 20 as workload and 181 as deadline.

NB: See the TAPTC paper for further explanation about how these data were generated. 

 
