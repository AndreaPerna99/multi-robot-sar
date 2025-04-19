# Multi-Agent System 2023/24, Master's Degree in Artificial Intelligence
# Distributed Multi-Robot Search and Rescue Operations Using Aggregative Optimization in a Multi-Room House Environment
# University of Bologna
# Main Project File
# Andrea Perna, andrea.perna3@studio.unibo.it, Automation Engineering

#############################################################################
############################### Libraries ###################################
#############################################################################

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.random.seed(0)
import time
import random
import os
import yaml

#############################################################################
################################# Credits ###################################
#############################################################################

def message_handler(mode): #hub for messages' handling

    """
    This function prints of the terminal messages related to the program state, according
    to the specific part of the process specified by the input parameter mode.
    """

    #initial plots
    if mode == "title":

        print("\n\n----------------------------------------------------------------------------------------------------------------------")
        print(f"Multi-Agent System (MAS) Project - 2023/24")
        print("Distributed Multi-Robot Search and Rescue Operations Using Aggregative Optimization in a Multi-Room House Environment")
        print("Automation Engineering - University of Bologna")
        print("Andrea Perna")
        print("All rights reserved")
        print("----------------------------------------------------------------------------------------------------------------------")
        print("\n")
        time.sleep(3) #wait few seconds

    #ending plots
    elif mode == "end":

        print("\n\n---------------------------------------------------------------")
        print(f"End of the SAR simulation.")
        print("---------------------------------------------------------------")

#simulation's reproducibility
reproducibility = True 
if reproducibility: np.random.seed(0)

#############################################################################
############################## External Files ###############################
#############################################################################

#define the yaml file for parameters' extraction
yaml_file_name = 'Andrea_Perna3_MAS_Parameters.yaml'

#get the current directory of the script
current_folder_path = os.path.dirname(os.path.realpath(__file__))

#define the yaml file's path
yaml_file_path = os.path.join(current_folder_path, yaml_file_name)

#define the txt file's path
rooms_name_txt = 'Andrea_Perna3_MAS_Rooms.txt'
rooms_path_txt = os.path.join(current_folder_path, rooms_name_txt) #.txt file's path

#############################################################################
############################### Environment Class ###########################
#############################################################################

class Environment:

    #######################################################
    ##################### Initialization ##################
    #######################################################

    def __init__(self):

        """
        Initializes the environment by creating walls, rooms, and generating
        obstacle points for each room and corridor.
        """

        #read parameters from yaml file
        self.load_yaml_parameters()
        
        #build the environment
        self.build_house()

        #create heatmap for people detection (helicopter)
        self.heatmap_generation()

        #spawn survivors according to heatmap
        self.people_positions = self.survivors_initialization()

        #generate obstacle points
        self.generate_obstacle_points()
        
        #initialize the ambulance
        self.ambulance = self.Ambulance(environment=self, position=self.EXIT, params=self.ambulance_params)

        #plot the environment
        self.plot_house()
    
    def load_yaml_parameters(self):

        """
        Loads environment parameters from a YAML file.
        """

        #open the file to extract parameters
        if os.path.exists(yaml_file_path):

            #load yaml file
            with open(yaml_file_path, 'r') as file: yaml_data = yaml.safe_load(file)

            #extract house's parameters
            self.ENTRANCE = yaml_data['ENTRANCE']
            self.EXIT = yaml_data['EXIT']
            self.house_gain = yaml_data['house_gain']
            
            #extract survivor's parameters
            self.n_people = yaml_data['n_survivors']
            self.people_spread = yaml_data['survivors_spread']
            self.person_step_size = yaml_data['person_step_size']
            self.escorted_radius = yaml_data['escorted_radius']

            #extract heatmap's parameters
            self.show_heatmap = yaml_data['show_heatmap']
            self.heatmap_factor = yaml_data['heatmap_factor']

            #extract obstacle's parameters
            self.plot_margin = yaml_data['plot_margin']
            self.barrier_step = yaml_data['barrier_step']
            self.opening_width = yaml_data['opening_width']
            self.show_obstacle_points = yaml_data['show_obstacle_points']

            #extract maximum number of iterations
            self.MAXITERS = yaml_data['MAXITERS']

            #extract marker's size
            self.goal_marker_size = yaml_data['goal_marker_size']

            #extract ambulance's parameters
            self.ambulance_params = yaml_data['ambulance']

    #######################################################
    ##################### House Factoring #################
    #######################################################
    
    def build_house(self):

        """
        Defines the coordinates and openings for each room in the environment and walls in the environment, 
        including outer walls, inner walls, opening midpoints and passages for entrances and exits.
        """

        #intialize the rooms' dictionary
        self.rooms = {}

        #define the house's colours
        self.colors = {
            'Kitchen': 'lightseagreen', 'Room': 'lightpink', 'Office': 'lightcyan',
            'Living Room': 'lightgoldenrodyellow', 'Corridor': 'lightgray',
            'Entrance': 'lightgreen', 'Exit': 'lightcoral'
        }

        #define the house's walls
        self.walls_list = [

            # Outer walls with openings for entrance and exit
            [(0, 0), (0, 4)], [(0, 6), (0, 10)], [(10, 0), (10, 4)], [(10, 6), (10, 10)],
            [(0, 0), (10, 0)], [(0, 10), (10, 10)],
            # Inner walls without openings between rooms
            [(5, 0), (5, 4)], [(5, 6), (5, 10)],
            # Inner horizontal walls with openings for corridor entrances
            [(0, 4), (2, 4)], [(3, 4), (7, 4)], [(8, 4), (10, 4)], [(0, 6), (2, 6)], [(3, 6), (7, 6)], [(8, 6), (10, 6)],
            # Additional tiny walls for the entrance and exit passages
            [(0, 4), (-1.5, 4)], [(0, 6), (-1.5, 6)], [(10, 4), (11.5, 4)], [(10, 6), (11.5, 6)]
        ]

        #define the rooms' coordinates
        self.rooms_dict = {

            'Kitchen': [[0, 0], [5, 0], [5, 4], [0, 4]],
            'Room': [[5, 0], [10, 0], [10, 4], [5, 4]],
            'Office': [[0, 6], [5, 6], [5, 10], [0, 10]],
            'Living Room': [[5, 6], [10, 6], [10, 10], [5, 10]],
            'Corridor': [[0, 4], [10, 4], [10, 6], [0, 6]],
            'Entrance': [[-1.5, 4], [0, 4], [0, 6], [-1.5, 6]],
            'Exit': [[10, 4], [11.5, 4], [11.5, 6], [10, 6]],
        }

        #scale room and wall coordinates
        self.rooms_dict = {name: [(self.house_gain * x, self.house_gain * y) for x, y in points] for name, points in self.rooms_dict.items()}
        self.walls_list = [[(self.house_gain * x0, self.house_gain * y0), (self.house_gain * x1, self.house_gain * y1)] for (x0, y0), (x1, y1) in self.walls_list]

        #define the openings' coordinates
        self.openings_dict = {

            'Kitchen': [[(2.5 - self.opening_width / 2, 4), (2.5 + self.opening_width / 2, 4)]],
            'Room': [[(7.5 - self.opening_width / 2, 4), (7.5 + self.opening_width / 2, 4)]],
            'Office': [[(2.5 - self.opening_width / 2, 6), (2.5 + self.opening_width / 2, 6)]],
            'Living Room': [[(7.5 - self.opening_width / 2, 6), (7.5 + self.opening_width / 2, 6)]],
            'Corridor': [
                [(2.5 - self.opening_width / 2, 4), (2.5 + self.opening_width / 2, 4)],
                [(7.5 - self.opening_width / 2, 4), (7.5 + self.opening_width / 2, 4)],
                [(2.5 - self.opening_width / 2, 6), (2.5 + self.opening_width / 2, 6)],
                [(7.5 - self.opening_width / 2, 6), (7.5 + self.opening_width / 2, 6)]
            ]
        }

        #scale the opening coordinates
        self.openings_dict = {name: [[(self.house_gain * ox0, self.house_gain * oy0), (self.house_gain * ox1, self.house_gain * oy1)] for (ox0, oy0), (ox1, oy1) in openings] for name, openings in self.openings_dict.items()}

        #calculate midpoints for each room
        corridor_y_mid = sum(y for x, y in self.rooms_dict['Corridor']) / len(self.rooms_dict['Corridor'])
        self.midpoints_dict = {
            name: [((self.rooms_dict['Office'][0][0] + self.rooms_dict['Kitchen'][1][0]) / 2, corridor_y_mid)] if name in ['Office', 'Kitchen'] else
                [((self.rooms_dict['Living Room'][0][0] + self.rooms_dict['Room'][1][0]) / 2, corridor_y_mid)] if name in ['Living Room', 'Room'] else []
            for name in self.rooms_dict.keys()
        }
        
        #create rooms' dictionary
        self.rooms = {
            name: {
                'points': points,
                'openings': self.openings_dict.get(name, []),
                'midpoints': self.midpoints_dict.get(name, [])
            } for name, points in self.rooms_dict.items()
        }

    def plot_house(self):

        """
        Plots the environment, including walls, rooms, corridor, and
        entrances/exits. This visualization helps to understand the layout
        and structure of the environment.
        """

        #print midpoints for each room
        print("\n\n---------------------------------------------------------------")
        print("MIDPOINTS")
        for name, midpoints in self.midpoints_dict.items():
            if midpoints:
                for midpoint in midpoints:
                    print(f"Room '{name}' has the following midpoints: {midpoint}")
            else:
                print(f"Room '{name}' has no midpoints.")
        print("---------------------------------------------------------------")

        #create a figure
        fig, ax = plt.subplots(figsize=(12, 12))

        #plot the house's walls
        for (x0, y0), (x1, y1) in self.walls_list:
            ax.plot([x0, x1], [y0, y1], 'k-', linewidth=3)

        #loop over the house's rooms
        for name, room in self.rooms.items():

            #colour the room
            x_points, y_points = zip(*room['points'])
            color = self.colors.get(name, 'white')
            ax.fill(x_points, y_points, color, alpha=0.5, label=name)

            #plot the obstacles
            if self.show_obstacle_points:

                ox, oy = zip(*room['obstacles'])
                ax.scatter(ox, oy, c='black', s=20, zorder=5)

            #plot the midpoints
            for midpoint in room['midpoints']:
                ax.scatter(*midpoint, c='blue', s=100, marker='x')

        #plot the people
        for room, position in self.people_positions: ax.scatter(*position, c='green', s=10, marker='o')

        #plot entrance and exit points without labels
        ax.plot(*self.ENTRANCE, 'go', markersize=self.goal_marker_size)
        ax.plot(*self.EXIT, 'ro', markersize=self.goal_marker_size)

        #set the plot's settings
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(self.ENTRANCE[0] - self.plot_margin, self.EXIT[0] + self.plot_margin)
        ax.set_ylim(-2, 12)
        ax.set_title('Multi-Room House Environment')
        ax.set_xlabel('Width (units)')
        ax.set_ylabel('Height (units)')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        plt.show()

    #######################################################
    ################## Obstacles Generation ###############
    #######################################################
    
    def generate_obstacle_points(self):

        """
        Generates obstacle points for each room based on their wall coordinates
        and openings. Obstacle points are used by agents to avoid collisions.
        """

        def discretize_wall(x0, y0, x1, y1):
            
            """
            Discretizes a wall segment into a series of points based on the given step size.
            """

            #calculate the length of the wall segment, i.e., number of points to generate
            length = np.hypot(x1 - x0, y1 - y0)
            num_points = int(length // self.barrier_step)

            #generate points
            return [(x0 + i * (x1 - x0) / num_points, y0 + i * (y1 - y0) / num_points) for i in range(num_points + 1)]

        def generate_entrance_obstacle_points(entrance_points, max_height=10):
            
            """
            Generate points perpendicular to the entrance to provide obstacle avoidance.
            """

            #extract entrance points
            barrier_step = self.barrier_step
            upper_left, lower_left = entrance_points[0], entrance_points[-1]
            
            #initialize points' list
            points = []

            #calculate the number of points to generate based on barrier_step
            num_points = min(int(max_height // barrier_step), int((max_height - upper_left[1]) // barrier_step))
            
            #generate points above the entrance
            for i in range(1, num_points + 1):

                #extract coordinates
                x_p = upper_left[0]
                y_p = upper_left[1] + i * barrier_step
                
                #ensure points are only above the opening
                if y_p > lower_left[1]:
                    points.append((x_p, y_p))

            #generate points below the entrance
            for i in range(1, num_points + 1):

                #extract coordinates
                x_p = lower_left[0]
                y_p = lower_left[1] - i * barrier_step

                #ensure points are only below the opening
                if y_p < upper_left[1]:
                    points.append((x_p, y_p))

            return points

        #iterate over the house's rooms
        for name, room in self.rooms.items():

            #extract room's points, openings and obstacles
            points, openings, obstacles = room['points'], room['openings'], []
            
            #iterate over corner's coordinates of the room
            for i in range(len(points)):

                #extract current and next point coordinates
                x0, y0 = points[i]
                x1, y1 = points[(i + 1) % len(points)]

                #iterate over the discretized points
                for point in discretize_wall(x0, y0, x1, y1):
                    
                    #check if the point lies within any of the openings
                    is_opening = any(ox0 <= point[0] <= ox1 and oy0 <= point[1] <= oy1 for (ox0, oy0), (ox1, oy1) in openings)
                    
                    #append point if it is not in an opening or in the entrance/exit areas
                    if not is_opening and not (
                        (name == 'Entrance' and (point[0] == -1.5 or point[0] == 0)) or 
                        (name == 'Exit' and (point[0] == 10 or point[0] == 11.5)) or
                        (name == 'Corridor' and (point[0] == 0 or point[0] == 10))
                    ): obstacles.append(point)

            #store the obstacle points in the room's dictionary entry
            self.rooms[name]['obstacles'] = obstacles

        # Generate and store entrance obstacle points
        entrance_obstacles = generate_entrance_obstacle_points(self.rooms['Entrance']['points'])
        self.rooms['Entrance']['obstacles'].extend(entrance_obstacles)

    def get_obstacle_points(self, room_name):

        """
        Retrieves the obstacle points for a specified room.
        """

        return self.rooms[room_name]['obstacles']

    #######################################################
    ################# Heatmap and Survivors ###############
    #######################################################

    def heatmap_generation(self):

        """
        Generate and plot a heatmap showing the likelihood of finding people in each room.
        Uses a weighted probability method to assign probabilities to each room and plots the results.
        """

        rooms = ['Kitchen', 'Room', 'Office', 'Living Room']

        #shuffle the rooms to make the central room random
        random.shuffle(rooms)

        #choose a random room as the "mean" room
        mean_room_index = np.random.choice(len(rooms))
        mean_room = rooms[mean_room_index]

        #assign rooms' weights and normalize them to find probabilities
        weights = np.array([np.exp(-abs(i - mean_room_index) / self.heatmap_factor) for i in range(len(rooms))])
        probabilities = weights / weights.sum()

        #debugging output to check the probabilities
        print(f"Rooms: {rooms}")
        print(f"Mean Room: {mean_room}")
        print(f"Probabilities: {probabilities}")

        #create and store the heatmap
        self.heatmap = dict(zip(rooms, probabilities))

        #create a bar plot for the heatmap 
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(rooms, probabilities, color='hotpink')

        #iterate over each bar in the plot
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        #set title and labels
        ax.set_title('Heatmap of Likelihood of Finding People in Rooms')
        ax.set_xlabel('Rooms')
        ax.set_ylabel('Probability')
        plt.show()

    def survivors_initialization(self):

        '''
        Spawn people in rooms according to the heatmap probabilities and plot the number of people in each room.
        This function ensures that each room receives a number of people proportional to its probability, while also
        guaranteeing that the total number of people matches the expected count (self.n_people). The process involves:
        
        '''

        #extract rooms and normalized probabilities
        rooms = list(self.heatmap.keys())
        probabilities = np.array(list(self.heatmap.values()))
        probabilities /= probabilities.sum()

        #initialize the people_positions list
        people_positions = []

        # Calculate the initial number of people per room
        expected_counts = np.round(probabilities * self.n_people).astype(int)


        #adjust counts to match self.n_people
        while expected_counts.sum() != self.n_people:

            if expected_counts.sum() > self.n_people: expected_counts[np.random.choice(np.where(expected_counts > 1)[0])] -= 1 
            else: expected_counts[np.random.choice(len(rooms))] += 1

            #ensure no room has zero people
        for idx, count in enumerate(expected_counts):

            if count == 0:
                transfer_idx = np.random.choice(np.where(expected_counts > 1)[0])
                expected_counts[transfer_idx] -= 1
                expected_counts[idx] += 1

        #generate people positions radially around the center
        people_positions = []
        for room, count in zip(rooms, expected_counts):

            if count > 0:

                #extract room center's coordinates
                x_coords, y_coords = zip(*self.rooms_dict[room])
                room_center = (np.mean(x_coords), np.mean(y_coords))

                #set people's positions
                for _ in range(count):

                    angle, radius = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, self.people_spread)
                    person_position = (room_center[0] + radius * np.cos(angle), room_center[1] + radius * np.sin(angle))
                    people_positions.append((room, person_position))

        #ensure that the total number of people matches self.n_people
        assert len(people_positions) == self.n_people, f"Total people spawned {len(people_positions)} does not match expected {self.n_people}"

        #initialize survivors' states to Steady
        self.people_trajectories = np.zeros((len(people_positions), self.MAXITERS, 2))
        self.people_status = ['Steady'] * len(people_positions)

        #initialize state counts array
        self.state_counts = np.zeros((self.MAXITERS, 5), dtype=int)

        #initialize survivors' trajectories
        for idx, (_, position) in enumerate(people_positions):
            self.people_trajectories[idx, 0, :] = position

        #count the number of people in each room
        room_people_count = {room: 0 for room in self.rooms_dict.keys()}
        for room, _ in people_positions:
            room_people_count[room] += 1

        #plot number of people per room
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(room_people_count.keys(), room_people_count.values(), color='skyblue')
        for bar in bars:
            ax.annotate(f'{bar.get_height()}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        #set labels and title
        ax.set_title('Number of People in Each Room')
        ax.set_xlabel('Rooms')
        ax.set_ylabel('Number of People')
        plt.show()

        return people_positions

    def survivors_FSM(self, k, barycenter=None):

        """
        Update the positions of people in the trajectory array.
        If barycenter is provided, update positions for escorted people around the barycenter.
        """

        #admissible states
        self.STEADY = "Steady"
        self.ESCORTED = "Escorted"
        self.SAVED = "Saved"
        self.TRANSPORTED = "Transported"
        self.HEALED = "Healed"

        def update_position(idx, k, new_pos):

            """
            Update the trajectory of a person.
            """

            self.people_trajectories[idx, k, :] = new_pos
            return new_pos
        
        def update_steady_person(idx, k, position):

            """
            Update the position of a steady person.
            """

            #leave the person's state unchanged
            new_pos = self.people_trajectories[idx, k - 1, :] if k > 0 else position

            return update_position(idx, k, new_pos)

        def update_escorted_person(idx, k, position, room, barycenter):

            """
            Update the position of an escorted person.
            """

            if barycenter is not None:

                #determine the angle
                angle_step = 2 * np.pi / len([s for s in self.people_status if s == self.ESCORTED])
                angle = idx * angle_step

                #determine the person's radial position
                new_pos = [
                    barycenter[0] + self.escorted_radius * np.cos(angle),
                    barycenter[1] + self.escorted_radius * np.sin(angle)
                ]

                #update the state
                self.people_positions[idx] = (room, new_pos)
                return update_position(idx, k, new_pos)
            
            return position

        def update_saved_person(idx, k, position, room):

            """
            Update the position of a saved person.
            """

            #update the person's state to reach the ambulance
            direction = np.array(self.EXIT) - np.array(position)
            new_pos = np.array(position) + self.person_step_size * direction / np.linalg.norm(direction)
            self.people_positions[idx] = (room, tuple(new_pos))

            #change state to transported if the ambulance's position is reached
            if np.linalg.norm(np.array(new_pos) - np.array(self.EXIT)) < 0.1:
                self.people_status[idx] = self.TRANSPORTED
            
            return update_position(idx, k, new_pos)

        def update_transported_person(idx, k):

            """
            Update the position of a transported person.
            """

            #update the person's state
            new_pos = self.ambulance.position
            
            #change state to healed if the hospital is reached
            if np.linalg.norm(np.array(new_pos) - np.array(self.ambulance.hospital_position)) < 0.1:
                self.people_status[idx] = self.HEALED

            return update_position(idx, k, new_pos)

        #FINITE STATE MACHINE
        for idx, ((room, position), status) in enumerate(zip(self.people_positions, self.people_status)):
            
            #FSM: STEADY state
            if status == self.STEADY: update_steady_person(idx, k, position)

            #FSM: ESCORTED state
            elif status == self.ESCORTED: update_escorted_person(idx, k, position, room, barycenter)
            
            #FSM: SAVED state
            elif status == self.SAVED: update_saved_person(idx, k, position, room)
            
            #FSM: TRANSPORTED state
            elif status == self.TRANSPORTED: update_transported_person(idx, k)

            #FSM: HEALED state
            elif status == self.HEALED: update_position(idx, k, self.ambulance.hospital_position)
        
        #update the state counts for this iteration
        self.state_counts[k] = [sum(1 for s in self.people_status if s == state) for state in [self.STEADY, self.ESCORTED, self.SAVED, self.TRANSPORTED, self.HEALED]]

    def get_number_of_people_in_room(self, room_name):
        
        """
        Returns the number of people inside the specified room.
        
        """
        return sum(1 for person in self.people_positions if person[0] == room_name)
    
    #######################################################
    ##################### Ambulance #######################
    #######################################################
    
    class Ambulance:

        def __init__(self, environment, position, params):

            """
            Initializes the ambulance with environment, position, and parameters.
            Defines FSM states and initial ambulance states.
            """
            
            #define FSM's states
            self.IDLE = 1
            self.DEPARTING = 2
            self.RETURNING = 3

            #store the environment
            self.environment = environment

            #compute max and min values of plots
            self.min_y, self.max_y = self.calculate_min_max_y()

            #initialize the positions
            self.position = position
            self.hospital_position = (self.environment.EXIT[0], self.max_y + 5)
            self.gate_position = (self.environment.EXIT[0], self.min_y - 5)  

            #extract the ambulance's parameters
            self.width = params['width']
            self.height = params['height']
            self.cross_size = params['cross_size']
            self.opening_width = params['opening_width']
            self.opening_height = params['opening_height']
            self.wheel_width = params['wheel_width']
            self.wheel_height = params['wheel_height']
            self.return_countdown = params['return_countdown']
            self.return_iterations = params['return_iterations']
            self.tolerance = params['tolerance']
            self.light_radius = params['light_radius']
            self.mov_step = params['mov_step']

            #ambulance's trajectory
            self.ambulance_trajectory = [] 

            #ambulance's initial states
            self.state = self.IDLE
            self.blinking = False
            self.blink_state = False
        
        def calculate_min_max_y(self):
            
            '''
            Calculate the minimum and maximum y-coordinates in the environment.
            '''

            #aggregate all points from all rooms in the environment
            all_points = [point for room in self.environment.rooms_dict.values() for point in room]
            all_points += [self.environment.ENTRANCE, self.environment.EXIT]

            #determine the extremal coordinates
            min_y = min(point[1] for point in all_points)
            max_y = max(point[1] for point in all_points)

            return min_y, max_y 
    
        def plot_ambulance(self, ax):
            
            """
            Plots the ambulance on the given matplotlib axis.
            """
            
            #xtract the ambulance's position
            ambulance_x, ambulance_y = self.position

            #draw the ambulance body
            ax.add_patch(patches.Rectangle((ambulance_x - self.width / 2, ambulance_y - self.height / 2),
                                        self.width, self.height, linewidth=1.5, edgecolor='black', facecolor='lightgray'))

            #draw the red cross on the ambulance
            ax.add_patch(patches.Rectangle((ambulance_x - self.cross_size / 2, ambulance_y - self.cross_size / 6),
                                        self.cross_size, self.cross_size / 3, linewidth=0, facecolor='red'))
            ax.add_patch(patches.Rectangle((ambulance_x - self.cross_size / 6, ambulance_y - self.cross_size / 2),
                                        self.cross_size / 3, self.cross_size, linewidth=0, facecolor='red'))

            #draw the opening on the left side of the ambulance
            ax.add_patch(patches.Rectangle((ambulance_x - self.width / 2 - self.opening_width / 2, ambulance_y - self.opening_height / 2),
                                        self.opening_width, self.opening_height, linewidth=1, edgecolor='black', facecolor='none'))

            #draw the wheels
            for dx in [-self.width / 2, self.width / 2]:
                for dy in [-self.height / 2, self.height / 2]:
                    ax.add_patch(patches.Rectangle((ambulance_x + dx - self.wheel_height / 2, ambulance_y + dy - self.wheel_width / 2),
                                                self.wheel_height, self.wheel_width, linewidth=1, edgecolor='black', facecolor='black'))

            #optionally draw the blinking lights
            if self.blinking and self.blink_state:
                for dx in [-self.width / 2 + self.light_radius, self.width / 2 - self.light_radius]:
                    ax.add_patch(patches.Circle((ambulance_x + dx, ambulance_y + self.height / 2 - self.light_radius), 
                                                self.light_radius, linewidth=1, edgecolor='red', facecolor='red'))

        def update_blink_state(self):

            """
            Toggles the blinking state for lights' animation.
            """
            
            self.blink_state = not self.blink_state
        
        def is_ambulance_ready_to_depart(self):

            '''Check whether the ambulance is ready to depart, as there are
            no Saved people and no all people have been Healed. '''

            #there is at least a person to be transported
            first_cond = any(status == 'Transported' for status in self.environment.people_status)

            #there are no people in the 'Saved' status
            second_cond = not any(status == 'Saved' for status in self.environment.people_status)

            return first_cond and second_cond

        def is_new_ambulance_needed(self):

            '''Check if there are any people in 'Steady' or 'Escorted' status,
            return True if a new ambulance is needed, False elsewhere.'''

            return not all(status in ['Transported', 'Healed'] for status in self.environment.people_status)
    
        def ambulance_FSM(self):
            
            """
            Finite State Machine (FSM) for managing ambulance states.
            """
            
            def move_up():

                """Moves the ambulance vertically upwards."""

                self.position = (self.position[0], self.position[1] + self.mov_step)

            def reached_end_of_screen():

                """Checks if the ambulance has moved off the top of the screen."""

                return self.position[1] > self.hospital_position[1]

            #FSM: IDLE state
            if self.state == self.IDLE:
                
                #check whether the ambulance can depart
                if self.is_ambulance_ready_to_depart():

                    self.state = self.DEPARTING
                    self.blinking = True

            #FSM: DEPARTING state
            elif self.state == self.DEPARTING:
                
                #move the ambulance
                move_up()

                #check whether the ambulance is out of sight
                if reached_end_of_screen():
                    
                    #spawn a new ambulance
                    self.position = self.gate_position    
                    self.blinking = True
                    self.return_countdown = self.return_iterations

                    #check whether a new ambulance is needed
                    self.state = self.RETURNING if self.is_new_ambulance_needed() else self.IDLE
            
            #FSM: RETURNING state
            elif self.state == self.RETURNING:
                
                #wait for the ambulance
                if self.return_countdown > 0: self.return_countdown -= 1

                else: #move to exit
                    
                    #move the ambulance
                    move_up()

                    #compute the distance to exit
                    distance = np.linalg.norm(np.array(self.position) - np.array(self.environment.EXIT))
                    
                    #check for arrival
                    if distance <= self.tolerance:

                        self.state = self.IDLE
                        self.blinking = False

            #update the ambulance's trajectory
            self.ambulance_trajectory.append((self.position, self.blinking, self.state == self.IDLE))

        #######################################################
        #################### Visualization ####################
        #######################################################

        def plot_house(self):

            """
            Plots the environment, including walls, rooms, corridor, and
            entrances/exits. This visualization helps to understand the layout
            and structure of the environment.
            """

            #print midpoints for each room
            print("\n\n---------------------------------------------------------------")
            print(f"MIDPOINTS")
            for name, midpoints in self.midpoints_dict.items():

                if midpoints:
                    for midpoint in midpoints: print(f"Room '{name}' has the following midpoints: {midpoint}")

                else:
                    print(f"Room '{name}' has no midpoints.")
            print("---------------------------------------------------------------")

            #create a figure
            fig, ax = plt.subplots(figsize=(12, 12))

            #plot the house's walls
            for (x0, y0), (x1, y1) in self.walls_list:
                ax.plot([x0, x1], [y0, y1], 'k-', linewidth=3)

            #loop over the house's rooms
            for name, room in self.rooms.items():

                #extract the room's points
                x_points, y_points = zip(*room['points'])
                
                #set the room's colour
                color=self.colors.get(name, 'white')

                #add room name in the center of the room and fill the room area with desired colours
                ax.fill(x_points, y_points, color, alpha=0.5, label=name)
                #ax.text(sum(x_points) / len(x_points), sum(y_points) / len(y_points), name, ha='center', va='center')

                #scatter plot for obstacle points
                if self.show_obstacle_points:

                    ox, oy = zip(*room['obstacles'])
                    ax.scatter(ox, oy, c='black', s=20, zorder=5)
                
                #plot the midpoints
                for midpoint in room['midpoints']:
                    ax.scatter(*midpoint, c='blue', s=100, marker='x')

            #plot the people
            for room, position in self.people_positions: ax.scatter(*position, c='red', s=10, marker='o')

            #plot entrance and exit points without labels
            ax.plot(*self.ENTRANCE, 'go', markersize=self.goal_marker_size)
            ax.plot(*self.EXIT, 'ro', markersize=self.goal_marker_size)

            #set the plot's settings
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(self.ENTRANCE[0] - self.plot_margin, self.EXIT[0] + self.plot_margin)
            ax.set_ylim(-2, 12)
            ax.set_title('Multi-Room House Environment')
            ax.set_xlabel('Width (units)')
            ax.set_ylabel('Height (units)')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True)
            plt.show()

#############################################################################
############################### Agent Class #################################
#############################################################################

class Agent:

    #######################################################
    ##################### Initialization ##################
    #######################################################

    def __init__(self, environment):

        '''
        Initializes the agent with the given environment. This includes loading parameters,
        generating the connected graph, initializing trajectories, and starting the BDI control loop.
        '''

        #save the environment's class
        self.environment = environment

        #extract parameters from yaml file
        self.load_yaml_parameters()
        
        #create the network graph for robots
        self.W, self.Adj = self.network_initialization()

        #initialize trajectories for simulation
        self.trajectories_initialization()

        #start the SAR's simulation
        self.BDI_control_loop()  

    #######################################################
    ################ Parameters Extraction ################
    #######################################################

    def load_yaml_parameters(self):

        """
        Loads environment parameters from a YAML file.
        """

        #open the file to extract parameters
        if os.path.exists(yaml_file_path):

            #load yaml file
            with open(yaml_file_path, 'r') as file: yaml_data = yaml.safe_load(file)
                
            #extract agents' parameters
            self.N = yaml_data['NN']
            self.n_z = yaml_data['n_z']
            self.p_ER = yaml_data['p_ER']

            #extract local targets' parameters
            self.radius = yaml_data['radius']
            self.gain_mid = yaml_data['gain_mid']
            self.gain_room = yaml_data['gain_room']
            self.gain_out = yaml_data['gain_out']

            #extract simulation's parameters
            self.MAXITERS = yaml_data['MAXITERS']
            self.step_size = yaml_data['step_size']
            self.dt = yaml_data['dt']
            self.global_goal_threshold = yaml_data['global_goal_threshold']
            self.skip_iter_factor = yaml_data['skip_iter_factor']
            self.rooms_order = yaml_data['rooms_order']
            self.reasoning_method = yaml_data['reasoning_method']

            #extract optimization's parameters
            self.gamma_r_lt = yaml_data['gamma_r_lt']
            self.gamma_agg = yaml_data['gamma_agg']
            self.gamma_bar = yaml_data['gamma_bar']

            #extract potential function's parameters
            self.K = yaml_data['K']
            self.q_star = yaml_data['q_star']

            #extract graphics' parameters
            self.goal_marker_size = yaml_data['goal_marker_size']
            self.robot_size = yaml_data['robot_size']
            self.robot_colour = yaml_data['robot_colour']
            self.barycenter_colour = yaml_data['barycenter_colour']
            self.trajectory_colour = yaml_data['trajectory_colour']
            self.view_type = yaml_data['view_type']
            self.show_local_targets= yaml_data['show_local_targets']
            self.show_global_targets = yaml_data['show_global_targets']
            self.logs_enabled = yaml_data['logs_enabled']

            #extract the survivors' parameters
            self.max_survivors_escort = yaml_data['max_survivors_escort']

    def load_rooms_to_visit(self):

        """
        Load rooms to visit based on the room_selection method.
        """
        
        #get rooms to visit from txt file
        if self.rooms_order == "txt":

            if os.path.exists(rooms_path_txt):
                
                #save the rooms to visit from txt file
                rooms_file = open(rooms_path_txt, 'r')
                self.rooms_to_visit = [self.environment.rooms_dict[line.strip('\n')] for line in rooms_file]
                rooms_file.close()
                #print("Rooms to visit based on txt file: ", self.rooms_to_visit)
            
            else: print(f"[MAP_HANDLER]: txt file '{rooms_name_txt}' not found in folder '{rooms_path_txt}'")

        #get rooms to visit based on heatmap probabilities
        elif self.rooms_order == "heat":

            sorted_rooms = sorted(self.environment.heatmap.items(), key=lambda item: item[1], reverse=True)
            self.rooms_to_visit = [self.environment.rooms_dict['Entrance']] + [self.environment.rooms_dict[room] for room, _ in sorted_rooms if room not in ['Entrance', 'Exit']] + [self.environment.rooms_dict['Exit']]
            #print("Rooms to visit based on heatmap probabilities: ", self.rooms_to_visit)

        else:
            print("[MAP_HANDLER]: Invalid room_selection method. Use 'txt' or 'heat'.")
        
        #extract the list of rooms' names
        rooms_to_visit = ['Entrance'] + [room for room, _ in sorted_rooms if room not in ['Entrance', 'Exit']] + ['Exit']
        
        return rooms_to_visit
            
    #######################################################
    ##################### Network #########################
    #######################################################

    def network_initialization(self):

        '''Generates a connected graph with a given probability parameter p_ER and constructs 
        an adjacency matrix with Metropolis-Hastings weights.'''

        def is_connected(adj_matrix):

            '''Check if the graph is connected using matrix power.'''
            
            return np.all(np.linalg.matrix_power(np.identity(self.N, dtype=int) + adj_matrix, self.N) > 0)

        while True:

            # Generate symmetric adjacency matrix with zero diagonal
            Adj = np.random.binomial(1, self.p_ER, (self.N, self.N))

            #set diagonal to zero and ensure matrix's symmetry
            np.fill_diagonal(Adj, 0)  
            Adj = np.maximum(Adj, Adj.T)

            if is_connected(Adj):
                break

            #print("\nThe graph is NOT connected\n")
            #quit()

        #compute Metropolis-Hastings weights
        degrees = Adj.sum(axis=1)
        W = np.where(Adj == 1, 1 / (1 + np.maximum(degrees[:, None], degrees[None, :])), 0)
        np.fill_diagonal(W, 1 - W.sum(axis=0))

        return W, Adj

    #######################################################
    #################### Trajectories #####################
    #######################################################

    def generate_circular_positions(self, N, n_z, central_target_pos, radius, gain):

        """
        Create target positions for leaders around a central point.
        
        Parameters:
            N (int): Number of agents
            central_position (tuple): Central position around which target positions are created (x, y)
            radius (float): Radius of the circle around the central position
            gain (float): Gain controlling the degree of closeness of the leaders to the central point
            
        Returns:
            target_positions (numpy.ndarray): Array of target positions for agents
        """
        
        target_positions = np.zeros((N, n_z))

        #generate the target positions around the central point
        for i in range(N):

            #define a target angle for the i-th robot
            angle = (2 * np.pi * i) / N
            
            #find the offset coordinates via trigonometry
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)

            #store the location of the i-th robot
            if n_z == 2: target_positions[i] = (central_target_pos[0] + gain * x_offset, central_target_pos[1] + gain * y_offset)
            else: target_positions[i, :2] = (central_target_pos[0] + gain * x_offset, central_target_pos[1] + gain * y_offset)

        return target_positions
    
    def trajectories_initialization(self):

        #initialize iterations
        self.k = 0

        #initialize global variables
        self.next_room_name = None
        self.room_reached = False
        self.people_escorted = False

        #initialize people to escort
        self.people_to_escort = []

        #define initial and final targets
        self.initial_point = self.environment.ENTRANCE
        self.final_point = self.environment.EXIT
        
        #initialize local targets of the robots
        self.R = np.zeros((self.N, self.n_z))
        self.local_targets = []

        #initialize global target of the robot
        self.b = np.zeros((self.n_z))
        self.global_targets = []

        #initialize robot and trackers' trajectories
        self.Z = np.zeros((self.N, self.MAXITERS, self.n_z))
        self.S = np.zeros((self.N, self.MAXITERS, self.n_z))
        self.V = np.zeros((self.N, self.MAXITERS, self.n_z))

        #initialize cost and gradients
        self.F = np.zeros((self.MAXITERS))
        self.grad_z = np.zeros((self.MAXITERS, self.n_z))
        self.grad_s = np.zeros((self.MAXITERS, self.n_z))
        self.grad_norm_z = np.zeros((self.MAXITERS))
        self.grad_norm_s = np.zeros((self.MAXITERS))
        self.true_barycenter = np.zeros((self.MAXITERS, self.n_z))

        #define initial radial positions of robots
        Z_init = self.generate_circular_positions(self.N, self.n_z, self.initial_point, self.radius, self.gain_out)
        self.Z[:, 0, :] = Z_init

    #######################################################
    ################ Optimization Problem #################
    #######################################################

    def cost_function(self, Z, S, R, b, gamma_r_lt, gamma_bar, gamma_agg, obstacles):
    
        """
        Defines the cost function that needs to be minimized from the distributed aggregative
        optimization algorithm. Based on the boolean input values, it can keeps into account
        the potential function for each point of corr_points, by means of the distances,
        it allows robots to stay close to their local targets while mantaining formation.
        """

        if obstacles is not None:

            #convert obstacles list into numpy array
            obstacles = np.array(obstacles)

            #get the number of obstacles
            n_obst = obstacles.shape[0]

        else: n_obst = 0

        #intialize repulsive cost and gradient
        repulsive_potential = 0
        repulsive_gradient = np.zeros(self.n_z)

        #compute repulsion cost and gradient
        for i in range(n_obst):

            #compute norm of the distance and gradient
            dist = 0.5* (Z - obstacles[i]) @ (Z - obstacles[i])
            grad_dist = (Z - obstacles[i])

            #apply potential function
            if dist <= self.q_star:
                repulsive_potential += 0.5 * self.K * ((1 / dist) - (1 / self.q_star))**2
                repulsive_gradient += self.K * (((1 / self.q_star) - (1 / dist)) * grad_dist / dist**2)

        #define cost sub-functions
        d_robot_target = (Z - R) @ (Z - R)  #distance between robot (x_i) and its local target (r_i)
        d_barycenter_target = (S - b) @ (S - b)  #distance between estimated barycenter (sigma) and its target (b)
        d_robot_barycenter = (S - Z) @ (S - Z)  #distance between robot (x_i) and team's estimated barycenter (sigma)

        #define the cost function for agent i to be minimized
        f_z = gamma_r_lt * d_robot_target + gamma_bar * d_barycenter_target + gamma_agg * d_robot_barycenter + repulsive_potential

        #define the gradients of the cost with respect to both the decision variables
        df_z = 2 * gamma_r_lt * (Z - R) - 2 * gamma_agg * (S - Z) + repulsive_gradient
        df_s = 2 * gamma_bar * (S - b) + 2 * gamma_agg * (S - Z)

        return f_z, df_z, df_s

    def phi_agent(self, z_i):

        #return agent's position
        return z_i, np.eye(z_i.shape[0])

    def distributed_aggregative_optimization(self, k, Z, S, V, F, R, grad_z, grad_s, b, obstacles, beliefs, desires, intentions):

        '''This function implements the aggregative optimization algorithm for a multi-robot system.
        It initializes decision variables, estimates of the aggregative variable, and cumulative
        gradients for each agent. Then, it iterates through optimization steps for a specified number
        of iterations. Within each iteration, it updates decision variables based on gradients and
        neighboring agents' information, calculates the estimate of the aggregative variable, and
        accumulates the cost function values. This process guides the collective movement of robots
        towards desired goals, such as the barycenter, while optimizing a defined cost function.'''

        #extract the gains
        gamma_r_lt = intentions['targets_attraction']
        gamma_agg = intentions['barycenter_attraction']
        gamma_bar = intentions['barycenter_repulsion']
        
        #compute initial values
        if k == 0:

            #initialize the estimate of the aggregative variable for each agent
            S[:,0], _ = self.phi_agent(Z[:,0,:])

            #initialize the cost and the cumulative gradient for each agent
            for i in range(self.N):
                
                #initialize the cost function
                f_i, df_z, df_s = self.cost_function(Z[i,0], S[i,0], R[i], b, gamma_r_lt, gamma_bar, gamma_agg, obstacles)
                F[0] += f_i
                
                #initialize the cumulative gradient for each agent
                V[i,0] = df_s
                grad_z[0] += df_z
                grad_s[0] += df_s

                #initialize true (centralized) quantities
                self.true_barycenter[0] += self.phi_agent(self.Z[i, 0])[0] / self.N

        #loop over each agent
        for i in range (self.N):

            #obtain the in-neighbors of the current agent based on adjacency
            Ni = np.nonzero(self.Adj[i])[0]

            #compute the cost function and its gradient at the current decision variable
            _, df_z, df_s = self.cost_function(Z[i,k], S[i,k], R[i], b, gamma_r_lt, gamma_bar, gamma_agg, obstacles) 
            phi_k, dphi_k = self.phi_agent(Z[i,k])

            #update of the agent's decision variable using a steepest descent step
            descent = df_z + dphi_k @ V[i,k]

            #update of the agent's decision variable using a steepest descent step 
            Z[i,k+1] = Z[i,k] - self.step_size * descent
            phi_kp, _ = self.phi_agent(Z[i,k+1])

            #update of the estimate (tracker) of the aggregative variable sigma
            S[i,k+1] = self.W[i,i] * S[i,k] + phi_kp - phi_k

            #compute cost and gradients for update
            for j in Ni: S[i, k+1] += self.W[i,j] * S[j,k]
            
            f_i, df_z_plus, df_s_plus = self.cost_function(Z[i,k+1], S[i,k+1], R[i], b, gamma_r_lt, gamma_bar, gamma_agg, obstacles) 

            #update of the tracker of the gradient sum
            V[i, k+1] = self.W[i,i] * V[i,k] + df_s_plus - df_s

            #sum the gradient's neighbors contribution
            for j in Ni: V[i,k+1] += self.W[i,j] * V[j,k]

            #store the cost function's value
            F[k+1] += f_i

            #store the gradients
            grad_z[k+1] += df_z_plus
            grad_s[k+1] += df_s_plus

            #store the true barycenter
            self.true_barycenter[k+1] += self.phi_agent(self.Z[i, k+1])[0] / self.N

            #logs
            if k % self.skip_iter_factor == 0:

                #create a summary of people statuses
                status_summary = {}
                for idx, status in enumerate(self.environment.people_status):
                    if status not in status_summary:
                        status_summary[status] = []
                    status_summary[status].append(idx)

                #convert the summary to a readable string
                status_log = ', '.join(f"{status}: {indices}" for status, indices in status_summary.items())

                print("\n\n---------------------------------------------------------------")
                print("DISTRIBUTED AGGREGATIVE ALGORITHM")
                print("Global Goal: ", self.b)
                print(f"Formation navigating to {self.intentions['target_room']}")
                print(f"Intention: {self.get_true_intention(intentions)}")
                print(f"Completion: {((k/self.MAXITERS)*100):.2f}%")
                print(f"Total Cost: {F[k+1]:.4f}")
                print(f"People status: {status_log}")
                '''
                for idx, ((room, position), status) in enumerate(zip(self.environment.people_positions, self.environment.people_status)):
                    print(f"Person {idx} status: {status}")
                '''
                print(f"Ambulance state: position={self.environment.ambulance.position}, blinking={self.environment.ambulance.blinking}")
                print("---------------------------------------------------------------")

        return Z, S, V, F, grad_s, grad_z

    #######################################################
    ################### BDI Functions #####################
    #######################################################

    def initialize_beliefs(self):

        '''
        Initialize the agent's beliefs about the environment, such as rooms to visit,
        current room, people status, obstacles, and ambulance state.
        '''
        
        beliefs = {

            'rooms_to_visit': self.load_rooms_to_visit(), #load the list of rooms to visit
            'current_room': 'Outside', #start outside the house
            'people_status': self.environment.people_status, #initial status of people
            'people_to_escort': [],  # people to escort
            'people_left_in_room': {room: 0 for room in self.environment.rooms_dict.keys()},
            'obstacles': {name: self.environment.get_obstacle_points(name) for name in self.environment.rooms_dict.keys()}, #obstacles points in each room
            'midpoints': self.environment.midpoints_dict,  # midpoints for each room
            'ambulance': self.environment.ambulance.state, #state of the ambulance
            'gains': {'gain_mid': self.gain_mid, 'gain_room': self.gain_room, 'gain_out': self.gain_out}, #possible targets' gains of agents
            'radius': self.radius,
            'MAXITERS': self.MAXITERS, #maximum number of iterations
            'visited_rooms': [] # initialize visited rooms
        }

        # Count initial people in rooms
        for room in beliefs['people_left_in_room']:
            beliefs['people_left_in_room'][room] = sum(1 for status, (r, _) in zip(beliefs['people_status'], self.environment.people_positions) if r == room and status == 'Steady')

        #log
        if self.logs_enabled:
            print("\n\n---->") 
            print(f"INITIAL BELIEFS:\nRooms to Visit: {beliefs['rooms_to_visit']}\nCurrent Room: {beliefs['current_room']}\nPeople Status:{beliefs['people_status']}\nPeople To Escort: {beliefs['people_to_escort']}\nMidpoints: {beliefs['midpoints']}\nAmbulance State: {beliefs['ambulance']}")
            print("---->\n\n")

        return beliefs
    
    def initialize_intentions(self, beliefs):

        '''
        Initialize the agent's intentions, including the target room, target position,
        whether to escort people, and whether to exit the house.
        '''

        intentions = {
            'target_room': 'Entrance', #initially target entrance name
            'target_position': self.environment.ENTRANCE, #initially target entrance's center
            'escort_people': False, #not escorting people initially
            'standby_mode': False, #not in standby mode
            'gain': beliefs['gains']['gain_out'],
            'radius': beliefs['radius'],
            'exit': False,  #not exiting initially
        }
        
        #log
        if self.logs_enabled:
            print("\n\n---->") 
            print(f"INITIAL INTENTIONS:\nTarget Room: {intentions['target_room']}\nTarget Position: {intentions['target_position']}\nEscort People: {intentions['escort_people']}\nExit: {intentions['exit']}")
            print("---->\n\n")

        return intentions
    
    def perceive_environment(self):

        '''
        Gather new perceptions from the environment, including the positions of people
        and the state of the ambulance.
        '''

        perceptions = {
            'people_positions': self.environment.people_positions, #current positions of people
            'people_status': self.environment.people_status, #current status of people
            'ambulance_position': self.environment.ambulance.position, #current position of the ambulance
            'ambulance_state': self.environment.ambulance.state, #current state of the ambulance
        }

        #log
        if self.logs_enabled:
            print("\n\n---->")  
            print(f"PERCEPTION:\nPeople Positions: {perceptions['people_positions']}\nPeople Status: {perceptions['people_status']}\nAmbulance in state {perceptions['ambulance_state']} placed in position {perceptions['ambulance_position']}")
            print("---->\n\n")

        return perceptions
    
    def belief_revision_function(self, perceptions, beliefs):

        '''
        Update beliefs based on new perceptions, such as people positions and ambulance state.
        '''

        beliefs['people_positions'] = perceptions['people_positions'] #update people positions
        beliefs['people_status'] = perceptions['people_status'] #update people status
        beliefs['ambulance_position'] = perceptions['ambulance_position'] #update ambulance position
        beliefs['ambulance_state'] = perceptions['ambulance_state'] #update ambulance state

        #update people_left_in_room belief
        for room in beliefs['people_left_in_room']:
            beliefs['people_left_in_room'][room] = sum(1 for status, (r, _) in zip(perceptions['people_status'], perceptions['people_positions']) if r == room and status == 'Steady')
        
        #add the current room to visited rooms if not already present
        if beliefs['current_room'] not in beliefs['visited_rooms'] and beliefs['current_room'] != 'Outside':
            beliefs['visited_rooms'].append(beliefs['current_room'])

        #log
        if self.logs_enabled:
            print("\n\n---->")  
            print(f"BELIEF REVISION FUNCTION:\nPeople Positions: {beliefs['people_positions']}\nPeople Status: {beliefs['people_status']}\nAmbulance in state {beliefs['ambulance_state']} placed in position {beliefs['ambulance_position']}")
            print("---->\n\n")

        return beliefs
    
    def generate_options(self, beliefs):

        '''
        Generate possible options (desires) based on beliefs, such as visiting rooms,
        escorting people, or exiting the house.
        '''

        def check_all_status(statuses, target_statuses):

            '''Helper function to check if all statuses are within target statuses'''

            return all(status in target_statuses for status in statuses)

        def check_any_status(statuses, target_status, current_room):

            '''Helper function to check if any status matches the target in the current room'''
            
            return any(status == target_status and room == current_room for status, (room, _) in zip(statuses, self.environment.people_positions))
    
        #extract beliefs
        people_status = beliefs['people_status']
        current_room = beliefs['current_room']
        rooms_to_visit = beliefs['rooms_to_visit']

        #determine desires
        desires = {
            'visit_rooms': rooms_to_visit,  # rooms left to visit
            'escort_people': check_any_status(people_status, 'Steady', current_room),  # people to be escorted in current room
            'standby_mode': check_all_status(people_status, ['Saved', 'Transported', 'Healed']) and not check_all_status(people_status, ['Healed']), #standby mode in exit area
            'exit_house': check_all_status(people_status, ['Transported', 'Healed']) or not rooms_to_visit, #exit from the house
            'should_return': beliefs['people_left_in_room'][rooms_to_visit[0]]>0  #people left in previous room
        }

        #log (to put in the algorithm)
        if self.logs_enabled:
            print("\n\n---->") 
            print(f"Standby Mode: {desires['standby_mode']}\n")
            print(f"GENERATE OPTIONS:\nVisit Rooms: {desires['visit_rooms']}\nEscort People: {desires['escort_people']}\nExit House: {desires['exit_house']}")
            print("---->\n\n")

        return desires

    def filter_options(self, beliefs, desires):
        
        '''
        Filter and prioritize new intentions from the generated options (desires) to determine the next
        actions for the agents. The function ensures that higher priority desires are addressed first,
        such as escorting people, entering standby mode, exiting the house, or continuing patrol.
        '''

        def generate_intention(exploring=False, escort_people=False, target_room=None, target_position=None, standby_mode=False, exit=False, gain=None, radius=None, gamma_r_lt=self.gamma_r_lt, gamma_agg=self.gamma_agg, gamma_bar=self.gamma_bar):
            
            '''
            Generate and set the intentions for the agents based on specified parameters.
            '''
                     
            return {
                'exit': exit,
                'escort_people': escort_people,
                'explore_house': exploring,
                'target_room': target_room,
                'target_position': target_position,
                'standby_mode': standby_mode,
                'gain': gain,
                'radius': radius,
                'targets_attraction': gamma_r_lt,
                'barycenter_attraction': gamma_agg,
                'barycenter_repulsion': gamma_bar
            }

       #define priorities and corresponding functions for desires
        priorities = {
            'escort_people': (1, lambda: generate_intention(False, True, 'Exit', self.get_room_center('Exit'), False, False, beliefs['gains']['gain_mid'], beliefs['radius'])),
            'standby_mode': (2, lambda: generate_intention(False, False, 'Exit', self.get_room_center('Exit'), True, False, beliefs['gains']['gain_mid'], beliefs['radius'])),
            'exit_house': (3, lambda: generate_intention(False, False, 'Exit', self.environment.EXIT, False, True, beliefs['gains']['gain_out'], beliefs['radius'], gamma_agg=0)),
            'visit_rooms': (4, lambda: generate_intention(True, False, next_room := desires['visit_rooms'][0], self.get_room_center(next_room), False, False, beliefs['gains']['gain_mid'], beliefs['radius']) if desires['visit_rooms'] else None)
        }

        #check whether there are no people left in the previous room
        if beliefs['current_room'] != 'Outside' and not desires['should_return']: desires['visit_rooms'].pop(0)

        #generate intentions based on the highest priority desire
        for key, (_, func) in sorted(priorities.items(), key=lambda item: item[1][0]):
            if desires.get(key):
                intention = func()
                if intention:
                    return intention

        #return default intentions if no valid desires found
        return generate_intention()

    def create_plan(self, beliefs, intentions):

        '''
        Develop a plan to achieve the current intentions by constructing a list of waypoints
        from the current room to the target room, i.e., to setup the travel for robot formation.
        '''

        plan = []

        #extract strings relayed to current and target rooms
        current_room = beliefs['current_room']
        target_room = intentions['target_room']

        #validate target_room
        if target_room not in self.environment.rooms_dict: raise ValueError(f"Invalid target room: {target_room}")

        #create escort group
        if intentions['escort_people']: beliefs['people_to_escort'] = self.create_escort_group(current_room) 

        #generate waypoints and add the final target position
        plan = self.get_waypoints(current_room, target_room, beliefs['midpoints']) + [intentions['target_position']]
        
        #fetch obstacles for the current and target rooms
        current_obstacles = self.environment.get_obstacle_points(current_room) if current_room != "Outside" else []
        target_obstacles = [] if target_room == 'Exit' else self.environment.get_obstacle_points(target_room)
        total_obstacles = current_obstacles + target_obstacles

        #log
        if self.logs_enabled:
            print("\n\n---->") 
            print(f"CREATE PLAN:\nCreating plan from {current_room} to {target_room}\nWaypoints: {plan}\nNumber of Obstacles: {len(total_obstacles)}")
            print("---->\n\n")

        return plan, total_obstacles

    def execute_plan(self, plan, obstacles, beliefs, desires, intentions):

        '''
        Execute the planned actions by setting the next targets and solving the optimization problem.
        '''
        
        while plan:
            
            #set the next target for the agents
            target = plan.pop(0)            
            self.set_next_targets(target, beliefs, desires, intentions)

            #continue optimizing until the goal is reached or the maximum iterations are reached
            while self.should_optimize(beliefs, desires, intentions):
                        
                #distributed aggregative optimization algorithm
                self.Z, self.S, self.V, self.F, self.grad_s, self.grad_z = self.distributed_aggregative_optimization(
                    self.k, self.Z, self.S, self.V, self.F, self.R, self.grad_z, self.grad_s, self.b, obstacles, beliefs, desires, intentions
                )

                #compute the average barycenter of the formation
                barycenter = self.S[:, self.k - 1, :].mean(axis=0)

                #update the state of survivors and the ambulance
                self.environment.survivors_FSM(self.k, barycenter)
                self.environment.ambulance.ambulance_FSM()

                #increment the iteration counter
                self.k += 1
            
            #if the target is the Exit and the intention is to escort people, mark them as "Saved"
            if target == self.get_room_center('Exit') and intentions['escort_people']:
                
                self.environment.people_status = ['Saved' if status == 'Escorted' else status for status in self.environment.people_status]
                self.beliefs['people_to_escort'] = []

        #update the current room after each iteration
        self.beliefs['current_room'] = self.intentions['target_room']

        print("\n---------------------------------------------------------------")
        print(f"Formation has been successfully arrived to {intentions['target_room']} in {intentions['target_position']}.")
        print("---------------------------------------------------------------\n")

    def BDI_control_loop(self):

        '''
        Main control loop of the agent using the BDI (Belief-Desire-Intention) structure.
        This loop continuously updates beliefs, generates desires, filters them to form
        intentions, and executes plans to achieve those intentions.
        '''

        #initialize beliefs and intentions
        self.beliefs = self.initialize_beliefs()
        self.intentions = self.initialize_intentions(self.beliefs)

        while True: #BDI control loop
            
            #perception
            perceptions = self.perceive_environment()
            
            #beliefs
            self.beliefs = self.belief_revision_function(perceptions, self.beliefs)
            
            #desires
            self.desires = self.generate_options(self.beliefs)
            
            #intentions
            self.intentions = self.filter_options(self.beliefs, self.desires)
            
            #plan
            plan, obstacles = self.create_plan(self.beliefs, self.intentions)
            self.execute_plan(plan, obstacles, self.beliefs, self.desires, self.intentions)

            #termination condition
            if self.check_termination(): break

        #visualization
        self.SAR_plots()
        self.SAR_animation()

    #######################################################
    ################### Supporting Functions ##############
    #######################################################

    def get_true_intention(self, intentions):

        """
        Helper function to get the key of the current intention that is True.
        """

        for key, value in intentions.items():
            if value:
                return key
        return None
    
    def should_optimize(self, beliefs, desires, intentions):

        '''
        The optimization should continue under the following conditions:
        1. The goal has not been reached and the maximum iterations have not been exceeded.
        2. The robot is in standby mode and not all people are healed.
        '''

        #compute logical terms of the expression
        goal_not_reached = not self.is_goal_reached(intentions['target_room'])  # Check if the current target has not been reached.
        time_not_passed = self.k < beliefs['MAXITERS'] - 1  # Check if the maximum number of iterations has not been exceeded.
        standby_mode = intentions['standby_mode']  # Check if the robot is in standby mode.
        not_all_healed = not all(status == 'Healed' for status in beliefs['people_status'])  # Check if not all people are healed.

        return (goal_not_reached and time_not_passed) or (standby_mode and not_all_healed)

    def check_termination(self):

        '''
        Check whether the simulation should terminate based on current intentions and status.
        '''

        #terminate if the intention is to exit and the target is reached
        if self.intentions.get('exit', False) and self.is_goal_reached('Exit', type="room"):
            print("Exiting the house. Simulation completed.")
            return True
        
        #check if the maximum number of iterations has been reached
        if self.k >= self.MAXITERS-1:
            print("Maximum iterations reached. Simulation terminated.")
            return True
        
        return False

    def set_next_targets(self, target, beliefs, desires, intentions):

        """Set the agents' next local and global target points."""

        '''gain = self.gain_out if type == "out" else self.gain_room if (target == room_b and room_name not in ["Entrance", "Exit"]) else self.gain_mid
        #get rooms' radius and gain
        if (target == room_b) and not(room_name in ["Entrance", "Exit"]): gain = self.gain_room  #entrance and exit
        else: gain = self.gain_mid #midpoints
        if type == "out": gain = self.gain_out #in and out'''

        #set the agents' local goals (append them to show trajectory)
        self.R = self.generate_circular_positions(self.N, self.n_z, target, beliefs['radius'], intentions['gain'])
        self.local_targets.append(self.R)

        #set the global formation goal
        self.b = np.asarray(target)
        self.global_targets.append(self.b)

        return self.R, self.b

    def is_goal_reached(self, room_name, type=None):

        #distributed check: each agent checks its distance to the goal
        self.distances = np.linalg.norm(self.S[:, self.k] - self.b, axis=1)

        #check whether global goal has been reached (message passing in ROS2)
        if np.all(self.distances < self.global_goal_threshold):
            return True
        
        return False
    
    def get_room_center(self, room_name):

        '''
        Compute the center of a given room.
        '''

        x_coords, y_coords = zip(*self.environment.rooms_dict[room_name])
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def get_waypoints(self, current_room, target_room, midpoints):

        '''
        Generate a list of waypoints from the current room to the target room.
        '''

        waypoints = []

        #get the midpoint for the current room, if any
        current_midpoint = midpoints.get(current_room, [])
        if current_midpoint: waypoints.append(current_midpoint[0])

        #get the midpoint for the target room, if any
        target_midpoint = midpoints.get(target_room, [])
        if target_midpoint: waypoints.append(target_midpoint[0])

        return waypoints
    
    def create_escort_group(self, next_room_name):

        """
        Add people in the current room to the robots' group for escorting.
        The returning object escort_group contains a list of tuples. Each
        tuple represents a person that was found in the specified room
        (next_room_name) and consists of:
        -> The name of the room from which the person was taken.
        -> The position of the person within that room.
        """

        def should_recruit_person():

            '''
            A person in the given room should be recruited into the escort group, if
            it needs to be escorted and the group has not exceeded yet the maximum size.
            '''

            return (room == next_room_name) and (len(escort_group) < self.max_survivors_escort) \
                and (self.environment.people_status[idx] == 'Steady')

        #initialize the escort group
        escort_group = []
        
        print("\n---------------------------------------------------------------")
        print("SAR GROUP FORMATION")
        print(f"{next_room_name} has {self.environment.get_number_of_people_in_room(next_room_name)} people.")

        #iterate over people in the environment and add them to the escort group
        for idx, (room, position) in enumerate(self.environment.people_positions):

            if should_recruit_person(): #add person to the escort group
                
                escort_group.append((room, position))
                self.environment.people_status[idx] = 'Escorted'
                print(f"Added person at {position} from {room} to escort group")
    
        #log the escort group
        print(f"Escort group created for room {next_room_name} composed of {len(escort_group)} people")
        print("---------------------------------------------------------------\n")

        return escort_group

    #######################################################
    ####################### Graphics ######################
    #######################################################

    def SAR_plots(self):

        #plot the cost function
        plt.figure()
        plt.plot(np.arange(self.k-1), self.F[:self.k-1], '--', linewidth=3)
        plt.yscale('log')        
        plt.xlabel(r"Iterations $t$")
        plt.ylabel(r"Cost")
        plt.title("Evolution of the cost function")
        plt.grid()
        plt.show()

        #compute the gradient norms   
        grad_norm_z = np.linalg.norm(self.grad_z[:self.k-1], axis=1)  
        grad_norm_s = np.linalg.norm(self.grad_s[:self.k-1], axis=1)

        # Create a figure with subplots arranged horizontally
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        #plot the gradient norm of agent's positions
        ax[0].plot(np.arange(self.k-1), grad_norm_z)
        ax[0].set_yscale('log')
        ax[0].set_xlabel(r"Iterations $t$")
        ax[0].set_ylabel(r"Z Gradient Norm")
        ax[0].set_title("Evolution of the agents' position gradient norm")
        ax[0].grid()
        
        # Plot the gradient norm of agent's positions
        for i in range(self.N): ax[1].plot(np.arange(self.k-1), np.linalg.norm(self.S[i, :self.k-1, :] - self.true_barycenter[:self.k-1], axis=1))
        ax[1].set_yscale('log')
        ax[1].set_xlabel(r"Iterations $t$")
        ax[1].set_ylabel(r"SS - true_barycenter")
        ax[1].set_title("Convergence of barycenter's tracker")
        ax[1].grid()
        
        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()

    def SAR_animation(self):

        '''Generates an animation illustrating the trajectory of multiple robots
        within a multi-robot system over the course of optimization iterations.'''
        
        #create the subplots
        fig, ax = plt.subplots(figsize=(12, 12))

        ####################################
        ########### Static Limits ##########
        ####################################

        #determine the full house plot limits based on its dimensions
        if self.view_type == 'static':

            #extract points
            all_points = [point for room in self.environment.rooms_dict.values() for point in room]
            all_points += [self.environment.ENTRANCE, self.environment.EXIT]
            
            #determine static plot limits
            min_x, max_x = min(p[0] for p in all_points), max(p[0] for p in all_points)
            min_y, max_y = min(p[1] for p in all_points), max(p[1] for p in all_points)
            min_x = min(min_x, self.Z[:, 0, 0].min(), self.Z[:, self.k, 0].min())
            max_x = max(max_x, self.Z[:, 0, 0].max(), self.Z[:, self.k, 0].max())
            min_y = min(min_y, self.Z[:, 0, 1].min(), self.Z[:, self.k, 1].min())
            max_y = max(max_y, self.Z[:, 0, 1].max(), self.Z[:, self.k, 1].max())
    
        ##ITERATIONS
        for k in range(0, self.k, self.dt):

            ####################################
            ############## House ###############
            ####################################

            ax.clear()
            for (x0, y0), (x1, y1) in self.environment.walls_list:
                ax.plot([x0, x1], [y0, y1], 'k-', linewidth=3)
            for name, room in self.environment.rooms.items():
                x_points, y_points = zip(*room['points'])
                ax.fill(x_points, y_points, color=self.environment.colors.get(name, 'white'), alpha=0.5, label=name)
                ax.text(sum(x_points) / len(x_points), sum(y_points) / len(y_points), name, ha='center', va='center')
                if self.environment.show_obstacle_points:
                    ox, oy = zip(*room['obstacles'])
                    ax.scatter(ox, oy, c='black', s=20, zorder=5)

            ####################################
            ########### Trajectories ###########
            ####################################

            plt.scatter(self.Z[:, 0, 0], self.Z[:, 0, 1], marker='o', s=70, color='lightblue', label='Initial Positions')
            
            #plot robots' trajectories
            for i in range(self.N):
                ax.plot(self.Z[i, :k + 1, 0], self.Z[i, :k + 1, 1], linestyle='--', color=self.trajectory_colour, alpha=0.3)
                ax.plot(self.Z[i, k, 0], self.Z[i, k, 1], marker='o', markersize=self.robot_size, color=self.robot_colour)
                ax.plot(self.S[i, k, 0], self.S[i, k, 1], marker='.', markersize=5, color=self.barycenter_colour)

            #plot people's trajectories
            for idx in range(self.environment.people_trajectories.shape[0]):
                ax.scatter(self.environment.people_trajectories[idx, k, 0], self.environment.people_trajectories[idx, k, 1], c="green", s=35, marker='o')

            ####################################
            ############# Ambulance ############
            ####################################

            #evaluate ambulance's state at current iteration
            if k < len(self.environment.ambulance.ambulance_trajectory):
                pos, blinking, available = self.environment.ambulance.ambulance_trajectory[k]
                self.environment.ambulance.position = pos
                self.environment.ambulance.blinking = blinking
                self.environment.ambulance.available = available

            #plot the ambulance
            self.environment.ambulance.update_blink_state()
            self.environment.ambulance.plot_ambulance(ax)

            ####################################
            ########## Visualization ###########
            ####################################

            #plot the markers
            ax.plot(*self.environment.ENTRANCE, 'go', markersize=self.goal_marker_size)
            ax.plot(*self.environment.EXIT, 'ro', markersize=self.goal_marker_size)

            #plot local targets
            if self.show_local_targets:

                for i, tar in enumerate(self.local_targets):
                    plt.scatter(tar[:, 0], tar[:, 1], marker='x', s=15, color="salmon")
            
            #plot global targets
            if self.show_global_targets:

                for i, tar in enumerate(self.global_targets):
                    plt.scatter(tar[0], tar[1], marker='o', s=15, color="salmon")

            ####################################
            ############### Axes ###############
            ####################################
            
            #determine the plot's limits
            if self.view_type == 'dynamic':
                min_x, max_x = self.Z[:, k, 0].min(), self.Z[:, k, 0].max()
                min_y, max_y = self.Z[:, k, 1].min(), self.Z[:, k, 1].max()

            #set the plot's limits
            ax.set_xlim(min_x - 1, max_x + 1)
            ax.set_ylim(min_y - 1, max_y + 1)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('Multi-Room House Environment with Robot Trajectories')
            ax.set_xlabel('Width (units)')
            ax.set_ylabel('Height (units)')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True)
            
            ####################################
            ############### Logs ###############
            ####################################

            if k % (10 * self.dt) == 0:
                print("\n\n---------------------------------------------------------------")
                print("SAR Animation")
                print(f"Completion: {(k/self.k)*100}%")
                print("---------------------------------------------------------------")
            
            #dynamic time left indication
            time_left = self.MAXITERS - k
            ax.text(0.21, 0.95, f'Iterations left: {time_left}', ha='right', va='top', transform=ax.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

            #dynamic indication of people's states
            states = ['Steady', 'Escorted', 'Saved', 'Transported', 'Healed']
            state_text = '\n'.join([f"{state}: {self.environment.state_counts[k, i]} / {self.environment.n_people}" for i, state in enumerate(states)])
            ax.text(0.05, 0.05, state_text, ha='left', va='bottom', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            plt.pause(0.05)

        print("\n\n---------------------------------------------------------------")
        print("End of Multi-Robot Animation.")
        print("---------------------------------------------------------------")

        plt.show()

#############################################################################
############################### Main ########################################
#############################################################################

#start the program
message_handler("title")

#create the environment
env = Environment()

#create agents
agent = Agent(environment=env)

#end the program
message_handler("end")