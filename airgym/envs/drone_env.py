import gym
from gym import spaces

import airsim
from airgym.envs.airsim_env import AirSimEnv
from airsim import Vector3r, MultirotorClient
from pyproj import Proj
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt
import time
import random
from PIL import Image
from argparse import ArgumentParser
SRID = 'EPSG:5555'
ORIGIN = (13.649799, 100.494283, 3.3)

class AirSimDroneEnv(AirSimEnv):

    def __init__(self, ip_address, step_length, destination, **kwargs):
        # super().__init__(image_shape)
        super().__init__(**kwargs)
        self.step_length = step_length
        # self.image_shape = image_shape
        self.destination = destination
        # self.middle_pixel = 0
        self.proj = Proj(init=srid)
        self.origin_proj = self.proj(*self.origin[0:2]) + (self.origin[2],)
        
        self.total_rewards = float(0.0)
        self.distance = 300.0
        # self.pass1 = False
        # self.pass2 = False
        # self.pass3 = False

        #NED coordinate system (X,Y,Z) : +X is North, +Y is East and +Z is Down
        self.MAX_ALTITUDE = -60
        self.AVERAGE_ALTITUDE = -30
        self.MIN_ALTITUDE = -10
        self.MAX_SOUTH =  -60
        self.MAX_NORTH = 250
        self.MAX_LEFT = -60
        self.MAX_RIGHT = 60

        # self.action_space = spaces.Discrete(6)    
        self.action_space = spaces.Discrete(4)

        # self.observation_space = spaces.Dict({ 'depth_cam': spaces.Box(low=0, high=255, shape=(84, 84, 1)),
        #                                        'position': spaces.Box(low=-60, high=250, shape=(3,)) })
        #                                     #    'collision' : spaces.Discrete(2) })
        
        self.state = {'gps': np.zeros(3),
                      'prev_gps': np.zeros(3) }

        self.drone = airsim.MultirotorClient(ip=ip_address,srid=SRID, origin=ORIGIN)
        self._setup_flight()
        # self.image_request = airsim.ImageRequest(
        #     3, airsim.ImageType.DepthPerspective, True, False
        # )

    def lonlatToProj(self, lon, lat, z, inverse=False):
        proj_coords = self.proj(lon, lat, inverse=inverse)
        return proj_coords + (z,)

    def projToAirSim(self, x, y, z):
        x_airsim = x - self.origin_proj[0]
        y_airsim = y - self.origin_proj[1]
        z_airsim = -z + self.origin_proj[2]
        return (x_airsim, -y_airsim, z_airsim)

    def lonlatToAirSim(self, lon, lat, z):
        return self.projToAirSim(*self.lonlatToProj(lon, lat, z))

    def nedToProj(self, x, y, z):
        """
        Converts NED coordinates to the projected map coordinates
        Takes care of offset origin, inverted z, as well as inverted y axis
        """
        x_proj = x + self.origin_proj[0]
        y_proj = -y + self.origin_proj[1]
        z_proj = -z + self.origin_proj[2]
        return (x_proj, y_proj, z_proj)

    def nedToGps(self, x, y, z):
        return self.lonlatToProj(*self.nedToProj(x, y, z), inverse=True)

    def getGpsLocation(self):
        """
        Gets GPS coordinates of the vehicle.
        """
        self.drone_state = self.drone.getMultirotorState()
        # self.state["prev_position"] = self.state["position"]
        self.state["prev_gps"] = self.state["gps"]
        pos = self.simGetGroundTruthKinematics().position
        gps = self.nedToGps(pos.x_val, pos.y_val, pos.z_val)
        self.state["gps"] = gps
        # self.state["position"] = pos
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        
        return gps

    def moveToPositionAsyncGeo(self, gps=None, proj=None, **kwargs):
        """
        Moves to the a position that is specified by gps (lon, lat, +z) or by the projected map 
        coordinates (x, y, +z).  +z represent height up.
        """
        coords = None
        if gps is not None:
            coords = self.lonlatToAirSim(*gps)
        elif proj is not None:
            coords = self.projToAirSim(*proj)
        if coords:
            return self.moveToPositionAsync(coords[0], coords[1], coords[2], **kwargs)
        else:
            print('Please pass in GPS (lon,lat,z), or projected coordinates (x,y,z)!')

    def __del__(self):
        self.drone.reset()


    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
    

    def _setup_starting_position(self):
        # Set starting position and velocity
        # Take off
        self.drone.takeoffAsync(timeout_sec=5).join()
        gps = self.getGpsLocation()
        # Go up by 15 meters
        gps_new = (gps[0], gps[1], gps[2] + 15.0)
        self.moveToPositionAsyncGeo(gps=gps_new, velocity=5).join()
        # self.drone.moveToPositionAsync(0, 0, -30, 5).join()
        # self.drone.moveByVelocityAsync(1, 0, 0, 5).join()


    # def _setup_destination(self):
    #     #random area a,b,c (ratio 1:2:3)
    #     area = random.randrange(1,7) # a -> 1 | b -> 2,3 | c -> 4,5,6
    #     if area < 2: # a
    #         x = random.randrange(230,271)
    #         y = random.randrange(45,71)
    #         print("\nDestination A ", [x,y,self.AVERAGE_ALTITUDE])
    #     elif area < 4: # b
    #         x = random.randrange(290,351)
    #         y = random.randrange(-75,-44)
    #         print("\nDestination B ", [x,y,self.AVERAGE_ALTITUDE])
    #     else: # c
    #         x = random.randrange(450,516)
    #         y = random.randrange(-75,56)
    #         print("\nDestination C ", [x,y,self.AVERAGE_ALTITUDE])


    #     return np.array([x,y,self.AVERAGE_ALTITUDE])

    

    # def transform_obs(self, responses):
    #     img1d = np.array(responses[0].image_data_float, dtype=np.float)
    #     img1d = np.where(img1d > 255, 255, img1d)
    #     img2d = np.reshape(img1d, (responses[0].height, responses[0].width))


    #     image = Image.fromarray(img2d)
    #     im_final = np.array(image.resize((84, 84)).convert("L"))

    #     return im_final.reshape([84, 84, 1])


    # def _get_obs(self):
    #     responses = self.drone.simGetImages([self.image_request])
    #     image = self.transform_obs(responses)
    #     self.drone_state = self.drone.getMultirotorState()

    #     self.state["prev_position"] = self.state["position"]
    #     self.state["position"] = self.drone_state.kinematics_estimated.position
    #     self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

    #     collision = self.drone.simGetCollisionInfo().has_collided
    #     self.state["collision"] = collision

    #     return image


    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            0,
            4,
        ).join()
        
    def dist(lat_pos, long_pos, lat_des, long_des):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lat_pos, long_pos, lat_des, long_des = map(radians, [lat_pos, long_pos, lat_des, long_des])
        # haversine formula 
        dlon = long_des - long_pos 
        dlat = lat_des - lat_pos 
        a = sin(dlat/2)**2 + cos(lat_pos) * cos(lat_des) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        # Radius of earth in kilometers is 6371
        km = 6371* c
        return km

    def _compute_reward(self, action):
        thresh_dist = 1
        beta = 1

        z = -10
        # pts = [np.array([-0.55265, -31.9786, -19.0225]),np.array([48.59735, -63.3286, -60.07256]),np.array([193.5974, -55.0786, -46.32256]),np.array([369.2474, 35.32137, -62.5725]),np.array([541.3474, 143.6714, -32.07256]),]

        # quad_pt = np.array(list((self.state["position"].x_val, self.state["position"].y_val,self.state["position"].z_val,)))
        pos = self.state["gps"]
        des = self.destination
        dist = 10000000
        for i in range(0, len(self.destination) - 1):
            dist = self.dist(pos[0],pos[1],des[0],des[1])

            if dist > thresh_dist:
                reward = -100
            elif dist < 0.1:
                reward = 100
            else:
                reward_dist = math.exp(-beta * dist) - 0.5
                reward_speed = (np.linalg.norm([self.state["velocity"].x_val, self.state["velocity"].y_val, self.state["velocity"].z_val,])- 0.5)
                reward = reward_dist + reward_speed

        # rewards = float(0.0)
        # done = False

        # if self.state["collision"]: # collide
        #     rewards = -100
        #     done = True
        # elif self.state["position"].x_val < -80 or self.state["position"].y_val > 80 or self.state["position"].y_val < -80 :
        #     done = True
        # elif self.state["position"].x_val > 200:
        #     rewards = 25
        #     done = True
        # elif self.state["position"].x_val > 150 and not self.pass3:
        #     rewards = 25
        #     self.pass3 = True
        # elif self.state["position"].x_val > 100 and not self.pass2:
        #     rewards = 25
        #     self.pass2 = True
        # elif self.state["position"].x_val > 50 and not self.pass1:
        #     rewards = 25
        #     self.pass1 = True
        done = 0
        if reward <= -10:
            done = 1

        return reward, done


    def step(self, action):
        self._do_action(action)
        gps = self.getGpsLocation()
        reward, done = self._compute_reward(action)

        # if done:

        #     self.total_rewards = 0
            
        if action == 0:
            movement = 'fore'
        elif action == 1:
            movement = 'back'
        elif action == 2:
            movement = 'right'
        elif action == 3:
            movement = 'left'
        elif action == 4:
            movement = 'down'
        else:
            movement = 'up'


        # print("reward ", format(reward, ".2f"),  "\t  done  " + str(done), "\t action ", movement, "\t    velocity [", format(self.state["velocity"].x_val, ".1f"), ",\t",  format(self.state["velocity"].y_val, ".1f"), ",\t" , format(self.state["velocity"].z_val, ".1f"), "]")
        # print("reward ", format(reward, ".2f"),  "\t  done  " + str(done), "\t action ", movement, "\t    position [", format(self.state["position"].x_val, ".1f"), ",\t",  format(self.state["position"].y_val, ".1f"), ",\t" , format(self.state["position"].z_val, ".1f"), "]")
        print("reward ", format(reward, ".0f"),  "\t  done  " + str(done), "\t action ", movement)
        if done:
            # self.pass1 = False
            # self.pass2 = False
            # self.pass3 = False
            print('Done!\n')

        # if done:
        #     self.total_rewards = 0
        #     if self.done_flag == 0:
        #         print("done : collision")
        #     elif self.done_flag == 1:
        #         print("done : out of range")
        #         print("destination : ", self.destination, "\tposition : [", format(self.state["position"].x_val, ".1f"), ",",  format(self.state["position"].y_val, ".1f"), "," , format(self.state["position"].z_val, ".1f"), "]" )
        #     elif self.done_flag == 2:
        #         print("done : reach destination")
        #     else:
        #         print("done : total rewards < -100")
        #     self.done_flag = -1

        # if done:
        #     if self.state["collision"]:
        #         print('collision\n')
        #     else:
        #         print("out of range -> position : [", format(self.state["position"].x_val, ".1f"), ",",  format(self.state["position"].y_val, ".1f"), "," , format(self.state["position"].z_val, ".1f"), "]\n"  )

        return gps, reward, done, self.state


    def reset(self):
        # self.destination = self._setup_destination()
        self.distance = 300.0
        self._setup_flight()
        self._setup_starting_position()
        time.sleep(5)
        self.drone.landAsync().join()
        return self.getGpsLocation


    def interpret_action(self, action):
        #NED coordinate system (X,Y,Z) : +X is North, +Y is East and +Z is Down
        # if action == 0: # forward
        #     quad_offset = (self.step_length, 0, 0)
        #     self.movement = 'fore'
        # elif action == 1: # slide right
        #     quad_offset = (0, self.step_length, 0)
        #     self.movement = 'right'
        # elif action == 2: # downward
        #     quad_offset = (0, 0, self.step_length)
        #     self.movement = 'down'
        # elif action == 3: # backward
        #     quad_offset = (-self.step_length, 0, 0)
        #     self.movement = 'back'
        # elif action == 4: # slide left
        #     quad_offset = (0, -self.step_length, 0)
        #     self.movement = 'left'
        # elif action == 5: # upward
        #     quad_offset = (0, 0, -self.step_length)
        #     self.movement = 'up'


        if action == 0: # forward
            quad_offset = (self.step_length, 0, 0)
        elif action == 1: # back
            quad_offset = (-self.step_length, 0, 0)
        elif action == 2: # slide right
            quad_offset = (0, self.step_length, 0)
        elif action == 3: # slide left
            quad_offset = (0, -self.step_length, 0)
        elif action == 4: # downward
            quad_offset = (0, 0, self.step_length)
        elif action == 5: # upward
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
