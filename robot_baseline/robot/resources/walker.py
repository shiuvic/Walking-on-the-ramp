#!/usr/bin/env python
import sys
import time
from threading import Thread
import pybullet as p
import numpy as np
from robot_baseline.robot.resources.walking import acc_process
import os
from robot_baseline.robot.resources.core.op3 import OP3
from robot_baseline.robot.resources.walking.wfunc import WFunc
# from walking.wfunc import WFunc
# from core.op3 import OP3
import math
class Walker(OP3):
    """
    Class for making Darwin walk
    """

    def __init__(self,client,x_vel=1, y_vel=0, ang_vel=0, *args, **kwargs):
        OP3.__init__(self,client *args, **kwargs)

        self.running = False

        self.velocity = [0, 0, 0]
        self.walking = False

        self.x_vel = x_vel
        self.y_vel = y_vel
        self.ang_vel = ang_vel

        self.wfunc = WFunc()
        # ~ self.ready_pos=get_walk_angles(10)
        self.ready()
        self.stop_all = False

        self.parameters = {"swing_scale": 0.0,
                            "step_scale": 0.1,
                            "step_offset": 0.25,
                            "ankle_offset": 0.0,
                            "vx_scale": 0.2,
                            "vy_scale": 0.2,
                            "vt_scale": 0.1}
        self.walk_offset = {'hip_pitch': -0.063,
                            'hip_roll': 0.0,
                            'hip_yaw': 0.0,
                            'ank_pitch': 0.0,
                            'ank_roll': 0.0,
                            'knee': 0.0,
                            'sho_pitch': -0.063,
                            'sho_roll': -0.063,
                            'el': 0.0}
        self._th_walk = None
        self.show_cube_weight()

    def ready(self):
            self.ready_pos = self.wfunc.get(True, 0, [0, 0, 0])
            self.ready_pos.update({"r_sho_pitch": 0, "l_sho_pitch": 0,
                                   "r_sho_roll": -1.0, "l_sho_roll": 1.0,
                                   "r_el": 0.5, "l_el": -0.5,
                                   })

    def cmd_vel(self, vx, vy, vt):
        print("cmdvel", (vx, vy, vt))
        self.start()
        self.set_velocity(vx, vy, vt)

    def init_walk(self):
        """
        If not there yet, go to initial walk position
        """
        if self.get_dist_to_ready() > 0.02:
            self.set_angles_slow(self.ready_pos)

    def start(self):
        if not self.running:
            print("Start Walking")
            self.running = True
            self.init_walk()
            self._th_walk = Thread(target=self._do_walk)
            self._th_walk.start()
            self.walking = True

    def stop(self):
        if self.running:
            self.walking = False
            print("Waiting for stopped")
            while self._th_walk is not None:
                time.sleep(0.1)
            print("Stopped")
            self.running = False

    def set_velocity(self, x, y, t):
        self.velocity = [x, y, t]

    def _do_walk(self):
        """
        Main walking loop, smoothly update velocity vectors and apply corresponding angles
        """

        # Global walk loop
        n = 60
        phrase = True
        i = 0

        self.current_velocity = [0, 0, 0]
        self.stop_count = 0

        while self.walking or i < n or self.is_walking():
            if not self.walking:
                self.velocity = [0, 0, 0]
            elif not self.is_walking() and i == 0:  # Do not move if nothing to do and already at 0
                self.update_velocity(self.velocity, n)
                time.sleep(1 / 480.)
                continue
            x = float(i) / n
            angles = self.wfunc.get(phrase, x, self.current_velocity)
            self.update_velocity(self.velocity, n)
            self.set_angles(angles)
            i += 1
            if i > n:
                i = 0
                phrase = not phrase
                self.wfunc._update_new_param(self.parameters)

            time.sleep(1. / 240.)
            if self.stop_all:
                break
        self._th_walk = None

    def is_walking(self):
        e = 0.02
        for v in self.current_velocity:
            if abs(v) > e: return True
        return False

    def rescale(self, angles, coef):
        z = {}
        for j, v in angles.items():
            offset = self.ready_pos[j]
            v -= offset
            v *= coef
            v += offset
            z[j] = v
        return z

    def update_velocity(self, target, n):
        a = 3 / float(n)
        b = 1 - a
        self.current_velocity = [a * t + b * v for (t, v) in zip(target, self.current_velocity)]

    def get_dist_to_ready(self):
        angles = self.get_angles()
        return get_distance(self.ready_pos, angles)

    def reset(self):
        self.stop()
        p.resetBasePositionAndOrientation(self.robot, self.op3StartPos, self.op3StartOrientation)
        self.start()
        self.set_velocity(self.x_vel, self.y_vel, 0)

    def apply_action(self, action, step):
        state = self.get_state()
        alpha = math.log(step, 300)
        if step > 300:
            alpha = 1
        action = action * 0.8 * alpha
        action = action + state
        # step_scale, step_offset, vx_scale, vy_scale, sho_pitch, sho_roll = action
        step_scale, step_offset, sho_roll = action
        step_scale = max(min(step_scale, 2.), -2.)
        step_offset = max(min(step_offset, 2.), -2.)
        # vx_scale = max(min(vx_scale, 2.), -2.)
        # vy_scale = max(min(vy_scale, 2.), -2.)
        # sho_pitch = max(min(sho_pitch, 2.), -2.)
        sho_roll = max(min(sho_roll, 2.), -2.)


        self.parameters = {"swing_scale": 0.0,
                      "step_scale": step_scale,
                      "step_offset": step_offset,
                      "ankle_offset": 0.0,
                      "vx_scale": 0.23, #0.23
                      "vy_scale": 0.23, #0.23
                      "vt_scale": 0.1}

        self.walk_offset = {'hip_pitch': -0.063,
                       'hip_roll': 0.0,
                       'hip_yaw': 0.0,
                       'ank_pitch': 0.0,
                       'ank_roll': 0.0,
                       'knee': 0.0,
                       'sho_pitch': -0.063, ##-0.063
                       'sho_roll': sho_roll,  ##-0.063
                       'el': 0.0}
        self.wfunc._update_new_offset(self.walk_offset)
        # self.wfunc._update_new_param(self.parameters, self.walk_offset)

    def get_state(self):
        # parameters = {"step_scale",
        #               "step_offset",
        #               "vx_scale",
        #               "vy_scale"}
        # walk_offset = {'sho_pitch',
        #                'sho_roll'}

        parameters = {"step_scale",
                      "step_offset"}
        walk_offset = {'sho_roll'}

        state = []
        for k, v in self.parameters.items():
            if k in parameters:
                state.append(v)
        for k, v in self.walk_offset.items():
            if k in walk_offset:
                state.append(v)

        return np.array(state)


    def get_obsevervation(self):
        acc = self.get_acc()
        acc = acc_process.ob(acc)
        # acc = np.asarray(acc).flatten()
        l_ank_pitch = self.get_force(4)
        l_ank_roll = self.get_force(5)
        r_ank_pitch = self.get_force(10)
        r_ank_roll = self.get_force(11)
        # print(acc)
        # ob = np.append(acc, [l_ank_pitch, l_ank_roll, r_ank_pitch, r_ank_roll])
        ob = {
            'accx': acc[0],
            'accy': acc[1],
            'accz': acc[2],
            # 'torque': np.array([l_ank_pitch, l_ank_roll, r_ank_pitch, r_ank_roll])
        }
        return ob

    def close(self):
        self.stop()
        # self._th_walk.terminate()
        # self.run_sim_th.terminate()
        self.stop_all = True
        # self._th_walk.join()
        self.stop_run = True
        self.NOOO()


def interpolate(anglesa, anglesb, coefa):
    z = {}
    joints = anglesa.keys()
    for j in joints:
        z[j] = anglesa[j] * coefa + anglesb[j] * (1 - coefa)
    return z


def get_distance(anglesa, anglesb):
    d = 0
    joints = anglesa.keys()
    if len(joints) == 0: return 0
    for j in joints:
        d += abs(anglesb[j] - anglesa[j])
    d /= len(joints)
    return d


if __name__ == '__main__':
    wewe = p.connect(p.GUI)
    walker = Walker(wewe)
    # time.sleep(1)
    walker.reset()
    while True:
        # walker.show_force(3)
        # walker.get_obsevervation()
        # acc = walker.get_acc()
        # walker.show_acc(acc)
        time.sleep(0.1)
        # walker.close()



