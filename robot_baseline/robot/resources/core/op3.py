import time
from threading import Thread
import numpy as np
np.set_printoptions(precision=2)

import pybullet as p
import pybullet_data
import sys
import math
import matplotlib.pyplot as plt
import matplotlib
import random
from robot_baseline.robot.resources.cube_weight import clean_weight, save_weight
from robot_baseline.robot.resources.ramp import load


matplotlib.use('tkagg')
if sys.platform == "win32":
    from ctypes import windll

    timeBeginPeriod = windll.winmm.timeBeginPeriod
    timeBeginPeriod(1)

op3_joints = ['l_hip_yaw',
              'l_hip_roll',
              'l_hip_pitch',
              'l_knee',
              'l_ank_pitch',
              'l_ank_roll',
              'r_hip_yaw',
              'r_hip_roll',
              'r_hip_pitch',
              'r_knee',
              'r_ank_pitch',
              'r_ank_roll',
              'l_sho_pitch',
              'l_sho_roll',
              'l_el',
              'r_sho_pitch',
              'r_sho_roll',
              'r_el',
              'head_pan',
              'head_tilt']


class OP3:
    def __init__(self, fallen_reset=False, sim_speed=1.0, client=None, *args, **kwargs):
        self.fallen_reset = fallen_reset
        # self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        self.physicsClient = client
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.speed = sim_speed

        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.8)
        slope = load()
        print(math.sin(math.pi/slope)*1.5)
        # self.op3StartPos = [0, 0, 0.5-0.03]
        self.op3StartPos = [0, 0, math.sin(math.pi/slope) + 0.3]

        # self.weightPos = [2.5, 0, 3]
        # self.weightPos = [0.1, 0, 0.6-0.03]
        self.weightPos = [0.1, 0, self.op3StartPos[2]+0.1]

        # self.rampPos = [1.6, 0, 0.1178-0.1]
        self.rampPos = [math.cos(math.pi/slope)*1.45, 0, math.sin(math.pi/slope)*1.5]
        # self.planePos = [-1.3823, 0, 0.2356-0.1]

        # self.rampOri = p.getQuaternionFromEuler([0, math.pi/40, 0])

        self.rampOri = p.getQuaternionFromEuler([0, -math.pi / slope, 0])

        self.op3StartOrientation = p.getQuaternionFromEuler([0, -math.pi / slope, 0])

        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0])
        # self.planeId = p.loadURDF("plane.urdf", [0, 0, -0.1])

        # self.robot = p.loadURDF("./robot/models/robotis_op4.urdf", self.op3StartPos, self.op3StartOrientation)
        # self.cube = p.loadURDF("./robot/models/Bell_weight.urdf", self.weightPos)
        self.robot = p.loadURDF("/Walking-on-the-ramp/robot_baseline/robot/models/robotis_op4.urdf", self.op3StartPos, self.op3StartOrientation)
        self.cube = p.loadURDF("/Walking-on-the-ramp/robot_baseline/robot/models/Bell_weight.urdf", self.weightPos)
        self.ramp = p.loadURDF("/Walking-on-the-ramp/robot_baseline/robot/models/Ramp.urdf", self.rampPos, self.rampOri)
        # self.plane = p.loadURDF("H:/Walking/robot_baseline/robot/models/Ramp.urdf", self.planePos)
        p.addUserDebugText(str(round((180 / slope[0]),  2)), [0.0, 0.0, 0.1], textSize=2.0, textColorRGB=[0, 0, 0],parentObjectUniqueId=self.ramp, parentLinkIndex=1)

        self.numJoints = p.getNumJoints(self.robot)
        self.targetVel = 0
        self.maxForce = 100
        self.camera_follow()
        self.accx = []
        self.accy = []
        self.accz = []
        self.acc = []
        self.angles = None
        self.update_angle_th()
        # self.check_reset_th()
        self.stop_run = False
        self.sim = None
        # self.run_acc_th()
        # self.update_camera_th()
        self.clear_force()
        self.run_sim_th()

        self._set_joint()
        # print(p.getDynamicsInfo(self.robot, -1))
        # self.throw()
        self.joints = op3_joints
        self.show_cube_weight()
        mass = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/cube_weight.npy')[0]
        self.set_cube_weight(mass)



    def ready(self):
            self.ready_pos = self.wfunc.get(True, 0, [0, 0, 0])
            self.ready_pos.update({"r_sho_pitch": 0, "l_sho_pitch": 0,
                                   "r_sho_roll": -1.0, "l_sho_roll": 1.0,
                                   "r_el": 0.5, "l_el": -0.5,
                                   })

    @property
    def sim_speed(self):
        return self.speed

    def get_orientation(self):
        _, orientation = p.getBasePositionAndOrientation(self.robot)
        return np.array(orientation)

    def get_position(self):
        position, _ = p.getBasePositionAndOrientation(self.robot)
        return np.array(position)

    def camera_follow(self, distance=1.0, pitch=-35.0, yaw=50.0):
        lookat = self.get_position() - [0, 0, 0.1]
        p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    def is_fallen(self):
        """Decide whether the rex has fallen.
        If the up directions between the base and the world is large (the dot
        product is smaller than 0.85), the rex is considered fallen.
        Returns:
          Boolean value that indicates whether the rex has fallen.
        """
        rot_mat = p.getMatrixFromQuaternion(self.get_orientation())
        local_up = rot_mat[6:]
        return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85

    def get_angles(self):
        if self.joints is None: return None
        if self.angles is None: return None
        return dict(zip(self.joints, self.angles))

    def set_angles(self, angles):
        for j, v in angles.items():
            if j not in self.joints:
                AssertionError("Invalid joint name " + j)
                continue
            p.setJointMotorControl(self.robot, op3_joints.index(j), p.POSITION_CONTROL, v, self.maxForce)

    def set_angles_slow(self, stop_angles, delay=2):
        start_angles = self.get_angles()
        start = time.time()
        stop = start + delay
        while True:
            t = time.time()
            if t > stop: break
            ratio = (t - start) / (delay / self.sim_speed)
            angles = interpolate(stop_angles, start_angles, ratio)
            self.set_angles(angles)
            time.sleep(0.1 / self.sim_speed)

    def run_sim_th(self):
        def _cb_sim():
            while True:
                p.stepSimulation()
                time.sleep(1.0 / (240.0 * self.sim_speed))
                # self.camera_follow(distance=0.5)
                if self.stop_run:
                    break

        self.sim = Thread(target=_cb_sim)
        self.sim.start()

    def NOOO(self):
        self.stop_run = True
        self.sim.join()



    def check_reset_th(self):
        def _cb_reset():
            self.prev_state = 1.0
            while True:
                # self.show_force(4)
                # self.get_acceleration()
                curr_state = p.readUserDebugParameter(self.bt_rst)
                if curr_state != self.prev_state or (self.fallen_reset and self.is_fallen()):
                    self.reset_and_start()
                    self.prev_state = curr_state
                time.sleep(0.001)

        Thread(target=_cb_reset).start()

    # def update_angle_th(self):
    #     def _cb_angles():
    #         while True:
    #             # self.show_force(4)
    #             angles = []
    #             for joint in range(self.numJoints):
    #                 angles.append(p.getJointState(self.robot, joint)[0])
    #             self.angles = angles
    #             time.sleep(0.001)
    #
    #     Thread(target=_cb_angles).start()

    def update_angle_th(self):
        angles = []
        for joint in range(self.numJoints):
            angles.append(p.getJointState(self.robot, joint)[0])
        self.angles = angles
        time.sleep(0.001)


    def _set_joint(self):
        for joint in range(self.numJoints):
            # print(p.getJointInfo(self.robot, joint))
            p.setJointMotorControl(self.robot, joint, p.POSITION_CONTROL, self.targetVel, self.maxForce)

    def run(self):

        try:
            self.set_angles_slow(self.ready_pos)
            while True:
                p.stepSimulation()
                time.sleep(1. / 240.)
                self.show_force(4)
        finally:
            OP3Pos, OP3Orn = p.getBasePositionAndOrientation(self.robot)
            # print(OP3Pos, OP3Orn)
            p.disconnect()

    def reset_and_start(self):
        p.resetBasePositionAndOrientation(self.robot, self.op3StartPos, self.op3StartOrientation)
        p.resetBasePositionAndOrientation(self.cube, self.weightPos, self.op3StartOrientation)

    def update_camera_th(self):
        def setCameraPicAndGetPic():
            while True:
                width = 224
                height = 224
                BASE_RADIUS = 3
                BASE_THICKNESS = 3
                # basePos = p.getLinkState(self.robot,19)[0]
                basePos, baseOrientation = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClient)

                matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=self.physicsClient)
                tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # 变换后的x轴
                tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # 变换后的z轴

                basePos = np.array(basePos)
                # 摄像头的位置
                cameraPos = basePos + BASE_RADIUS * tx_vec + 0.5 * BASE_THICKNESS * tz_vec
                targetPos = cameraPos + 1 * tx_vec

                viewMatrix = p.computeViewMatrix(
                    cameraEyePosition=cameraPos,
                    cameraTargetPosition=targetPos,
                    cameraUpVector=tz_vec,
                    physicsClientId=self.physicsClient
                )
                projectionMatrix = p.computeProjectionMatrixFOV(
                    fov=50.0,  # 摄像头的视线夹角
                    aspect=1.0,
                    nearVal=0.01,  # 摄像头焦距下限
                    farVal=20,  # 摄像头能看上限
                    physicsClientId=self.physicsClient
                )

                width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                    width=width, height=height,
                    viewMatrix=viewMatrix,
                    projectionMatrix=projectionMatrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    physicsClientId=self.physicsClient
                )
        Thread(target=setCameraPicAndGetPic).start()

    def throw(self):
        while True:
            p.resetBaseVelocity(self.cube, [-10.5, 0.0, -10.0])
            time.sleep(1)
            p.resetBasePositionAndOrientation(self.cube, self.weightPos, self.op3StartOrientation)

    def get_mass(self, body):
        mass = p.getDynamicsInfo(body)[0]
        return mass

    def get_acceleration(self):
        t = 1. / 240.
        pos, ang = p.getBasePositionAndOrientation(self.robot)
        ang = p.getEulerFromQuaternion(ang)
        # 角度轉成方向
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        # 取得xy座標
        pos = pos[:2]
        # 取得速度
        pre_vel = np.zeros(3)
        new_vel = np.zeros(3)
        pre_vel[0] = p.getBaseVelocity(self.robot)[0][0]
        pre_vel[1] = p.getBaseVelocity(self.robot)[0][1]
        pre_vel[2] = p.getBaseVelocity(self.robot)[0][2]
        # 串街 位置, 方向, 加速度
        time.sleep(t)
        new_vel[0] = p.getBaseVelocity(self.robot)[0][0]
        new_vel[1] = p.getBaseVelocity(self.robot)[0][1]
        new_vel[2] = p.getBaseVelocity(self.robot)[0][2]
        acc = np.zeros(3)
        acc[0] = (new_vel[0] - pre_vel[0]) / t
        acc[1] = (new_vel[1] - pre_vel[1]) / t
        acc[2] = (new_vel[2] - pre_vel[2]) / t
        # self.show_acc(acc)
        # self.acc = acc
        return acc

    def run_acc_th(self):

        def _cb_acc():
            while True:
                p.stepSimulation()
                pre_vel = np.zeros(3)
                new_vel = np.zeros(3)
                pre_vel[0] = p.getBaseVelocity(self.robot)[0][0]
                pre_vel[1] = p.getBaseVelocity(self.robot)[0][1]
                pre_vel[2] = p.getBaseVelocity(self.robot)[0][2]
                # 串街 位置, 方向, 加速度 當作 observation
                time.sleep(1.0 / (240.0 * self.sim_speed))
                new_vel[0] = p.getBaseVelocity(self.robot)[0][0]
                new_vel[1] = p.getBaseVelocity(self.robot)[0][1]
                new_vel[2] = p.getBaseVelocity(self.robot)[0][2]
                acc = np.zeros(3)
                acc[0] = (new_vel[0] - pre_vel[0]) / (1.0 / (240.0 * self.sim_speed))
                acc[1] = (new_vel[1] - pre_vel[1]) / (1.0 / (240.0 * self.sim_speed))
                acc[2] = (new_vel[2] - pre_vel[2]) / (1.0 / (240.0 * self.sim_speed))
                self.acc = [acc[0], acc[1], acc[2]]
        Thread(target=_cb_acc).start()

    def show_acc(self,acc):
        self.accx.append(acc[0])
        self.accy.append(acc[1])
        self.accz.append(acc[2])
        # fig = plt.figure()
        # plt.subplot(131)

        plt.plot(self.accx, label='X')
        # plt.subplot(132)
        plt.plot(self.accy, label='Y')
        # plt.subplot(133)
        plt.plot(self.accz, label='Z')
        # plt.tight_layout()
        plt.legend()
        plt.draw()
        plt.pause(1./240.)
        plt.clf()

    def show_force(self, num):
        x = p.getJointState(self.robot, num)
        # print(x)
        self.accx.append(x[3])
        plt.plot(self.accx)
        # plt.legend()
        plt.draw()
        plt.pause(1./240.)
        plt.clf()

    def get_force(self, num):
        return p.getJointState(self.robot, num)[3]

    def save_force(self):
        l_ank_pitch = self.get_force(4)
        l_ank_roll = self.get_force(5)
        r_ank_pitch = self.get_force(10)
        r_ank_roll = self.get_force(11)

        l_ank_pitch_arr = np.load('l_ank_pitch_force.npy')
        l_ank_roll_arr = np.load('l_ank_roll_force.npy')
        r_ank_pitch_arr = np.load('r_ank_pitch_force.npy')
        r_ank_roll_arr = np.load('r_ank_roll_force.npy')

        l_ank_pitch_arr = np.append(l_ank_pitch_arr, l_ank_pitch)
        l_ank_roll_arr = np.append(l_ank_roll_arr, l_ank_roll)
        r_ank_pitch_arr = np.append(r_ank_pitch_arr, r_ank_pitch)
        r_ank_roll_arr = np.append(r_ank_roll_arr, r_ank_roll)

        np.save('l_ank_pitch_force.npy', l_ank_pitch_arr)
        np.save('l_ank_roll_force.npy', l_ank_roll_arr)
        np.save('r_ank_pitch_force.npy', r_ank_pitch_arr)
        np.save('r_ank_roll_force.npy', r_ank_roll_arr)

    def clear_force(self):
        arr = []
        np.save('l_ank_pitch_force.npy', arr)
        np.save('l_ank_roll_force.npy', arr)
        np.save('r_ank_pitch_force.npy', arr)
        np.save('r_ank_roll_force.npy', arr)

    def get_acc(self):
        self.acc = self.get_acceleration()
        return self.acc

    def chane_cube_weight(self):
        # mass = random.random()
        # mass = round(mass, 3)
        rand = random.randint(1, 3)
        if rand == 1:
            mass = 0.2
        elif rand == 2:
            mass = 0.4
        elif rand == 3:
            mass = 0.8

        p.changeDynamics(self.cube, -1, mass=mass)
        p.addUserDebugText(str(mass)+"kg", [0.0, -0.15, 0.5], textSize=3.0, textColorRGB=[0, 0, 0], replaceItemUniqueId=self.txt_id, parentObjectUniqueId=self.robot, parentLinkIndex=1)
        save_weight(mass)


    def set_cube_weight(self, mass):
        p.changeDynamics(self.cube, -1, mass=mass)
        p.addUserDebugText(str(mass)+"kg", [0.0, -0.15, 0.5], textSize=3.0, textColorRGB=[0, 0, 0], replaceItemUniqueId=self.txt_id, parentObjectUniqueId=self.robot, parentLinkIndex=1)
        print('cube mass = ', mass)

    def show_cube_weight(self):
        mass = p.getDynamicsInfo(self.cube, -1)[0]
        self.txt_id = p.addUserDebugText(str(mass)+"kg", [0.0, -0.15, 0.5], textSize=3.0, textColorRGB=[0, 0, 0], parentObjectUniqueId=self.robot, parentLinkIndex=1)

    def rand_ramp(self):
        ramp = random.randint(1, 3)
        if ramp == 1:
            a = [25]
        elif ramp == 2:
            a = [30]
        elif ramp == 3:
            a = [40]
        # ramp = random.randint(25, 40)
        # a = [ramp]
        np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/ramp.npy', a)

    # def stop_all_thread(self):
    #     thread_kill(self.sim.ident, SIGTSTP)


def interpolate(anglesa, anglesb, coefa):
    z = {}
    joints = anglesa.keys()
    for j in joints:
        z[j] = anglesa[j] * coefa + anglesb[j] * (1 - coefa)
    return z


if __name__ == '__main__':
    op3 = OP3(p.connect(p.GUI))
    # op3.run()
    pass
