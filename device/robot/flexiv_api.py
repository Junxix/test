"""Flexiv robot ethernet communication python wrapper 2.8.2 api.

Author: Junfeng Ding, Hongjie Fang, Hao-Shu Fang
"""

import time

from . import flexivrdk
import numpy as np
import threading
from multiprocessing import shared_memory

from transforms3d.euler import quat2euler

LOG_LEVEL = "info"


class FlexivApi:
    """Flexiv Robot Control Class.

    This class provides python wrapper for Flexiv robot control in different modes.
    Features include:
        - torque control mode
        - plan execution mode
        - primitive excution mode
        - get robot status information
        - force torque record
    """

    logger_name = "FlexivApi"

    def __init__(self, serial, with_streaming = False):
        """initialize.

        Args:
            serial: robot serial number

        Raises:
            RuntimeError: error occurred when ip_address is None.
        """
        self.mode = flexivrdk.Mode
        self.robot = flexivrdk.Robot(serial)
        if self.robot.fault():
            if not self.robot.ClearFault():
                raise RuntimeError("Cannot clear faults on the robot.")
        self.robot.Enable()
        while not self.robot.operational():
            time.sleep(0.5)
        self.current_mode = None
        time.sleep(2)
        self.with_streaming = with_streaming
        self._prepare_shm()
        self.is_streaming = False


    def _prepare_shm(self):
        '''
        Prepare shared memory objects.
        '''
        if self.with_streaming:
            tcp = self.get_tcp_info()
            self.shm_tcp = shared_memory.SharedMemory(name = 'tcp', create = True, size = tcp.nbytes)
            self.shm_tcp_buf = np.ndarray(tcp.shape, dtype = tcp.dtype, buffer = self.shm_tcp.buf)
            joint = self.get_joint_info()
            self.shm_joint = shared_memory.SharedMemory(name = 'joint', create = True, size = joint.nbytes)
            self.shm_joint_buf = np.ndarray(joint.shape, dtype = joint.dtype, buffer = self.shm_joint.buf)
    
    def streaming(self, delay_time = 0.0):
        '''
        Start streaming.

        Parameters
        ----------
        delay_time: float, optional, default: 5.0, the delay time before collecting data.
        '''
        if self.with_streaming is False:
            raise AttributeError('If you want to use streaming function, the "with_streaming" attribute should be set True.')
        self.thread = threading.Thread(target = self.streaming_thread, kwargs = {'delay_time': delay_time})
        self.thread.setDaemon(True)
        self.thread.start()
    
    def streaming_thread(self, delay_time = 0.0):
        time.sleep(delay_time)
        self.is_streaming = True
        print('[Robot] Start streaming ...')
        while self.is_streaming:
            tcp = self.get_tcp_info()
            joint = self.get_joint_info()
            self.shm_tcp_buf[:] = tcp[:]
            self.shm_joint_buf[:] = joint[:]
            time.sleep(0.05)
    
    def stop_streaming(self, permanent = True):
        '''
        Stop streaming process.

        Parameters
        ----------
        permanent: bool, optional, default: True, whether the streaming process is stopped permanently.
        '''
        self.is_streaming = False
        self.thread.join()
        if permanent:
            self._close_shm()
            self.with_streaming = False

    def _get_robot_status(self):
        return self.robot.states()

    # def _get_system_status(self):
    #     self.robot.getSystemStatus(self.system_status)
    #     return self.system_status

    def switch_mode(self, mode):
        mode = getattr(self.mode, mode)
        if mode != self.current_mode:
            self.robot.SwitchMode(mode)
            self.current_mode = mode

    def get_tcp_info(self):
        '''
        get current robot's tcp information.
        '''
        status = self._get_robot_status()
        info = status.tcp_pose + status.ext_wrench_in_tcp[:6]
        return np.array(info)
        
    def get_tcp_pose(self):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcp_pose)

    def get_tcp_vel(self):
        """get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcp_vel)

    def get_tcp_force(self):
        """get current robot's tool force torque wrench in TCP frame.

        Returns:
            6-dim list consisting of (fx,fy,fz,wx,wy,wz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().ext_wrench_in_tcp[:6])

    def get_tcp_force_base(self):
        """get current robot's tool force torque wrench in base frame.

        Returns:
            6-dim list consisting of (fx,fy,fz,wx,wy,wz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().ext_wrench_in_world[:6])

    def get_joint_pos(self):
        """get current joint value.

        Returns:
            7-dim numpy array of 7 joint position

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().q)

    def get_joint_vel(self):
        """get current joint velocity.

        Returns:
            7-dim numpy array of 7 joint velocity

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().dq)

    def get_joint_torque(self):
        """get measured link-side joint torque.

        Returns:
            7-dim numpy array of 7 link torques
        """
        return np.array(self._get_robot_status().tau)

    def get_joint_info(self):
        '''
        get current robot's joint information.
        '''
        status = self._get_robot_status()
        info = status.q + status.dq
        return np.array(info)
        
    def get_external_joint_torque(self):
        """get estimated EXTERNAL link-side joint torque.

        Returns:
            7-dim numpy array of 7 link torques
        """
        return np.array(self._get_robot_status().tau_ext)

    def stop(self):
        """Stop current motion and switch mode to idle."""
        self.robot.Stop()

    def set_max_contact_wrench(self, max_wrench):
        self.switch_mode("NRT_CARTESIAN_MOTION_FORCE")
        self.robot.SetMaxContactWrench(max_wrench)

    def send_impedance_online_pose(
        self, tcp, wrench=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    ):
        """make robot move towards target pose in impedance control mode,
        combining with sleep time makes robot move smmothly.

        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            wrench: 6-dim list or numpy array, max moving force (fx,fy,fz,wx,wy,wz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("NRT_CARTESIAN_MOTION_FORCE")
        pose = np.array(tcp)
        self.robot.SendCartesianMotionForce(pose, max_linear_vel = 0.5, max_linear_acc = 1.0, max_angular_vel = 0.5, max_angular_acc = 1.0)

    def send_tcp_pose(
        self, tcp, wrench=np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
    ):
        """make api align with 2.6
        """
        self.send_impedance_online_pose(tcp, wrench)

    # def cali_force_sensor(self, data_collection_time=0.2):
    #     self.switch_mode("NRT_PRIMITIVE_EXECUTION")
    #     self.robot.ExecutePrimitive("ZeroFTSensor")

    def send_joint_position(
        self,
        target_jnt_pos,
        target_jnt_vel=[0, 0, 0, 0, 0, 0, 0],
        target_jnt_acc=[0, 0, 0, 0, 0, 0, 0],
        max_jnt_vel=[2, 2, 2, 2, 2, 2, 2],
        max_jnt_acc=[3, 3, 3, 3, 3, 3, 3]
    ):
        """make robot move towards target joint position in position control
        online mode without reach check.

        Args:
            target_jnt_pos: 7-dim list or numpy array, target joint position of 7 joints
            target_jnt_vel: 7-dim list or numpy array, target joint velocity of 7 joints
            target_jnt_acc: 7-dim list or numpy array, target joint acceleration of 7 joints
            max_jnt_vel: 7-dim list or numpy array, maximum joint velocity of 7 joints
            max_jnt_acc: 7-dim list or numpy array, maximum joint acceleration of 7 joints
            max_jnt_jerk: 7-dim list or numpy array, maximum joint jerk of 7 joints

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("NRT_JOINT_POSITION")
        
        self.robot.SendJointPosition(
            target_jnt_pos,
            target_jnt_vel,
            target_jnt_acc,
            max_jnt_vel,
            max_jnt_acc
        )

        
    def _close_shm(self):
        '''
        Close shared memory objects.
        '''
        if self.with_streaming:
            self.shm_tcp.close()
            self.shm_tcp.unlink()
            self.shm_joint.close()
            self.shm_joint.unlink()
