"""Flexiv robot ethernet communication python wrapper api.
Author: Junfeng Ding, Hongjie Fang
"""

import time
import numpy as np
import threading
from multiprocessing import shared_memory

##########---------- !!!IMPORTANT!!! ----------##########
########## first put `lib_cxx` under the same directory ##########
try:
    import lib_cxx
except:
    print("First put `lib_cxx` under `utils/flexiv_sdk`")
    exit(-1)

LOG_LEVEL = "info"


class ModeMap:
    idle = "CTRL_MODE_IDLE"
    plan = "CTRL_MODE_PLAN_EXECUTION"
    primitive = "CTRL_MODE_PRIMITIVE_EXECUTION"
    torque = "CTRL_MODE_CARTESIAN_POSE"
    online = "CTRL_MODE_ONLINE_MOVE"
    line = "CTRL_MODE_MOVE_LINE"
    joint = "CTRL_MODE_JOINT_PVAT_DOWNLOAD"


class FlexivApi:
    """Flexiv Robot Control Class.
    This class provides python wrapper for Flexiv robot control in different modes.
    Features include:
        - torque control mode
        - online move mode
        - plan execution mode
        - get robot status information
        - force torque record
    """

    logger_name = "FlexivApi"

    def __init__(self, robot_ip_address, pc_ip_address, with_streaming = False):
        """initialize.
        Args:
            robot_ip_address: robot_ip address string
            pc_ip_address: pc_ip address string
            with_streaming:  whether with streaming option.
        Returns:
            None
        Raises:
            RuntimeError: error occurred when ip_address is None.
        """
        self.robot = lib_cxx.FlexivRdkWrapper.RobotClientHandler()
        ret = self.robot.init(robot_ip_address, pc_ip_address, 7)
        if ret != lib_cxx.FvrParamsWrapper.FvrStg.FVR_OKg:
            print("[Robot] Init failed.")
            print(ret)
        else:
            time.sleep(0.6)
            self.extCtrlMode = lib_cxx.FlexivRdkWrapper.ControlMode
            print("[Robot] Init success")
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
        robot_status_data = lib_cxx.FlexivRdkWrapper.RobotStates()
        self.robot.readRobotStatus(robot_status_data)
        return robot_status_data

    def _get_system_status(self):
        system_status_data = lib_cxx.FlexivRdkWrapper.SystemStatus()
        self.robot.readSystemStatus(system_status_data)
        return system_status_data

    def mode_mapper(self, mode):
        assert mode in ModeMap.__dict__.keys(), "unknown mode name: %s" % mode
        return getattr(self.extCtrlMode, getattr(ModeMap, mode))

    def get_control_mode(self):
        return self.robot.getCtrlMode()

    def set_control_mode(self, mode):
        control_mode = self.mode_mapper(mode)
        self.robot.setControlMode(control_mode)

    def switch_mode(self, mode, sleep_time=0.01):
        """switch to different control modes.
        Args:
            mode: 'torque', 'online', 'plan', 'idle', 'line', 'joint', 'primitive'
            sleep_time: sleep time to control mode switch time
        Returns:
            None
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        if self.get_control_mode() == self.mode_mapper(mode):
            return

        while self.get_control_mode() != self.mode_mapper("idle"):
            self.set_control_mode("idle")
            time.sleep(sleep_time)

        while self.get_control_mode() != self.mode_mapper(mode):
            self.set_control_mode(mode)
            time.sleep(sleep_time)

        print("set mode: {}".format(str(self.get_control_mode())))

    def get_emergency_state(self):
        """get robot is emergency stopped or not. The emergency state means the
        E-Stop is pressed. instead of being soft fault.
        Returns:
            True indicates robot is not stopped, False indicates robot is emergency stopped.
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return not self._get_system_status().m_emergencyStop

    def clear_fault(self):
        self.robot.clearFault()

    def is_fault(self):
        """Check if robot is in FAULT state."""
        return self.robot.isFault()

    def is_connected(self):
        """return if connected.
        Returns: True/False
        """
        return self.robot.connected()

    def target_reached(self):
        """Only work when using move related api in EthernetConnection."""
        return self.robot.targetReached()

    def get_tcp_pose(self):
        """get current robot's tool pose in world frame.
        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().m_tcpPose)

    def get_tcp_force(self):
        """get current robot's tool force torque wrench.
        Returns:
            6-dim list consisting of (fx,fy,fz,wx,wy,wz)
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().m_tcpWrench[:6])

    def get_tcp_info(self):
        '''
        get current robot's tcp information.
        '''
        status = self._get_robot_status()
        info = status.m_tcpPose + status.m_tcpWrench[:6]
        return np.array(info)

    def get_camera_pose(self, camera_id: int = None):
        """get current wrist camera pose in world frame.
        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        # TODO: (chongzhao) check ethernet connection api for camera pose
        if camera_id is None:
            return np.array(self._get_robot_status().m_camPose)
        else:
            return np.array(self._get_robot_status().m_camPose[camera_id])

    def get_joint_pos(self):
        """get current joint value.
        Returns:
            7-dim numpy array of 7 joint position
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().m_jntPos)

    def get_joint_vel(self):
        """get current joint velocity.
        Returns:
            7-dim numpy array of 7 joint velocity
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().m_jntVel)
    
    def get_joint_info(self):
        '''
        get current robot's joint information.
        '''
        status = self._get_robot_status()
        info = status.m_jntPos + status.m_jntVel
        return np.array(info)

    def get_plan_info(self, attribute="m_ptName"):
        """get current robot's running plan info.
        Returns:
            name string of running node in plan
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        plan_info = lib_cxx.FlexivRdkWrapper.PlanInfo()
        self.robot.getPlanInfo(plan_info)
        return plan_info

    def get_plan_name_list(self):
        plan_list = []
        self.robot.getPlanNameList(plan_list)
        return plan_list


    def write_io(self, port, value):
        """Set io value.

        Args:
            port: 0~15
            value: True/False
        """
        assert port >= 0 and port <= 15
        self.robot.writeDigitalOutput(port, value)

    def execute_primitive(self, cmd):
        """Execute primitive.

        Args:
            cmd: primitive command string, e.x. "ptName(inputParam1=xxx, inputParam2=xxx, ...)"
        """
        self.switch_mode("primitive")
        self.logger.info("Execute primitive: {}".format(cmd))
        lib_cxx.FvrUtilsWrapper.checkError(self.robot.executePrimitive(cmd))

    def stop(self):
        """Stop current motion and switch mode to idle."""
        self.robot.stop()
        while self.get_control_mode() != self.mode_mapper("idle"):
            time.sleep(0.005)

    def send_tcp_pose(self, tcp, wrench=np.array([25.0, 25.0, 10.0, 20.0, 20.0, 20.0])):
        """make robot move towards target pose in torque control mode,
        combining with sleep time makes robot move smmothly.

        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            wrench: 6-dim list or numpy array, max moving force (fx,fy,fz,wx,wy,wz)

        Returns:
            None

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("torque")
        pose = np.array(tcp)
        time_count = 0.0
        index = 1
        self.robot.sendTcpPose(pose, wrench, time_count, index)

    def send_online_pose(self, tcp, max_v=0.05, max_a=0.1, max_w=0.2, max_dw=1):
        """make robot move towards target pose in online mode without reach
        check.

        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            max_v: double, max linear velocity
            max_a: double, max linear acceleration
            max_w: double, max angular velocity
            max_dw: double, max angular acceleration

        Returns:
            None

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("online")
        tcp = np.array(tcp)
        self.robot.sendOnlinePose(
            tcp, [0, 0, 0, 0, 0, 0], max_v, max_a, max_w, max_dw, 1
        )

    def move_online(
        self,
        tcp,
        max_v=0.05,
        max_a=0.2,
        max_w=1.5,
        max_dw=2.0,
        trans_epsilon=0.001,
        rot_epsilon=0.5,
    ):
        """move in online mode until reaching the target.
        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            max_v: double, max linear velocity
            max_a: double, max linear acceleration
            max_w: double, max angular velocity
            max_dw: double, max angular acceleration
            trans_epsilon: unit: meter, translation threshold to judge whether reach the target x,y,z
            rot_epsilon: unit: degree, rotation threshold to judge whether reach the target rotation degree
        Returns:
            None
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("online")
        while not self.target_reached():
            self.robot.sendOnlinePose(
                tcp, [0, 0, 0, 0, 0, 0], max_v, max_a, max_w, max_dw, 1
            )
            if self.is_fault():
                self.clear_fault()
                time.sleep(0.5)
                if self.is_fault():
                    print("FAULT in moveOnline")
                    return
            time.sleep(0.1)
        print("Reached")

    def move_line(self, waypoints, maxV, maxA, maxW, maxdW, level):
        """follow given trajectory composed of waypoints in online mode until
        finishing.
        Args:
            waypoints: n*7 numpy array composed of n waypoints, which is target pose (x,y,z,rw,rx,ry,rz) in world frame
            max_v: n-dim list of double, max linear velocity
            max_a: n-dim list of double, max linear acceleration
            max_w: n-dim list of double, max angular velocity
            max_dw: n-dim list of double, max angular acceleration
            level: n-dim list of int, control level
        Returns:
            None
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("line")
        self.robot.sendMoveLineWaypoints(waypoints, maxV, maxA, maxW, maxdW, level)
        while not self.target_reached():
            if self.is_fault():
                self.clear_fault()
                time.sleep(0.1)
                if self.is_fault():
                    print("FAULT in moveLine")
                    return
            time.sleep(0.01)
        print("Reached")

    def execute_plan_by_name(
        self,
        name,
    ):
        """execute plan by name, make sure control mode is switched into plan.
        Args:
            name: string of plan name
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("plan")
        lib_cxx.FvrUtilsWrapper.checkError(self.robot.executePlanByName(name))

    def execute_plan_by_index(self, index):
        """execute plan by index, make sure control mode is switched into plan.
        Args:
            index: int, index of plan
        Returns:
            None
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("plan")
        lib_cxx.FvrUtilsWrapper.checkError(self.robot.executePlanByIndex(index))

    def stop_plan_execution(self):
        self.robot.stopPlanExecution()
    
    def _close_shm(self):
        '''
        Close shared memory objects.
        '''
        if self.with_streaming:
            self.shm_tcp.close()
            self.shm_tcp.unlink()
            self.shm_joint.close()
            self.shm_joint.unlink()