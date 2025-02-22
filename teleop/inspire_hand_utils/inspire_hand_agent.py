
import serial
import numpy as np
import time
from .inspire_hand_serial_control import read6, write6, openSerial

class InspireHandAgent:
    def __init__(self,
                 port,
                 hand_id,
                 baudrate,
                 speed=1000):
        super().__init__()
        self.serial = openSerial(port, baudrate)
        self.id = hand_id

        # limit angle of hand joint (degree)
        self.finger_limit = [19, 176.7]
        self.thumb_band_limit = [-13, 53.6]
        self.thumb_rot_limit = [90, 165]

        if speed < 0 or speed > 1000:
            raise ValueError("Joint speed out of range!")
        self.joint_speed = speed


    def convert_degree_to_servo(self, degree_position: np.ndarray):
        servo_angle = list()
        for i, joint in enumerate(degree_position):
            if i <= 3:
                servo_angle.append(
                    int((degree_position[i] - self.finger_limit[0]) /
                        (self.finger_limit[1] - self.finger_limit[0]) * 1000)
                )
            elif i == 4:
                servo_angle.append(
                    int((degree_position[i] - self.thumb_band_limit[0]) /
                        (self.thumb_band_limit[1] - self.thumb_band_limit[0]) * 1000)
                )
            elif i == 5:
                servo_angle.append(
                    int((degree_position[i] - self.thumb_rot_limit[0]) /
                        (self.thumb_rot_limit[1] - self.thumb_rot_limit[0]) * 1000)
                )
            else:
                raise ValueError("position list should no longer than 6!")

        return servo_angle

    def convert_servo_to_degree(self, servo_position: np.ndarray):
        degree_angle = list()
        for i, joint in enumerate(servo_position):
            if i <= 3:
                degree_angle.append(
                    ((servo_position[i] - 0) *
                     (self.finger_limit[1] - self.finger_limit[0]) / 1000) + self.finger_limit[0]
                )
            elif i == 4:
                degree_angle.append(
                    ((servo_position[i] - 0) *
                     (self.thumb_band_limit[1] - self.thumb_band_limit[0]) / 1000) + self.thumb_band_limit[0]
                )
            elif i == 5:
                degree_angle.append(
                    ((servo_position[i] - 0) *
                     (self.thumb_rot_limit[1] - self.thumb_rot_limit[0]) / 1000) + self.thumb_rot_limit[0]
                )
            else:
                raise ValueError("position list should no longer than 6!")

        return degree_angle

    def set_joint_position(self, radian_position: np.ndarray):
        assert len(radian_position) == 6

        degree_position = np.degrees(radian_position)
        # print(f"Setting degree position is {degree_position}")
        servo_position = self.convert_degree_to_servo(degree_position)
        # print(f"Setting servo position is {servo_position}")

        # if any(angle < -1 or angle > 1000 for angle in servo_position):
        #     raise ValueError("Joint angle of inspire hand out of range!")
        servo_position = np.clip(servo_position, 0, 1000)

        write6(self.serial, self.id, 'angleSet', servo_position)

    def set_joint_speed(self):
        write6(self.serial, self.id, 'speedSet', np.full(6, self.joint_speed))

    def read_joint_position(self):
        servo_angle = read6(self.serial, self.id, 'angleAct')
        # print(f"Reading servo position is {servo_angle}")

        degree_angle = self.convert_servo_to_degree(np.array(servo_angle, dtype=np.float64))
        # print(f"Reading degree position is {degree_angle}")

        return np.deg2rad(degree_angle)

