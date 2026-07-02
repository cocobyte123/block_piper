#!/usr/bin/env python3
"""Send Piper to the project zero pose."""

import time

from piper_sdk import C_PiperInterface_V2


def main() -> None:
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()

    while not piper.EnablePiper():
        time.sleep(0.01)

    factor = 57295.7795  # 1000 * 180 / pi
    position = [0, 0, 5, 0, 0, 0, 0]
    joints = [round(value * factor) for value in position]

    piper.ModeCtrl(0x01, 0x01, 30, 0x00)
    print(*joints)
    piper.JointCtrl(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])
    piper.GripperCtrl(abs(joints[6]), 1000, 0x01, 0)


if __name__ == "__main__":
    main()
