"""
Library for configuring HANWHA PTZ cameras using Sunapi
"""
import logging
import sys
import argparse
import ast
import time
import numpy as np
import math
import requests
from requests.auth import HTTPDigestAuth
from bs4 import BeautifulSoup


class CameraConfiguration:
    """
    Module for configuration of HANWHA cameras using Sunapi
    """

    def __init__(self, ip, user, password):
        self.__cam_ip = ip
        self.__cam_user = user
        self.__cam_password = password

    def _camera_command(self, value_cgi, payload: dict):
        """
        Function used to send commands to the camera
        Args:
            payload: argument dictionary for camera configuration

        Returns:
            Returns the response from the device to the command sent

        """

        logging.info('camera_command(%s)', payload)

        url = 'http://' + self.__cam_ip + '/stw-cgi/' + value_cgi

        resp = requests.get(url, auth=HTTPDigestAuth(self.__cam_user, self.__cam_password),
                            params=payload)

        # resp = requests.post(url, auth=HTTPDigestAuth(self.__cam_user, self.__cam_password),
        #                     params=payload)

        print(payload)
        print(resp.url)
        print(resp)
        print(resp.text)

        if resp.status_code == 200:
            return resp.text

        text = str(resp)
        text += str(resp.text)
        return text

    def swing_setup(self, action: str = None, channel: int = None, mode: str = None,
                    from_preset: int = None, to_preset: int = None, speed: int = None,
                    dwell_time: int = None):
        """
              Configures the Swing settings

              Args:
                  action = select an action of either "view" or "set"
                  channel = choose channel
                  mode = select mode of either: "Pan", "Tilt", "PanTilt"

              Returns:
                  Returns the response from the device to the command sent

              """

        if action not in ("view", "set", None):
            raise Exception(
                "Unauthorized command: Please enter a string from the choices: 'view' or 'set'")

        if mode not in ("Pan", "Tilt", "PanTilt", None):
            raise Exception(
                "Unauthorized command: Please enter a string from the choices: 'Pan', 'Tilt', 'PanTilt'")

        resp = self._camera_command('ptzconfig.cgi', {'msubmenu': 'swing', 'action': action,
                                                      'Channel': channel, 'Mode': mode,
                                                      'FromPreset': from_preset, 'ToPreset': to_preset,
                                                      'Speed': speed, 'DwellTime': dwell_time})
        if action == 'view':
            print(resp)

    def group_setup(self, action: str = None, channel: int = None, group: int = None,
                    preset_sequence: int = None, preset: int = None, speed: int = None,
                    dwell_time: int = None):
        """
              Configures the Group settings. Multiple presets are grouped and called in sequence according
              to the Group feature.

              Args:
                  action = select an action of either: "view", "add", "update", "remove"
                  channel = choose channel
                  group = select group number
                  preset_sequence = select preset sequence
                  preset = select preset

                  * The number of presets supported in total is 300. The maximum
                  number of presets per group is 128.

                  speed = select speed of movement from one preset to the next
                  dwell_time = determine the length of time spent at a given preset

              Returns:
                  Returns the response from the device to the command sent

              """

        if action not in ("set", "view", "add", "update", "remove", None):
            raise Exception(
                "Unauthorized command: Please enter a string from the choices: 'view', 'add', 'update', 'remove'")

        resp = self._camera_command('ptzconfig.cgi', {'msubmenu': 'group', 'action': action,
                                                      'Channel': channel, 'Group': group,
                                                      'PresetSequence': preset_sequence, 'Preset': preset,
                                                      'Speed': speed, 'DwellTime': dwell_time})
        if action == 'view':
            print(resp)


def main():
    class CustomAction(argparse.Action):
        '''
        In this CustomAction class, an array will be made with action destinations being placed in 0 up through
        '+' EVEN# positions and their corresponding values paired in 1 up through '+' ODD# positions.
        '''

        def __call__(self, parser, namespace, values, option_string=None):
            if not 'ordered_args' in namespace:  # if 'ordered_args' is not found in namespace
                setattr(namespace, 'ordered_args', [])  # Add 'ordered_args' to namespace
            previous = namespace.ordered_args  # saves each previous entry to keep track of terminal commands
            previous.append(self.dest)  # appends action_destination to 'ordered_args' in namespace
            previous.append(str(values))  # appends action_value to 'ordered_args' in namespace
            setattr(namespace, 'ordered_args', previous)  # appends action's value to 'ordered_args' in namespace

    parser = argparse.ArgumentParser(description='Process some actions.')

    parser.add_argument('-ip', '--ipAddress', action='store')
    parser.add_argument('-un', '--username', action='store')
    parser.add_argument('-pw', '--password', action='store')

    parser.add_argument('-sc', '--swing_config', help='to execute a swing action',
                        nargs='+', action=CustomAction)

    parser.add_argument('-gc', '--group_config', help='to execute a group action',
                        nargs='+', action=CustomAction)

    parser.add_argument('-tc', '--tour_config', help='to execute a tour action',
                        nargs='+', action=CustomAction)

    parser.add_argument('-trace', '--trace_control', help='to execute a trace action',
                        nargs='+', action=CustomAction)

    args = parser.parse_args()

    dictionary = vars(args)

    ordered_args = dictionary.get('ordered_args')

    terminal_control = CameraConfiguration(args.ipAddress, args.username, args.password)  # calling the Class CameraConfiguration

    if ordered_args:

        k = 1  # offsets i by 1, to provide the corresponding value[k] to destination[i]

        for i in ordered_args[::2]:  # starts from first position and iterates by two to get next destination

            if i == 'swing_config':
                swing_param = ast.literal_eval(ordered_args[k])

                channel, mode = swing_param[:2]

                terminal_control.swing_config(channel=channel, mode=mode)

            if i == 'group_control':
                group_param = ast.literal_eval(ordered_args[k])

                channel, group, mode = group_param[:3]

                terminal_control.group_control(channel=channel, group=group, mode=mode)

            if i == 'tour_control':
                tour_param = ast.literal_eval(ordered_args[k])

                channel, tour, mode = tour_param[:3]

                terminal_control.tour_control(channel=channel, tour=tour, mode=mode)

            if i == 'trace_control':
                trace_param = ast.literal_eval(ordered_args[k])

                channel, trace, mode = trace_param[:3]

                terminal_control.trace_control(channel=channel, trace=trace, mode=mode)

            k = k + 2


if __name__ == '__main__':
    main()
