"""
Library for the control of HANWHA PTZ cameras using Sunapi
"""
import logging
import sys
import argparse
import ast
import time
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from bs4 import BeautifulSoup

logging.basicConfig(filename='sunapi.log', filemode='w', level=logging.DEBUG)
logger = logging.getLogger("HANWHA_camera")
logger.info('Started')

class CameraControl:
    """
    Module for the control of HANWHA cameras using Sunapi
    """

    def __init__(self, ip, user, password):
        self.__cam_ip = ip
        self.__cam_user = user
        self.__cam_password = password

    def _camera_command(self, value_cgi, payload: dict):
        """
        Function used to send commands to the camera
        Args:
            payload: argument dictionary for camera control

        Returns:
            Returns the response from the device to the command sent

        """

        logger.info('camera_command(%s)', payload)

        url = 'http://' + self.__cam_ip + '/stw-cgi/' + value_cgi

        resp = requests.get(url, auth=HTTPDigestAuth(self.__cam_user, self.__cam_password),
                            params=payload)

        if (resp.status_code != 200) and (resp.status_code != 204):
            soup = BeautifulSoup(resp.text, features="lxml")
            logger.error('%s', soup.get_text())
            if resp.status_code == 401:
                sys.exit(1)

        return resp

    def operation_finished(self):
        """
        Operation to request PTZ status.

        Returns:
            Returns status and notifies when the operation is finished

        """
        resp = self._camera_command('ptzcontrol.cgi',
                                    {'msubmenu': 'query', 'action': 'view', 'Query': 'Pan,Tilt,Zoom'})

        current_pan = float(resp.text.split()[0].split('=')[1])
        current_tilt = float(resp.text.split()[1].split('=')[1])
        current_zoom = float(resp.text.split()[2].split('=')[1])
        current_zoom_pulse = float(resp.text.split()[3].split('=')[1])

        if abs(360 - current_pan) < 0.02 or current_pan < 0.02:
            # This if statement is necessary for when absolute pan is zero. When the camera position
            # is requested, the query returned was either approximately 360 or zero. This statement
            # sets out to fix that bug by forcing the current pan position to be read as zero.
            current_pan = 0

        ptz_list = (current_pan, current_tilt, current_zoom)

        return ptz_list, current_zoom_pulse

    def stop_control(self):
        """
        Operation to stop ongoing pan, tilt and zoom movements of absolute relative and
        continuous type

        Returns:
            Returns the response from the device to the command sent

        """

        resp = self._camera_command('ptzcontrol.cgi', {'msubmenu': 'stop', 'action': 'control',
                                                       'OperationType': 'All'})

        logger.info(resp.url + "\n" + str(resp.status_code) + "\n" + resp.text)

        return resp

    def absolute_control(self, pan: float = None, tilt: float = None, zoom: float = None, zoom_pulse: int = None,
                         channel: int = None):
        """
        Operation to move pan, tilt or zoom to an absolute destination.

        Args:
            pan: pans the device relative to the (0,0) position.
            tilt: tilts the device relative to the (0,0) position.
            zoom: zooms the device n steps relative to 1 zoom.

        Returns:
            Returns the response from the device to the command sent.

        """

        init_pos, initial_zoom_pulse = self.operation_finished()  # takes current (pan, tilt, zoom) values as an array
                                                                  # and initial_zoom_pulse

        resp = self._camera_command('ptzcontrol.cgi', {'msubmenu': 'absolute', 'action': 'control',
                                                       'Pan': pan, 'Tilt': tilt, 'Zoom': zoom,
                                                       'ZoomPulse': zoom_pulse, 'Channel': channel})

        logger.info(resp.url + "\n" + str(resp.status_code) + "\n" + resp.text)

        current_position = np.sum(init_pos)  # sums elements in the array

        current_zoom_pulse = initial_zoom_pulse  # set current_zoom_pulse to the initial_zoom_pulse position

        """
        If either pan, tilt, or zoom were not chosen, set their final position to be equal to
        their current position
        
        """

        error_margin = 0.4

        if pan is None:
            pan = init_pos[0]
            error_margin = error_margin-0.1

        if tilt is None:
            tilt = init_pos[1]
            error_margin = error_margin-0.1

        if zoom is None:
            zoom = init_pos[2]
            error_margin = error_margin-0.1
        """
        Since operation finished will set pan equal to zero if it is approximately close,
        pan must be set equal to zero here as well, since the finished_position pan will be zero
        
        """
        if abs(pan - 360) < 0.02:
            pan = 0

        finished_position = pan + tilt + zoom  # given values for the finished position

        i = int()
        
        start_time = time.time()

        if zoom_pulse is None:

            while abs(current_position - finished_position) > error_margin:

                final_position = current_position  # final_position = last position recorded

                init_pos, initial_zoom_pulse = self.operation_finished()  # update current position array and zoom_pulse

                current_position = np.sum(init_pos)  # update current position

                if final_position == current_position:  # if the final position and current position are the same
                    i = i + 1  # add one to counter

                if i == 5:  # if the current_position and final_position have been the same for five times in a row
                    logger.error('end of command error')
                    break  # break the loop

            time.sleep(1)

        else:  # if zoom pulse is True, pass in zoom_pulse as final zoom_pulse
            finished_zoom_pulse = zoom_pulse  # finished value given for zoom_pulse

            while abs(current_zoom_pulse - finished_zoom_pulse) > 5:

                final_zoom_pulse = current_zoom_pulse  # final_zoom_pulse = last zoom pulse recorded

                time.sleep(0.1)

                init_pos, initial_zoom_pulse = self.operation_finished()  # update current position array and zoom_pulse

                current_zoom_pulse = initial_zoom_pulse  # update current_zoom_pulse

                if final_zoom_pulse == current_zoom_pulse:  # if the final_zoom_pulse and current_zoom_pulse are the same
                    i = i + 1  # add one to counter

                if i == 5:  # if the current_zoom_pulse and final_zoom_pulse have been the same for five times in a row
                    logger.error('end of command error')
                    break  # break the loop

            time.sleep(0.5)

        logger.info('Finished')

        end_time = time.time()

        elapsed_time = end_time - start_time

        logger.info("elapsed_time: " + str(elapsed_time))


    def relative_control(self, pan: float = None, tilt: float = None, zoom: int = None, zoom_pulse: int = None,
                         channel: int = None):
        """
        Operation for Relative Pan/Tilt and Zoom Move.

        Args:
            pan: pans the device n degrees relative to the current position.
            tilt: tilts the device n degrees relative to the current position.
            zoom: zooms the device n steps relative to the current position.

        Returns:
            Returns the response from the device to the command sent.

        """

        init_pos, initial_zoom_pulse = self.operation_finished()  # takes current position values as an array

        current_position = np.sum(init_pos)  # sums elements in the initial position array

        current_pan = init_pos[0]  # provides the absolute pan position

        current_tilt = init_pos[1]  # provides the absolute tilt position

        current_zoom = init_pos[2]  # provides the absolute zoom position

        current_zoom_pulse = initial_zoom_pulse  # provides the absolute zoom_pulse position

        if pan is not None:

            # If the relative pan given causes the absolute pan position to surpass 360 degrees,
            # set pan to go the other direction to reach the same location

            if (current_pan + pan) > 360:
                pan = pan - 360

            # If the relative pan given causes the absolute pan position to fall below 0 degrees,
            # set pan to go the other direction to reach the same location

            elif (current_pan + pan) < 0:
                pan = 360 + pan

        if tilt is not None:

            # if the relative tilt given exceeds the 90 degree threshold, set the relative tilt
            # equal to the difference that will result in the maximum 90-degree tilt

            if 90 < (current_tilt + tilt):
                tilt = 90 - current_tilt

            # if the relative tilt given exceeds the -20 degree threshold, set the relative tilt
            # equal to the difference that will result in the minimum -20-degree tilt

            elif (current_tilt + tilt) < -20:
                tilt = -20 + abs(current_tilt)

        if zoom is not None:

            # if the relative zoom given exceeds the 40 zoom threshold, set the relative zoom
            # equal to the difference that will result in the maximum 40 zoom

            if 40 < (current_zoom + zoom):
                zoom = 40 - current_zoom

            # if the relative zoom given exceeds the 1 degree threshold, set the relative zoom
            # equal the difference that will result in the minimum -20-degree tilt

            elif (current_zoom + zoom) < 1:
                zoom = 1 - current_zoom

        """
        If current_pan is zero, then do absolute move from the zero position. Sometimes zero is read as 359.9... and
        thus relative move will not go beyond zero and will stall there.
        
        """
        if current_pan != 0:
            resp = self._camera_command('ptzcontrol.cgi', {'msubmenu': 'relative', 'action': 'control',
                                                           'Pan': pan, 'Tilt': tilt, 'Zoom': zoom,
                                                           'ZoomPulse': zoom_pulse, 'Channel': channel})

        elif current_pan == 0:
            resp = self._camera_command('ptzcontrol.cgi', {'msubmenu': 'absolute', 'action': 'control',
                                                           'Pan': pan, 'Tilt': tilt, 'Zoom': zoom,
                                                           'ZoomPulse': zoom_pulse, 'Channel': channel})

        logger.info(resp.url + "\n" + str(resp.status_code) + "\n" + resp.text)

        # If either pan, tilt, or zoom were not chosen, set their relative movement equal to
        # zero as nothing will be changed.

        if pan is None:
            pan = 0

        if tilt is None:
            tilt = 0

        if zoom is None:
            zoom = 0

        # The finished_position will be set to the current_position plus whatever relative changes
        # will be made.

        if current_pan + pan == 360:  # pan at 360 will be set to zero. Remove pan from finished_position
            finished_position = tilt + zoom + current_position - current_pan

        else:
            finished_position = tilt + zoom + current_position + pan

        i = int()
        start_time = time.time()

        if zoom_pulse is None:

            while abs(current_position - finished_position) > 0.5:

                final_position = current_position  # final_position = last position recorded

                init_pos, initial_zoom_pulse = self.operation_finished()  # update current position array and zoom_pulse

                current_position = np.sum(init_pos)  # update current position

                if final_position == current_position:  # if the final position and current position are the same
                    i = i + 1  # add one to counter

                if i == 5:  # if the current_position and final_position have been the same for five times in a row
                    break  # break the loop

        else:  # if zoom pulse is True, pass in zoom_pulse as final_zoom_pulse

            i = int()

            while i < 2:

                final_zoom_pulse = current_zoom_pulse  # final_zoom_pulse = last zoom pulse recorded

                time.sleep(0.2)

                init_pos, initial_zoom_pulse = self.operation_finished()  # update current position array and zoom_pulse

                current_zoom_pulse = initial_zoom_pulse  # update current_zoom_pulse

                if final_zoom_pulse == current_zoom_pulse:  # if the final_zoom_pulse and current_zoom_pulse are the same
                    i = i + 1  # add one to counter

        time.sleep(1)

        logger.info('Finished')

        end_time = time.time()

        elapsed_time = end_time - start_time

        logger.info("elapsed_time: " + str(elapsed_time))

    def continuous_control(self, normalized_speed: bool = None, pan: int = None, tilt: int = None, zoom: int = None,
                           focus: str = None):
        """
        Operation for continuous Pan/Tilt and Zoom movements.

        Args:
            normalized_speed: enables or disables the normalized speed range for pan, tilt, zoom.
            pan: speed of movement of Pan.
            tilt: speed of movement of Tilt.
            zoom: speed of movement of Zoom.
            focus: focus control. This parameter cannot be sent together with pan, tilt, or zoom.

        Returns:
            Returns the response from the device to the command sent.

        """

        if focus not in ("Near", "Far", "Stop", None):
            raise Exception("Unauthorized command: Please enter a string from the choices: 'Near', 'Far', or 'Stop'")

        return self._camera_command('ptzcontrol.cgi', {'msubmenu': 'continuous', 'action': 'control',
                                                       'NormalizedSpeed': normalized_speed, 'Pan': pan,
                                                       'Tilt': tilt, 'Zoom': zoom, 'Focus': focus})

    def area_zoom(self, x1: int = None, x2: int = None, y1: int = None,
                  y2: int = None, tilewidth: int = None, tileheight: int = None):
        """
        Centers on positions x,y (like the center command) and zooms by a factor of z/100.

        Args:
            x1: value of the x1 coordinate.
            x2: value of the x2 coordinate.
            y1: value of the y1 coordinate.
            y2: value of the y2 coordinate.
            tilewidth: sets tile pixel width
            tileheight: sets tile pixel height

        Returns:
            Returns the response from the device to the command sent

        """

        # current_zoom = self.operation_finished()[2]

        resp = self._camera_command('ptzcontrol.cgi', {'msubmenu': 'areazoom', 'action': 'control',
                                                'X1': x1, 'X2': x2, 'Y1': y1, 'Y2': y2, 'TileWidth': tilewidth,
                                                'TileHeight': tileheight})

        logger.info(resp.url + "\n" + str(resp.status_code) + "\n" + resp.text)

        if tilewidth is None:
            tilewidth = 10000

        if tileheight is None:
            tileheight = 10000

        init_pos, initial_zoom_pulse = self.operation_finished()  # takes current (pan, tilt, zoom) values as an array
                                                                  # and initial_zoom_pulse

        current_zoom_pulse = initial_zoom_pulse  # provides the absolute zoom_pulse position
        """
        Checks to see if area zoom is finished

        """
        i = int()

        start_time = time.time()

        while i < 2:

            final_zoom_pulse = current_zoom_pulse  # final_zoom_pulse = last zoom pulse recorded

            time.sleep(0.2)

            init_pos, initial_zoom_pulse = self.operation_finished()  # update current position array and zoom_pulse

            current_zoom_pulse = initial_zoom_pulse  # update current_zoom_pulse

            if final_zoom_pulse == current_zoom_pulse:  # if the final_zoom_pulse and current_zoom_pulse are the same
                i = i + 1  # add one to counter

        logger.info('Finished')

        end_time = time.time()

        elapsed_time = end_time - start_time

        logger.info("elapsed_time: " + str(elapsed_time))

    def movement_control(self, direction: str = None, movespeed: float = None):
        """
        Moves the device continuously in the specified direction.

        Args:
            direction: direction to move. (home, up, down, left, right, upleft, upright, downleft...)
            movespeed: speed to move camera.

        Returns:
            Returns the response from the device to the command sent

        """

        return self._camera_command('ptzcontrol.cgi', {'msubmenu': 'move', 'action': 'control',
                                                       'Direction': direction, 'MoveSpeed': movespeed})

    def moving_to_home_position(self, channel: int = None):
        """
        Operation to move the PTZ device to its "home" position.

        Args:
            channel: returns to home position for provided channel

        Returns:
            Returns the response from the device to the command sent

        """
        return self._camera_command('ptzcontrol.cgi', {'msubmenu': 'home', 'action': 'control',
                                                       'Channel': channel})

    def requesting_cameras_position_information(self, show: str = None):
        """
        Operation to request PTZ status.

        Returns:
            Returns a tuple with the position of the camera (P, T, Z)

        """
        resp = self._camera_command('ptzcontrol.cgi',
                                    {'msubmenu': 'query', 'action': 'view', 'Query': 'Pan,Tilt,Zoom'})
        pan = float(resp.text.split()[0].split('=')[1])
        tilt = float(resp.text.split()[1].split('=')[1])
        zoom = float(resp.text.split()[2].split('=')[1])
        ptz_list = (pan, tilt, zoom)

        if show:
            logger.info(resp.text)
            logger.info(ptz_list)

        return ptz_list

    def moving_to_preset_position(self, preset: int = None, presetname: str = None):
        """
        Move to the position associated with the preset on server.

        Args:
            preset: numbered position of preset
            presetname: name of preset position server.
            * cannot be sent together

        Returns:
            Returns the response from the device to the command sent

        """
        return self._camera_command('ptzcontrol.cgi', {'msubmenu': 'preset', 'action': 'control',
                                                       'Preset': preset, 'PresetName': presetname})

    def zoom_out(self):
        """
        Zoom Out to 1x

        Returns:
            Returns the response from the device to the command sent

        """
        init_pos, initial_zoom_pulse = self.operation_finished()  # takes current position values

        current_zoom_pulse = initial_zoom_pulse  # provides the absolute zoom_pulse position

        resp = self._camera_command('ptzcontrol.cgi', {'msubmenu': 'areazoom', 'action': 'control',
                                                       'Type': '1x'})

        logger.info(resp.url + "\n" + str(resp.status_code) + "\n" + resp.text)

        start_time = time.time()

        while current_zoom_pulse != 0:

            init_pos, initial_zoom_pulse = self.operation_finished()  # update current position array and zoom_pulse

            current_zoom_pulse = initial_zoom_pulse  # update current_zoom_pulse

        time.sleep(0.5)

        logger.info('Finished')

        end_time = time.time()

        elapsed_time = end_time - start_time

        logger.info("elapsed_time: " + str(elapsed_time))

    def aux_control(self, command: str = None):
        """
           Execute aux action

           Args:
               command = choice between WiperOn, HeaterOn, HeaterOff

           Returns:
               Returns the response from the device to the command sent

           """

        return self._camera_command('ptzcontrol.cgi', {'msubmenu': 'aux', 'action': 'control',
                                                       'Command': command})

    def attributes_information(self):
        """
           create url link to attributes information

           Returns:
               Returns the response from the device to the command sent

           """

        attributes = self._camera_command('attributes.cgi', {})

        logger.info(attributes.url)

    def swing_control(self, channel: int = None, mode: str = None):
        """
              Move from one preset to another

              Args:
                  channel = choose channel
                  mode = select mode of either: "Pan", "Tilt", "PanTilt", "Stop"

              Returns:
                  Returns the response from the device to the command sent

              """

        if mode not in ("Pan", "Tilt", "PanTilt", "Stop", None):
            raise Exception(
                "Unauthorized command: Please enter a string from the choices: 'Pan', 'Tilt', 'PanTilt', 'Stop'")

        return self._camera_command('ptzcontrol.cgi', {'msubmenu': 'swing', 'action': 'control',
                                                       'Channel': channel, 'Mode': mode})

    def group_control(self, channel: int = None, group: int = None, mode: str = None):
        """
              Starts and stops a Group operation in which various presets are grouped and called
              in sequence.

              Args:
                  channel = choose channel
                  group = select a group sequence set in channel
                  mode = choose a mode of either "Start" or "Stop"

              Returns:
                  Returns the response from the device to the command sent

              """

        if mode not in ("Start", "Stop", None):
            raise Exception("Unauthorized command: Please enter a string from the choices: 'Start' or 'Stop'")

        return self._camera_command('ptzcontrol.cgi', {'msubmenu': 'group', 'action': 'control',
                                                       'Channel': channel, 'Group': group,
                                                       'Mode': mode})

    def tour_control(self, channel: int = None, tour: int = None, mode: str = None):
        """
              Starts and stops a Tour operation, calling groups of presets in sequence.

              Args:
                  channel = choose channel
                  tour = select a tour sequence set in channel
                  mode = choose a mode of either "Start" or "Stop"

              Returns:
                  Returns the response from the device to the command sent

              """

        if mode not in ("Start", "Stop", None):
            raise Exception("Unauthorized command: Please enter a string from the choices: 'Start' or 'Stop'")

        return self._camera_command('ptzcontrol.cgi', {'msubmenu': 'tour', 'action': 'control',
                                                       'Channel': channel, 'Tour': tour,
                                                       'Mode': mode})

    def trace_control(self, channel: int = None, trace: int = None, mode: str = None):
        """
              Starts and stops a Trace operation

              Args:
                  channel = choose channel
                  trace = select a trace action that has been set in channel
                  mode = choose a mode of either "Start" or "Stop"

              Returns:
                  Returns the response from the device to the command sent

              """

        if mode not in ("Start", "Stop", None):
            raise Exception("Unauthorized command: Please enter a string from the choices: 'Start' or 'Stop'")

        return self._camera_command('ptzcontrol.cgi', {'msubmenu': 'trace', 'action': 'control',
                                                       'Channel': channel, 'Trace': trace,
                                                       'Mode': mode})

    def applications(self):
        """
              Creates url and shares installed applications information
              Returns:
                  Returns the response from the device to the command sent

              """

        resp = self._camera_command('opensdk.cgi', {'msubmenu': 'apps', 'action': 'view'})

        logger.info(resp.url + "\n" + str(resp.status_code) + "\n" + resp.text)

        return resp

    def snap_shot(self, directory: str = None):
        """
            Sends camera command snapshot
            Returns:
                 Returns the response from the device to the command sent

        """

        resp = self._camera_command('video.cgi', {'msubmenu': 'snapshot', 'action': 'view'})

        logger.info(resp.url + "\n" + str(resp.status_code) + "\n")

        if directory is None:  # if no directory is passed
            a1 = __file__  # copy source 'sunapi_control.py' directory
            string = a1.replace('/source', '/snapshots')  # save image to file 'snapshots' instead of source
            idx = string.rfind('/')  # remove name of .py file from end of string
            if (idx != -1):  # only do this if '/' is found in directory
                string = string[:idx + 1]  # remove characters until '/' is reached
            # concatenate snapshots' directory with (date&time) and '.jpg' as this is where the image will be saved on the local machine
            directory = string + time.strftime('%b_%d_%Y_%H_%M_%S_') + '.jpg'
            logger.info(directory)
        # Save image in a set directory
        if resp.status_code == 200:
            with open(directory.replace(' ', '_'), 'wb') as f:
                f.write(resp.content)


