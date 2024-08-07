"""
Library for control AXIS PTZ cameras using Vapix
"""
import time
import logging
import sys
import os
import requests
import cv2
from requests.auth import HTTPDigestAuth
from bs4 import BeautifulSoup

logging.basicConfig(filename='vapix.log',
                    filemode='w',
                    level=logging.DEBUG)
logger = logging.getLogger("AXIS_camera")
logger.info('Started')

# pylint: disable=R0904

# timeout (seconds)
TIME_TOLERANCE = 10

# focus threshold
FOCUS_THRESHOLD = 1000.0

class CameraControl:
    """
    Module for control cameras AXIS using Vapix
    """

    def __init__(self, ip, user, password, pan_margin=0.1, tilt_margin=0.1, zoom_margin=1):
        self.__cam_ip = ip
        self.__cam_user = user
        self.__cam_password = password

        self.__pan_margin = pan_margin
        self.__tilt_margin = tilt_margin
        self.__zoom_margin = zoom_margin

    @staticmethod
    def __merge_dicts(*dict_args) -> dict:
        """
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts

        Args:
            *dict_args: argument dictionary

        Returns:
            Return a merged dictionary
        """
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def _camera_command(self, payload: dict):
        """
        Function used to send commands to the camera
        Args:
            payload: argument dictionary for camera control

        Returns:
            Returns the response from the device to the command sent

        """
        logger.info('camera_command(%s)', payload)

        base_q_args = {
            'camera': 1,
            'html': 'no',
            'timestamp': int(time.time())
        }

        payload2 = CameraControl.__merge_dicts(payload, base_q_args)

        url = 'http://' + self.__cam_ip + '/axis-cgi/com/ptz.cgi'

        resp = requests.get(url, auth=HTTPDigestAuth(self.__cam_user, self.__cam_password),
                            params=payload2)

        if (resp.status_code != 200) and (resp.status_code != 204):
            soup = BeautifulSoup(resp.text, features="lxml")
            logger.error('%s', soup.get_text())
            if resp.status_code == 401:
                sys.exit(1)

        return resp

    def absolute_move(self, pan: float = None, tilt: float = None, zoom: int = None,
                      speed: int = None):
        """
        Operation to move pan, tilt or zoom to a absolute destination.

        Args:
            pan: pans the device relative to the (0,0) position.
            tilt: tilts the device relative to the (0,0) position.
            zoom: zooms the device n steps.
            speed: speed move camera.

        Returns:
            Returns the response from the device to the command sent.

        """
        resp = None
        start_time = time.time()
        resp = self._camera_command({'pan': pan, 'tilt': tilt, 'zoom': zoom, 'speed': speed})

        current_pan, current_tilt, current_zoom = self.get_ptz()
        while abs(current_pan - pan) > self.__pan_margin and abs(current_tilt - tilt) > self.__tilt_margin and abs(current_zoom - zoom) > self.__zoom_margin:
            current_pan, current_tilt, current_zoom = self.get_ptz()
            time.sleep(1)
            if time.time() - start_time > TIME_TOLERANCE:
                        break


        logger.info('Finished')

        end_time = time.time()

        elapsed_time = end_time - start_time

        logger.info("elapsed_time: %s", elapsed_time)

        return resp

    def continuous_move(self, pan: int = None, tilt: int = None, zoom: int = None):
        """
        Operation for continuous Pan/Tilt and Zoom movements.

        Args:
            pan: speed of movement of Pan.
            tilt: speed of movement of Tilt.
            zoom: speed of movement of Zoom.

        Returns:
            Returns the response from the device to the command sent.

        """
        pan_tilt = str(pan) + "," + str(tilt)
        return self._camera_command({'continuouspantiltmove': pan_tilt, 'continuouszoommove': zoom})

    def relative_move(self, rpan: float = None, rtilt: float = None, rzoom: int = None,
                      speed: int = None):
        """
        Operation for Relative Pan/Tilt and Zoom Move.

        Args:
            pan: pans the device n degrees relative to the current position.
            tilt: tilts the device n degrees relative to the current position.
            zoom: zooms the device n steps relative to the current position.
            speed: speed move camera.

        Returns:
            Returns the response from the device to the command sent.

        """
        resp = None
        start_time = time.time()

        current_pan, current_tilt, current_zoom = self.get_ptz()
        pan = current_pan + rpan
        tilt = current_tilt + rtilt
        zoom = current_zoom + rzoom
        resp = self._camera_command({'rpan': rpan, 'rtilt': rtilt, 'rzoom': rzoom, 'speed': speed})
        while abs(current_pan - pan) > self.__pan_margin and abs(current_tilt - tilt) > self.__tilt_margin and abs(current_zoom - zoom) > self.__zoom_margin:
            current_pan, current_tilt, current_zoom = self.get_ptz()
            time.sleep(1)
            if time.time() - start_time > TIME_TOLERANCE:
                        break


        logger.info('Finished')

        end_time = time.time()

        elapsed_time = end_time - start_time

        logger.info("elapsed_time: " + str(elapsed_time))

        return resp



    def stop_move(self):
        """
        Operation to stop ongoing pan, tilt and zoom movements of absolute relative and
        continuous type

        Returns:
            Returns the response from the device to the command sent

        """
        return self._camera_command({'continuouspantiltmove': '0,0', 'continuouszoommove': 0})

    def center_move(self, pos_x: int = None, pos_y: int = None, speed: int = None):
        """
        Used to send the coordinates for the point in the image where the user clicked. This
        information is then used by the server to calculate the pan/tilt move required to
        (approximately) center the clicked point.

        Args:
            pos_x: value of the X coordinate.
            pos_y: value of the Y coordinate.
            speed: speed move camera.

        Returns:
            Returns the response from the device to the command sent

        """
        pan_tilt = str(pos_x) + "," + str(pos_y)
        return self._camera_command({'center': pan_tilt, 'speed': speed})

    def area_zoom(self, pos_x: int = None, pos_y: int = None, zoom: int = None,
                  speed: int = None):
        """
        Centers on positions x,y (like the center command) and zooms by a factor of z/100.

        Args:
            pos_x: value of the X coordinate.
            pos_y: value of the Y coordinate.
            zoom: zooms by a factor.
            speed: speed move camera.

        Returns:
            Returns the response from the device to the command sent

        """
        xyzoom = str(pos_x) + "," + str(pos_y) + "," + str(zoom)
        return self._camera_command({'areazoom': xyzoom, 'speed': speed})

    def move(self, position: str = None, speed: float = None):
        """
        Moves the device 5 degrees in the specified direction.

        Args:
            position: position to move. (home, up, down, left, right, upleft, upright, downleft...)
            speed: speed move camera.

        Returns:
            Returns the response from the device to the command sent

        """
        return self._camera_command({'move': str(position), 'speed': speed})

    def go_home_position(self, speed: int = None):
        """
        Operation to move the PTZ device to it's "home" position.

        Args:
            speed: speed move camera.

        Returns:
            Returns the response from the device to the command sent

        """
        return self._camera_command({'move': 'home', 'speed': speed})

    def get_ptz(self):
        """
        Operation to request PTZ status.

        Returns:
            Returns a tuple with the position of the camera (P, T, Z)

        """
        resp = self._camera_command({'query': 'position'})
        logger.debug("response text: %s", resp.text)
        pan = float(resp.text.split()[0].split('=')[1])
        tilt = float(resp.text.split()[1].split('=')[1])
        zoom = float(resp.text.split()[2].split('=')[1])
        ptz_list = (pan, tilt, zoom)

        return ptz_list

    def go_to_server_preset_name(self, name: str = None, speed: int = None):
        """
        Move to the position associated with the preset on server.

        Args:
            name: name of preset position server.
            speed: speed move camera.

        Returns:
            Returns the response from the device to the command sent

        """
        return self._camera_command({'gotoserverpresetname': name, 'speed': speed})

    def go_to_server_preset_no(self, number: int = None, speed: int = None):
        """
        Move to the position associated with the specified preset position number.

        Args:
            number: number of preset position server.
            speed: speed move camera.

        Returns:
            Returns the response from the device to the command sent

        """
        return self._camera_command({'gotoserverpresetno': number, 'speed': speed})

    def go_to_device_preset(self, preset_pos: int = None, speed: int = None):
        """
        Bypasses the presetpos interface and tells the device to go directly to the preset
        position number stored in the device, where the is a device-specific preset position number.

        Args:
            preset_pos: number of preset position device
            speed: speed move camera

        Returns:
            Returns the response from the device to the command sent

        """
        return self._camera_command({'gotodevicepreset': preset_pos, 'speed': speed})

    def list_preset_device(self):
        """
        List the presets positions stored in the device.

        Returns:
            Returns the list of presets positions stored on the device.

        """
        return self._camera_command({'query': 'presetposcam'})

    def list_all_preset(self):
        """
        List all available presets position.

        Returns:
            Returns the list of all presets positions.

        """
        resp = self._camera_command({'query': 'presetposall'})
        soup = BeautifulSoup(resp.text, features="lxml")
        resp_presets = soup.text.split('\n')
        presets = []

        for i in range(1, len(resp_presets)-1):
            preset = resp_presets[i].split("=")
            presets.append((int(preset[0].split('presetposno')[1]), preset[1].rstrip('\r')))

        return presets

    def set_speed(self, speed: int = None):
        """
        Sets the head speed of the device that is connected to the specified camera.
        Args:
            speed: speed value.

        Returns:
            Returns the response from the device to the command sent.

        """
        return self._camera_command({'speed': speed})

    def get_speed(self):
        """
        Requests the camera's speed of movement.

        Returns:
            Returns the camera's move value.

        """
        resp = self._camera_command({'query': 'speed'})
        return int(resp.text.split()[0].split('=')[1])

    def info_ptz_comands(self):
        """
        Returns a description of available PTZ commands. No PTZ control is performed.

        Returns:
            Success (OK and system log content text) or Failure (error and description).

        """
        resp = self._camera_command({'info': '1'})
        return resp.text

    def snap_shot(self, filename: str = None):
        """
        Captures and image from the PTZ camera.
        Success Image is saved as the filename.

        Args:
            filename: name of the file to save the image.

        """
        start_time = time.time()
        lap = 0.0

        # URL to capture image
        url = f"http://{self.__cam_ip}/axis-cgi/jpg/image.cgi"
        while lap < FOCUS_THRESHOLD:
            # Sometimes it need '@', sometimes don't
            # need to check the path
            # os.system("wget " + url + ' -O ' + directory.replace(' ', '_'))
            filename = filename.replace(' ', '_')
            # check this out for 401 error reason
            # https://stackoverflow.com/questions/2384230/what-is-digest-authentication
            res = requests.get(url,
                               auth=HTTPDigestAuth(self.__cam_user,
                                                   self.__cam_password),
                               timeout=5)
            if res.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(res.content)
            else:
                logger.error('Failed to capture image, will try again in 1 second')
                logger.error('Status code: %s', res.status_code)
            time.sleep(1)

            # Load the image
            image = cv2.imread(filename)
            # compute the Laplacian of the image and then return the focus
            # measure, which is simply the variance of the Laplacian
            lap = cv2.Laplacian(image, cv2.CV_64F).var()
            logger.info('lap is %s', lap)
            if time.time() - start_time > TIME_TOLERANCE:
                break


