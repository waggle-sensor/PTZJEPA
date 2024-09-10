import os
import redis
import signal
import time
import uuid
import sys
import subprocess
from pathlib import Path

class MultiLockerSystem:
    def __init__(self, redis_host, redis_port, redis_password, locker_prefix, num_lockers, expire_in_sec=20000):
        self.redis = redis.Redis(host=redis_host, port=redis_port, password=redis_password, db=0)
        self.locker_prefix = locker_prefix
        self.num_lockers = num_lockers
        self.expire_in_sec = expire_in_sec
        self.identifier = str(uuid.uuid4())

    def acquire_locker(self, locker_num):
        end = time.time() + self.expire_in_sec
        while time.time() < end:
            #for i in range(self.num_lockers):
                #locker_name = f"{self.locker_prefix}:{i}"
            locker_name = f"{self.locker_prefix}:{locker_num}"
            if self.redis.setnx(locker_name, self.identifier):
                self.redis.expire(locker_name, self.expire_in_sec)
                return locker_num  # Return the locker number
                #return i  # Return the locker number
            time.sleep(0.1)
        return None  # Couldn't acquire any locker within the time limit

    def release_locker(self, locker_num):
        locker_name = f"{self.locker_prefix}:{locker_num}"
        pipe = self.redis.pipeline(True)
        while True:
            try:
                pipe.watch(locker_name)
                if pipe.get(locker_name).decode('utf-8') == self.identifier:
                    pipe.multi()
                    pipe.delete(locker_name)
                    pipe.execute()
                    return True
                pipe.unwatch()
                break
            except redis.WatchError:
                pass
        return False







class SSHFSMounter:
    def __init__(self, host_username, host_ip, host_data_directory, local_data_directory, password):
        self.host_username = host_username
        self.host_ip = host_ip
        self.host_data_directory = host_data_directory
        self.local_data_directory = local_data_directory
        self.password = password

    def mount(self):
        Path(self.local_data_directory).mkdir(parents=True, exist_ok=True)

        mount_command = [
            'sshfs', '-v',
            '-o', f'password_stdin',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'debug',
            f'{self.host_username}@{self.host_ip}:{self.host_data_directory}',
            self.local_data_directory
        ]
        
        print(f"Executing command: {' '.join(mount_command)}")
        
        process = subprocess.Popen(mount_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=self.password)
        
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        
        if process.returncode != 0:
            print(f"Mount failed. Return code: {process.returncode}")
            print(f"Error: {stderr}")
            self.check_ssh_connection()
        else:
            print("Mount command completed. Verifying mount...")
            self.verify_mount()

    def check_ssh_connection(self):
        print("Checking SSH connection...")
        ssh_command = ['ssh', '-v', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no', f'{self.host_username}@{self.host_ip}', 'echo', 'SSH connection successful']
        
        process = subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        print(f"SSH STDOUT: {stdout}")
        print(f"SSH STDERR: {stderr}")
        
        if process.returncode == 0:
            print("SSH connection successful")
        else:
            print(f"SSH connection failed. Return code: {process.returncode}")

    def verify_mount(self):
        try:
            result = subprocess.run(['mountpoint', '-q', self.local_data_directory], check=True)
            print(f"Mount verified successfully.")
        except subprocess.CalledProcessError:
            print(f"Mount verification failed. The directory is not mounted.")
        
        try:
            contents = os.listdir(self.local_data_directory)
            print(f"Contents of {self.local_data_directory}:")
            for item in contents:
                print(f" - {item}")
        except Exception as e:
            print(f"Failed to list directory contents: {e}")

    def unmount(self):
        unmount_command = ['fusermount', '-u', '-v', self.local_data_directory]
        result = subprocess.run(unmount_command, capture_output=True, text=True)
        print(f"Unmount STDOUT: {result.stdout}")
        print(f"Unmount STDERR: {result.stderr}")
        if result.returncode != 0:
            print(f"Unmount failed. Return code: {result.returncode}")
            self.verify_mount()
        else:
            print("Unmount successful")

    def create_test_file(self, filename):
        path = Path(self.local_data_directory) / filename
        try:
            path.touch()
            print(f"Created test file: {path}")
            if path.exists():
                print(f"File {filename} exists in the mounted directory.")
            else:
                print(f"File {filename} does not exist in the mounted directory.")
        except Exception as e:
            print(f"Failed to create test file: {e}")







class SSHFSMounter_1:
    def __init__(self, host_username, host_ip, host_data_directory, local_data_directory, password):
        self.host_username = host_username
        self.host_ip = host_ip
        self.host_data_directory = host_data_directory
        self.local_data_directory = local_data_directory
        self.password = password

    def mount(self):
        print("Starting mount process...")
        Path(self.local_data_directory).mkdir(parents=True, exist_ok=True)

        mount_command = [
            'sshfs', '-v',
            '-o', f'password_stdin',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'debug',
            f'{self.host_username}@{self.host_ip}:{self.host_data_directory}',
            self.local_data_directory
        ]
        
        print(f"Executing command: {' '.join(mount_command)}")
        
        try:
            process = subprocess.Popen(mount_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            def timeout_handler(signum, frame):
                print("Mount process timed out")
                process.kill()
                raise TimeoutError("Mount process timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # Set a 30-second timeout

            print("Sending password...")
            stdout, stderr = process.communicate(input=self.password, timeout=30)
            
            signal.alarm(0)  # Cancel the alarm

            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            
            if process.returncode != 0:
                print(f"Mount failed. Return code: {process.returncode}")
                print(f"Error: {stderr}")
                self.check_ssh_connection()
            else:
                print("Mount command completed. Verifying mount...")
                self.verify_mount()
        except subprocess.TimeoutExpired:
            print("Mount process timed out")
            process.kill()
        except Exception as e:
            print(f"An error occurred: {e}")

    def check_ssh_connection(self):
        print("Checking SSH connection...")
        ssh_command = ['ssh', '-v', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no', f'{self.host_username}@{self.host_ip}', 'echo', 'SSH connection successful']
        
        try:
            process = subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=10)
            
            print(f"SSH STDOUT: {stdout}")
            print(f"SSH STDERR: {stderr}")
            
            if process.returncode == 0:
                print("SSH connection successful")
            else:
                print(f"SSH connection failed. Return code: {process.returncode}")
        except subprocess.TimeoutExpired:
            print("SSH connection timed out")
            process.kill()
        except Exception as e:
            print(f"An error occurred during SSH check: {e}")

    def verify_mount(self):
        try:
            result = subprocess.run(['mountpoint', '-q', self.local_data_directory], check=True, timeout=5)
            print(f"Mount verified successfully.")
        except subprocess.CalledProcessError:
            print(f"Mount verification failed. The directory is not mounted.")
        except subprocess.TimeoutExpired:
            print("Mount verification timed out")
        
        try:
            contents = os.listdir(self.local_data_directory)
            print(f"Contents of {self.local_data_directory}:")
            for item in contents:
                print(f" - {item}")
        except Exception as e:
            print(f"Failed to list directory contents: {e}")

    def unmount(self):
        print("Starting unmount process...")
        unmount_command = ['fusermount', '-u', '-v', self.local_data_directory]
        try:
            result = subprocess.run(unmount_command, capture_output=True, text=True, timeout=10)
            print(f"Unmount STDOUT: {result.stdout}")
            print(f"Unmount STDERR: {result.stderr}")
            if result.returncode != 0:
                print(f"Unmount failed. Return code: {result.returncode}")
                self.verify_mount()
            else:
                print("Unmount successful")
        except subprocess.TimeoutExpired:
            print("Unmount process timed out")
        except Exception as e:
            print(f"An error occurred during unmount: {e}")









class SSHFSMounter_2:
    def __init__(self, host_username, host_ip, host_data_directory, local_data_directory, password):
        self.host_username = host_username
        self.host_ip = host_ip
        self.host_data_directory = host_data_directory
        self.local_data_directory = local_data_directory
        self.password = password

    def mount(self):
        print("Starting mount process...")
        Path(self.local_data_directory).mkdir(parents=True, exist_ok=True)

        mount_command = [
            'sshpass', '-p', self.password,
            'sshfs', '-v',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'debug',
            f'{self.host_username}@{self.host_ip}:{self.host_data_directory}',
            self.local_data_directory
        ]
        
        print(f"Executing command: {' '.join(mount_command)}")
        
        try:
            process = subprocess.Popen(mount_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            def timeout_handler(signum, frame):
                print("Mount process timed out")
                process.kill()
                raise TimeoutError("Mount process timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # Set a 30-second timeout

            stdout, stderr = process.communicate(timeout=30)
            
            signal.alarm(0)  # Cancel the alarm

            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            
            if process.returncode != 0:
                print(f"Mount failed. Return code: {process.returncode}")
                print(f"Error: {stderr}")
                self.check_ssh_connection()
            else:
                print("Mount command completed. Verifying mount...")
                self.verify_mount()
        except subprocess.TimeoutExpired:
            print("Mount process timed out")
            process.kill()
        except Exception as e:
            print(f"An error occurred: {e}")

    def check_ssh_connection(self):
        print("Checking SSH connection...")
        ssh_command = ['sshpass', '-p', self.password, 'ssh', '-v', '-o', 'StrictHostKeyChecking=no', f'{self.host_username}@{self.host_ip}', 'echo', 'SSH connection successful']
        
        try:
            process = subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=10)
            
            print(f"SSH STDOUT: {stdout}")
            print(f"SSH STDERR: {stderr}")
            
            if process.returncode == 0:
                print("SSH connection successful")
            else:
                print(f"SSH connection failed. Return code: {process.returncode}")
        except subprocess.TimeoutExpired:
            print("SSH connection timed out")
            process.kill()
        except Exception as e:
            print(f"An error occurred during SSH check: {e}")

    def verify_mount(self):
        try:
            result = subprocess.run(['mountpoint', '-q', self.local_data_directory], check=True, timeout=5)
            print(f"Mount verified successfully.")
        except subprocess.CalledProcessError:
            print(f"Mount verification failed. The directory is not mounted.")
        except subprocess.TimeoutExpired:
            print("Mount verification timed out")
        
        try:
            contents = os.listdir(self.local_data_directory)
            print(f"Contents of {self.local_data_directory}:")
            for item in contents:
                print(f" - {item}")
        except Exception as e:
            print(f"Failed to list directory contents: {e}")

    def unmount(self):
        print("Starting unmount process...")
        unmount_command = ['fusermount', '-u', '-v', self.local_data_directory]
        try:
            result = subprocess.run(unmount_command, capture_output=True, text=True, timeout=10)
            print(f"Unmount STDOUT: {result.stdout}")
            print(f"Unmount STDERR: {result.stderr}")
            if result.returncode != 0:
                print(f"Unmount failed. Return code: {result.returncode}")
                self.verify_mount()
            else:
                print("Unmount successful")
        except subprocess.TimeoutExpired:
            print("Unmount process timed out")
        except Exception as e:
            print(f"An error occurred during unmount: {e}")





class SSHFSMounter_3:
    def __init__(self, host_username, host_ip, host_data_directory, local_data_directory, password):
        self.host_username = host_username
        self.host_ip = host_ip
        self.host_data_directory = host_data_directory
        self.local_data_directory = local_data_directory
        self.password = password

    def run_command(self, command, timeout=30):
        print(f"Executing command: {' '.join(command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"Command timed out after {timeout} seconds")
            return False

    def check_ssh(self):
        print("Checking SSH connection...")
        command = ['sshpass', '-p', self.password, 'ssh', '-o', 'StrictHostKeyChecking=no', 
                   f'{self.host_username}@{self.host_ip}', 'echo', 'SSH connection successful']
        return self.run_command(command)

    def check_sftp(self):
        print("Checking SFTP connection...")
        command = ['sshpass', '-p', self.password, 'sftp', '-o', 'StrictHostKeyChecking=no', 
                   f'{self.host_username}@{self.host_ip}:/']
        return self.run_command(command, timeout=10)

    def check_fuse(self):
        print("Checking FUSE availability...")
        return self.run_command(['fusermount', '-V'])

    def check_sshfs(self):
        print("Checking sshfs availability...")
        return self.run_command(['sshfs', '-V'])

    def mount(self):
        print("Starting mount process...")
        
        if not self.check_ssh():
            print("SSH connection failed. Cannot proceed with mount.")
            return False

        if not self.check_sftp():
            print("SFTP connection failed. Cannot proceed with mount.")
            return False

        if not self.check_fuse():
            print("FUSE is not available. Cannot proceed with mount.")
            return False

        if not self.check_sshfs():
            print("sshfs is not available. Cannot proceed with mount.")
            return False

        Path(self.local_data_directory).mkdir(parents=True, exist_ok=True)

        mount_command = [
            'sshpass', '-p', self.password,
            'sshfs', '-v',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'debug',
            f'{self.host_username}@{self.host_ip}:{self.host_data_directory}',
            self.local_data_directory
        ]
        
        if self.run_command(mount_command, timeout=60):
            print("Mount command completed. Verifying mount...")
            return self.verify_mount()
        else:
            print("Mount command failed.")
            return False

    def verify_mount(self):
        try:
            result = subprocess.run(['mountpoint', '-q', self.local_data_directory], check=True, timeout=5)
            print(f"Mount verified successfully.")
            return True
        except subprocess.CalledProcessError:
            print(f"Mount verification failed. The directory is not mounted.")
        except subprocess.TimeoutExpired:
            print("Mount verification timed out")
        return False

    def unmount(self):
        print("Starting unmount process...")
        unmount_command = ['fusermount', '-u', '-v', self.local_data_directory]
        if self.run_command(unmount_command):
            print("Unmount successful")
        else:
            print("Unmount failed")
