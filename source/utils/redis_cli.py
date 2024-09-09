import redis
import time
import uuid
import sys

class MultiLockerSystem:
    def __init__(self, redis_host, redis_port, redis_password, locker_prefix, num_lockers, expire_in_sec=10):
        self.redis = redis.Redis(host=redis_host, port=redis_port, password=redis_password, db=0)
        self.locker_prefix = locker_prefix
        self.num_lockers = num_lockers
        self.expire_in_sec = expire_in_sec
        self.identifier = str(uuid.uuid4())

    def acquire_locker(self):
        end = time.time() + self.expire_in_sec
        while time.time() < end:
            for i in range(self.num_lockers):
                locker_name = f"{self.locker_prefix}:{i}"
                if self.redis.setnx(locker_name, self.identifier):
                    self.redis.expire(locker_name, self.expire_in_sec)
                    return i  # Return the locker number
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
