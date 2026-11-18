##
#
# Minimal OptiTrack client: pose of one rigid body streamed by Motive (NatNet 4.1+, multicast).
#
##

# standard imports
import argparse
import socket
import struct
import threading
import time


############################################################################
# CONSTANTS
############################################################################

# NatNet message IDs, ports, and rigid body layout
NAT_REQUEST_MODELDEF = 4
NAT_MODELDEF = 5
NAT_FRAMEOFDATA = 7
NAT_DATASET_RIGIDBODY = 1
MULTICAST_GROUP = "239.255.42.99"
COMMAND_PORT = 1510
DATA_PORT = 1511
RIGID_BODY = struct.Struct("<i3f4ffH")   # id, pos (x, y, z), quat (x, y, z, w), mean error, params


############################################################################
# MOCAP CLIENT
############################################################################

class MocapClient:
    """
    Keep the latest tracked pose of one Motive rigid body (by name), updated in the background.
    """

    def __init__(self, server_ip: str, body_name: str):

        # ask Motive for its model definitions, retrying lost packets (this socket also gives our local IP)
        cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cmd.connect((server_ip, COMMAND_PORT))
        cmd.settimeout(1.0)
        for _ in range(3):
            cmd.send(struct.pack("<HH", NAT_REQUEST_MODELDEF, 0))
            try:
                data = cmd.recv(65535)
            except (socket.timeout, ConnectionRefusedError):
                continue
            if struct.unpack_from("<H", data)[0] == NAT_MODELDEF:
                break
        else:
            raise TimeoutError(f"No reply from Motive at [{server_ip}]: check streaming is on and the network.")
        local_ip = cmd.getsockname()[0]
        cmd.close()

        # find the body's streaming ID: descriptions are [type, size, data], a rigid body starts with name\0, ID
        self.body_id = None
        off = 8
        for _ in range(struct.unpack_from("<i", data, 4)[0]):
            data_type, size = struct.unpack_from("<ii", data, off)
            off += 8
            if data_type == NAT_DATASET_RIGIDBODY and data[off:].startswith(body_name.encode() + b"\0"):
                self.body_id = struct.unpack_from("<i", data, off + len(body_name) + 1)[0]
                break
            off += size
        if self.body_id is None:
            raise KeyError(f"No rigid body [{body_name}] in Motive.")

        # join the multicast stream on the interface that routes to Motive
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", DATA_PORT))
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                             socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton(local_ip))

        # latest tracked (pos, quat (w, x, y, z), arrival time), written only by the receive thread
        self.pose = None
        threading.Thread(target=self.receive_loop, daemon=True).start()

    # receive every frame and keep our body's pose whenever it is tracked
    def receive_loop(self):
        while True:
            data = self.sock.recv(65535)
            t_arrival = time.monotonic()

            # a malformed or truncated packet is skipped, so it can never kill this thread
            try:
                if struct.unpack_from("<H", data)[0] != NAT_FRAMEOFDATA:
                    continue

                # skip header and frame number, then marker sets and loose markers ([count, size, data] each)
                off = 8
                off += 8 + struct.unpack_from("<i", data, off + 4)[0]
                off += 8 + struct.unpack_from("<i", data, off + 4)[0]

                # rigid bodies [count, size, bodies]: read only IDs until ours shows up
                n_bodies = struct.unpack_from("<i", data, off)[0]
                off += 8
                for _ in range(n_bodies):
                    if struct.unpack_from("<i", data, off)[0] == self.body_id:
                        _, px, py, pz, qx, qy, qz, qw, _, params = RIGID_BODY.unpack_from(data, off)
                        if params & 0x01:
                            self.pose = ((px, py, pz), (qw, qx, qy, qz), t_arrival)
                        break
                    off += RIGID_BODY.size
            except struct.error:
                continue

    # latest tracked ((x, y, z), (w, x, y, z), age [s]); None if never tracked or older than max_age [s]
    def get_pose(self, max_age: float = 0.1):
        pose = self.pose
        if pose is None:
            return None
        age = time.monotonic() - pose[2]
        return (pose[0], pose[1], age) if age <= max_age else None


############################################################################
# MAIN
############################################################################

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Print the pose of one Motive rigid body.')
    parser.add_argument('--server', type=str, default='192.168.1.15', help='IP of the Motive PC')
    parser.add_argument('--body', type=str, default='kcf', help='rigid body name in Motive')
    args = parser.parse_args()

    # connect to Motive and track the body
    mocap = MocapClient(args.server, args.body)
    print(f"Tracking rigid body [{args.body}] (ID {mocap.body_id}) from Motive [{args.server}].")

    # print the pose at 10 Hz (display only, the pose itself updates every frame)
    try:
        while True:
            time.sleep(0.1)
            pose = mocap.get_pose()
            if pose is None:
                print(f"\rBody [{args.body}] not tracked (no pose in the last 0.1 s).\033[K", end="", flush=True)
                continue
            pos, quat, age = pose
            print(f"\rpos [{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}] m  "
                  f"quat (w, x, y, z) [{quat[0]:+.3f} {quat[1]:+.3f} {quat[2]:+.3f} {quat[3]:+.3f}]  "
                  f"age {1000 * age:5.1f} ms\033[K", end="", flush=True)
    except KeyboardInterrupt:
        print()
