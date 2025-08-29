# The situation is the following
# gr00t
#   - policy_server
#   - no robot_client but an interface class to communicate with the server (through zmq or http)
# lerobot
#    - policy_server
#    - policy client (non-compatible with gr00t policy server)
# Easiest integration:
# start from the lerobot policy_client, remove grpc and use the gr00t interface class to communicate with
# gr00t policy_server. This will keep the lerobot async logic described in their blog post intact.
# if we do not want to modify the gr00t policy_server we will need to change the data format sent by the client
# (e.g. no must_go flag and hopefully not much more)

