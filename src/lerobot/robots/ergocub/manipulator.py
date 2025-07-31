import numpy as np
from os import path
from klampt.model import ik
from klampt import WorldModel


class Manipulator:
    names = ['ergocub_hand_left', 'ergocub_hand_right']
    num_anchor = 5

    def __init__(self, urdf_path, verbose=False):
        """
        init manipulator class
        :param model_name: str name of the available model
        :param verbose: bool verbose model details
        """
        
        self.world = WorldModel()

        # load robot
        f_urdf = path.join(urdf_path)
        self.world.loadRobot(f_urdf)
        self.robot = self.world.robot(0)
        self.dof = self.robot.numDrivers()
        self.ik_dof = [self.robot.driver(i).getName() for i in range(self.dof)]

        # verbose
        if verbose:
            print('-------------------------------------------')
            print('LINK: idx | type | name | l_limit | u_limit')
            for i in range(self.robot.numLinks()):
                print(i, self.robot.getJointType(i), self.robot.link(i).name,
                      self.robot.getJointLimits()[0][i], self.robot.getJointLimits()[1][i])
            print('---------------------------------------------')
            print('DRIVER: idx | type | name | l_limit | u_limit')
            for i in range(self.dof):
                print(i, self.robot.driver(i).getType(), self.robot.driver(i).getName(),
                      self.robot.driver(i).getLimits()[0], self.robot.driver(i).getLimits()[1])

    def reset(self):
        """
        reset all joint configuration to init state
        """
        self.robot.setConfig([0 for _ in self.robot.getConfig()])

    def forward_kinematic(self, q):
        """
        set driver joint values
        :param q: list of joint values
        :return: None
        """
        assert len(q) == self.dof, (
            'q is not in the correct length. Expect {}, given {}'.format(self.dof, len(q)))
        self.robot.setConfig(self.robot.configFromDrivers(q))

    def get_anchor(self):
        """
        get anchors position
        :return: list of list: anchor x,y,z position [num_anchor, 3]
        """
        anchors = [self.robot.link('A_{:02d}'.format(i)).getTransform()[1] for i in range(self.num_anchor)]
        return anchors

    def get_driver_value(self):
        """
        get driver joint value
        :return: list of joint value
        """
        q = [self.robot.driver(i).getValue() for i in range(self.dof)]
        return q

    def inverse_kinematic(self, pos_anchor, niter=20000):
        """
        set driver joint values from anchor position
        :param pos_anchor: list of anchor position [x,y,z]
        :param niter: int number of iteration
        :return: None
        """
        objs = [ik.objective(self.robot.link('A_{:02d}'.format(i)), local=[0, 0, 0], world=pos_anchor[i])
                for i in range(len(pos_anchor))]
        ik.solve(objs, iters=niter, tol=1e-3, activeDofs=self.ik_dof)

    def denormalize_joint(self, qn):
        """
        convert normalized values to actual driver joint values
        :param qn: list of normalized(0-1) values
        :return: joint values respecting joint limit
        """
        assert len(qn) == self.dof
        q = []
        for i in range(self.dof):
            l_limit, u_limit = self.robot.driver(i).getLimits()
            q.append(l_limit + (u_limit - l_limit) * qn[i])
        return q

    @staticmethod
    def q1_to_q2(q1):
        """
        Convert ergocub-hand-finger q1 to q2 .
        :param q1: list of degree of q1
        :return: list of degree of q2
        """

        # Mk5.0 config
        l0x = [-0.00555, -0.005, -0.005, -0.005, -0.005]
        l0y = [0.00285, 0.004, 0.004, 0.004, 0.004]
        q1off = [4.29, 2.86, 2.86, 2.86, 3.43]
        q2bias = [-180, -173.35, -173.35, -173.35, -170.54]
        d = [0.02006, 0.03004, 0.03004, 0.03004, 0.02504]
        l = [0.0085, 0.00604, 0.00604, 0.00604, 0.00608]
        k = [0.0171, 0.02918, 0.02918, 0.02918, 0.02425]

        q2 = []
        for f in range(5):
            p1x_q1 = d[f] * np.cos(np.deg2rad(q1[f] + q1off[f]))
            p1y_q1 = d[f] * np.sin(np.deg2rad(q1[f] + q1off[f]))
            h_q1 = ((p1x_q1 - l0x[f]) ** 2 + (p1y_q1 - l0y[f]) ** 2) ** 0.5

            q2f = (np.arctan2(p1y_q1 - l0y[f], p1x_q1 - l0x[f]) +
                   np.arccos((l[f] ** 2 - k[f] ** 2 + h_q1 ** 2) / (2 * l[f] * h_q1)) +
                   -np.deg2rad(q2bias[f]) - np.pi)
            q2.append(np.rad2deg(q2f) - q1[f])
        return q2

    @staticmethod
    def linear_mimic():
        """
        compute parameter of linear regression for finger joints (from q1 to q2)
        https://icub-tech-iit.github.io/documentation/hands/hands_mk5_coupling/#coupling-laws
        Pre-computed Result:
        thumb : 0.5909373177810342, 0.0822527687394746
        index : 1.0362078387151505, 0.0752746251429467
        middle: 1.0362078387151505, 0.0752746251429467
        ring  : 1.0362078387151505, 0.0752746251429467
        pinky : 0.9857188196265708, 0.0300500624978842
        """
        import matplotlib.pyplot as plt

        # convert
        q1max = [86.35, 90, 90, 90, 90]
        n_sample = 64
        q1s = np.stack([np.linspace(0, q1max_, n_sample) for q1max_ in q1max], axis=1)
        q2s = np.stack([Manipulator.q1_to_q2(q1s[i]) for i in range(n_sample)], axis=0)
        multi_offset = []

        # linear fitting
        for f, fname in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
            z = np.polyfit(q1s[:, f], q2s[:, f], 1)
            multi_offset.append([fname, z[0], np.deg2rad(z[1])])

            # plotting
            plt.plot(q1s[:, f], q2s[:, f], '.', label=fname)
            plt.plot(q1s[:, f], q1s[:, f] * z[0] + z[1], label=fname + '-linear')

        print(*multi_offset, sep='\n')
        plt.xlabel('q1')
        plt.ylabel('q2')
        plt.xlim(0, 90)
        plt.ylim(0, 100)
        plt.legend(loc='upper left')
        plt.show()