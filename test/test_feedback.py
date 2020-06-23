import matplotlib.pyplot as plt
import numpy as np

from gym_guppy.tools import _feedback

TRAJECTORY_LEN = 500
GUPPY_POSE = np.array([-0.05, -0.08, 0.0])
ROBOT_POSE = np.array([0.01, 0.02, 0.0])


def plot():
    def no_movement():
        for i in range(TRAJECTORY_LEN):
            feedback.update(guppy_pose=GUPPY_POSE.copy(), robot_pose=ROBOT_POSE.copy())
            fear.append(feedback.fear)
            getfear.append(feedback._get_fear())
            follow.append(feedback.follow)
            getfollow.append(feedback._get_follow())
            robot[i, :] = ROBOT_POSE
            guppy[i, :] = GUPPY_POSE
            approach_dist.append(feedback.get_approach_dist())

    def approach(robot_to_fish):
        def approach_inner():
            robot_pose = ROBOT_POSE.copy()
            guppy_pose = GUPPY_POSE.copy()
            initial_dist = np.linalg.norm(
                np.array(GUPPY_POSE[:2]) - np.array(ROBOT_POSE[:2])
            )
            for i in range(TRAJECTORY_LEN):
                v = np.array(np.array(robot_pose[:2] - guppy_pose[:2]))
                if robot_to_fish:
                    v = -v
                norm = np.linalg.norm(v)
                if norm:
                    v = v / norm
                    speed = 3 * initial_dist / TRAJECTORY_LEN
                    # move both in same direction
                    robot_pose[:2] = robot_pose[:2] + v * speed
                    guppy_pose[:2] = guppy_pose[:2] + v * speed
                feedback.update(
                    guppy_pose=guppy_pose.copy(), robot_pose=robot_pose.copy()
                )
                fear.append(feedback.fear)
                getfear.append(feedback._get_fear())
                follow.append(feedback.follow)
                getfollow.append(feedback._get_follow())
                approach_dist.append(feedback.get_approach_dist())
                robot[i, :] = robot_pose
                guppy[i, :] = guppy_pose

        return approach_inner

    cases = [approach(robot_to_fish=False), approach(robot_to_fish=True), no_movement]
    fig, axes = plt.subplots(nrows=len(cases), ncols=2, figsize=(12, 16))

    for i, case in enumerate(cases):
        feedback = _feedback.Feedback()
        fear = []
        follow = []
        getfear = []
        getfollow = []
        approach_dist = []
        robot = np.zeros((TRAJECTORY_LEN, 3))
        guppy = np.zeros((TRAJECTORY_LEN, 3))

        case()

        axes[i][0].plot(fear, label="fear")
        axes[i][0].plot(getfear, label="get_fear")
        axes[i][0].plot(follow, label="follow")
        axes[i][0].plot(getfollow, label="get_follow")
        axes[i][0].plot(approach_dist, label="approach_dist")
        axes[i][0].legend()
        # axes[i][1].axis("equal")
        axes[i][1].plot(robot[:, 0], robot[:, 1])
        axes[i][1].scatter(robot[-1, 0], robot[-1, 1], label="robot")
        axes[i][1].plot(guppy[:, 0], guppy[:, 1])
        axes[i][1].scatter(guppy[-1, 0], guppy[-1, 1], label="guppy")
        axes[i][1].set_ylim(top=-0.5, bottom=0.5)
        axes[i][1].set_xlim(left=-0.5, right=0.5)
        axes[i][1].legend()
    plt.show()


if __name__ == "__main__":
    plot()
