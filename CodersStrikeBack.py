import sys
import math
import numpy as np
import time
from inspect import currentframe


class DebugTool:
    def __init__(self):
        try:
            self.fd = open(r"C:\Users\JUNJI\Documents\Condingame\pyCharmProject\CodersStrikeBack\input.txt")
        except (ImportError, OSError):
            self.debug_mode = False
        else:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.fg = None
            self.ax = None
            self.debug_mode = True

    def input(self):
        if self.debug_mode:
            data = self.fd.readline()
        else:
            data = input()
        print(data, file=sys.stderr, flush=True)
        return data

    def start_timer(self):
        self.timer = time.time()

    def elapsed_time(self):
        end_time = time.time()
        interval = end_time - self.timer
        DT.stderr(interval * 1000, "m sec")

    @staticmethod
    def stderr(*args):
        cf = currentframe()
        print(*args, "@" + str(cf.f_back.f_lineno), file=sys.stderr, flush=True)

    def plot_vector_clock(self, vct, clr="b", txt=""):
        # todo: refactor in OO style
        self.plt.plot((0, vct[0]), (0, vct[1]), color=clr)
        self.plt.text(vct[0], vct[1], txt)

    def plot_pod_trail(self, name, trl, append=None):
        """Displays a figure showing trajectory of Pod.
        To display it, you may need to set break point at plt.figure()."""

        tx = [t.location.x() for t in trl]
        ty = [t.location.y() for t in trl]

        if not self.fg or not append:
            self.fg = self.plt.figure(name)
            self.ax = self.fg.add_subplot(111)

        self.ax.set_xlim(0, 16000)
        self.ax.set_ylim(9000, 0)
        self.ax.grid()

        # trl of location by BLUE line
        self.ax.plot(tx, ty, linewidth=2, color="b")

        # start point by Blue circle
        self.ax.plot(tx[0], ty[0], "bo")

        # CP by RED circle
        c1 = self.plt.Circle((CP[p.next_cp_id][0], CP[p.next_cp_id][1]), CP_RADIUS, color="r", alpha=.2)
        c2 = self.plt.Circle((CP[p.following_cp_id(1)][0], CP[p.following_cp_id(1)][1]), CP_RADIUS, color="r", alpha=.2)
        self.ax.add_patch(c1)
        self.ax.add_patch(c2)

        for t in trl:
            # angle by BLACK arrow
            self.ax.plot((t.location[0], t.pod_angle_as_vector[0] + t.location[0]),
                         (t.location[1], t.pod_angle_as_vector[1] + t.location[1]), color="black")

            # inertia by YELLOW arrow
            self.ax.plot((t.location[0], t.inertia[0] + t.location[0]), (t.location[1], t.inertia[1] + t.location[1]),
                         color="y")

            # target by BLUE line
            # if not t.thrust_power:
            #     self.plt.plot((t.location[0], t.thrust_target[0]), (t.location[1], t.thrust_target[1]), color="b")


# Pod is instantiate EVERY TURN by using standard input
class Pod:
    """Pod is a racing flight object.
    Most of properties represent Pod status, which should be read-only.
    The following properties are to bet set in order to steer Pod, which affect the Pod's move of the next turn:
      thrust_target, thrust_power, and shield
    """

    def __init__(self, pod_id, iff, x, y, vx, vy, angle, next_cp_id):
        self.pod_id = int(pod_id)
        self.iff = str(iff)
        self.location = Vector(int(x), int(y))
        self.inertia = Vector(int(vx), int(vy))
        self.abs_pod_angle = math.radians(float(angle))
        self.pod_angle_as_vector = Vector(math.cos(self.abs_pod_angle), math.sin(self.abs_pod_angle)).as_magnitude(99)
        self.next_cp_id = int(next_cp_id)

        self.laps_to_go = LAPS_TO_GO
        self.thrust_target = Vector(0, 0)
        self.thrust_power = 0
        self.shield = 0

    def copy(self):
        pd = Pod(self.pod_id, self.iff, self.location[0], self.location[1], self.inertia[0], self.inertia[1],
                 math.degrees(self.abs_pod_angle), self.next_cp_id)
        pd.laps_to_go = self.laps_to_go
        pd.thrust_target = self.thrust_target
        pd.thrust_power = self.thrust_power
        pd.shield = self.shield
        return pd

    def following_cp_id(self, i):
        return (self.next_cp_id + i) % NUMBER_OF_CP

    def angle_for_location(self, point):
        """Returns radians between -pi and pi.
        In THIS GAME field, POSITIVE number means CLOCKWISE direction from front face of SELF to the POINT."""
        return self.pod_angle_as_vector.angle_for(point - self.location)

    def reached(self, checkpoint):
        return self.inertia.as_magnitude(-1 / FRICTION).distance_from(checkpoint - self.location) < CP_RADIUS

    def thrust_as_vector(self):
        """If thrust_power is 0, approximates it with 0.1 in order to hold its angle.
        NOTE: Direction of the actual thrust is restricted by the Pod angle and ROTATE_PER_TURN."""
        # ToDo: this method should return actual thrust considering restriction of ROTATE_PER_TURN.
        if self.thrust_power == 0:
            return (self.thrust_target - self.location).as_magnitude(MIN_THRUST)
        else:
            return (self.thrust_target - self.location).as_magnitude(self.thrust_power)

    def next_location(self):
        angle = self.pod_angle_as_vector.angle_for(self.thrust_as_vector())
        if abs(angle) < ROTATE_PER_TURN:
            actual_thrust_vector = self.thrust_as_vector()
        elif angle > 0:
            actual_thrust_vector = self.pod_angle_as_vector.rotate(ROTATE_PER_TURN)
        else:
            actual_thrust_vector = self.pod_angle_as_vector.rotate(-ROTATE_PER_TURN)
        return self.location + self.inertia + actual_thrust_vector

    def next_move(self):
        angle = self.pod_angle_as_vector.angle_for(self.thrust_as_vector())
        if abs(angle) < ROTATE_PER_TURN:
            actual_thrust_vector = self.thrust_as_vector()
        elif angle > 0:
            actual_thrust_vector = self.pod_angle_as_vector.rotate(ROTATE_PER_TURN)
            actual_thrust_vector = actual_thrust_vector.as_magnitude(max(MIN_THRUST, self.thrust_power))
        else:
            actual_thrust_vector = self.pod_angle_as_vector.rotate(-ROTATE_PER_TURN)
            actual_thrust_vector = actual_thrust_vector.as_magnitude(max(MIN_THRUST, self.thrust_power))
        loc = self.location + self.inertia + actual_thrust_vector
        inr = loc - self.location
        inr = inr.as_magnitude(inr.magnitude() * .85)
        return Pod(self.pod_id, self.iff, loc[0], loc[1], inr[0], inr[1],
                   math.degrees(actual_thrust_vector.abs_angle()), self.next_cp_id)

    def plan_rush_moves(self, checkpoint, limit_turns=11):
        """Returns a list of Pods that represents a trajectory of rushing moves to the target with max thrust heedlessly.
        The trajectory ends when Pod reaches thrust_target or when the specified turns passes.

        NOTE:
            The first element of return value represents current Pod with resetting thrust_target and thrust_power.
            Therefore, original values of thrust_target and thrust_power are ignored.

        NOTE:
            Does not update self.next_cp_id.
        """
        trj = [self.copy()]  # type: List[Pod]
        while True:
            DT.elapsed_time()

            last_pod = trj[-1]  # type: Pod
            if last_pod.reached(checkpoint) or (limit_turns and len(trj) > limit_turns):
                return trj
            last_pod.thrust_target = checkpoint - last_pod.inertia
            last_pod.thrust_power = MAX_THRUST
            trj.append(last_pod.next_move())
            if DT.debug_mode:
                DT.plot_pod_trail("rush move", trj)  # , append=True)

    def plan_operated_moves(self, thrust_operations, checkpoint=None):
        """Returns a list of Pods that represents a trajectory of moves according to the thrust_operations.
        The trajectory ends when Pod reaches checkpoint.

        NOTE:
            The first element of return value represents current Pod with resetting thrust_target and thrust_power.
            Therefore, original values of thrust_target and thrust_power are ignored.

        NOTE:
            Does not update self.next_cp_id.
        """
        trj = [self.copy()]  # type: List[Pod]
        for ope in thrust_operations:  # type: List[Vector, float] # [thrust_target, thrust_power]
            last_pod = trj[-1]  # type: Pod
            if last_pod.reached(checkpoint):
                return trj
            last_pod.thrust_target = ope[0]
            last_pod.thrust_power = ope[1]
            trj.append(last_pod.next_move())
        if DT.debug_mode:
            DT.plot_pod_trail("operated move", trj)
        return trj

    def plan_ep_move(self):
        """Returns a list of Pods that represents trajectory of moves according to the simple Early-Pivot tactics.
        If location_without_thrust reaches next_checkpoint within turn_to_pivot, begin pivoting for the following
        checkpoint before reaching next checkpoint.
        """
        # ToDo: Under Construction
        p = self
        return p

    def plan_fep_moves(self):
        """Returns a list of Pods that represents trajectory of moves according to Fast-Early-Pivot tactics.
        Fast-Early-Pivot simulates beginning pivot for the following CP after the next with thrust power as
        much as possible and then choose the best way for both the next CP and the following CP.

        - Detail -
        Prerequisite: Pod angle for the next checkpoint < ROTATE_PER_TURN

        A = Target following CP and thrust max
        B = Target following CP and thrust min
        C = Target next CP and thrust max

        1. Try only A during TTR. If it reaches CP, adjust thrust.
        2. Replace last A with B, check if it reaches CP, adopt its combination.
        3. Repeat #2 and all B does not reach CP, adopt C and adjust thrust.

        + Adjusting thrust +
        1. Calculate min thrust to reach CP and the location and the inertia with the min thrust.
        2. Calculate the location and the inertia with max thrust.
        3. Choose 1 or 2 by distance from the following target to the location with inertia.
        """
        # Trajectory going straightforward to the next checkpoint
        rush_trj = self.plan_rush_moves(CP[self.next_cp_id])  # type: List[Pod]
        ttr = len(rush_trj) - 1

        # ToDo: Need to test if 10 is good for performance
        if ttr < 9:
            operation_a = [[CP[self.following_cp_id(1)], MAX_THRUST]]
            operation_b = [[CP[self.following_cp_id(1)], MIN_THRUST]]
            for i in range(ttr + 1):
                DT.elapsed_time()
                operations = operation_a * (ttr - i) + operation_b * i
                trj = self.plan_operated_moves(operations, CP[self.next_cp_id])
                if trj[-1].reached(CP[self.next_cp_id]):
                    # todo: need to adjust thrust
                    # if len(trj) <= 2:
                    #     pp = trj[0].copy()
                    #     pp.thrust_power=
                    return trj
        return rush_trj


class Vector(np.ndarray):
    def __new__(cls, x, y):
        vctr = np.r_[x, y]
        return vctr.view(cls)

    def magnitude(self):
        return np.linalg.norm(self)

    def x(self):
        return self[0]

    def y(self):
        return self[1]

    def normalized(self):
        if self.magnitude() == 0:
            return Vector(0, 0)
        else:
            copy = self.copy()
            copy = copy / copy.magnitude()
            return copy

    def as_magnitude(self, scalar):
        return self.normalized() * scalar

    def abs_angle(self):
        """Returns radians between 0 and 2 * pi for absolute angle.
        In THIS GAME field, 0 means facing EAST while 90 means facing SOUTH."""
        # Flip because atan2() takes y then x
        angle = math.atan2(*np.flipud(self))

        # Make angle hold radians between 0 and 2 * pi
        # because atan2() returns radians between -pi and pi
        if angle < 0:
            angle += math.pi * 2
        return angle

    def angle_for(self, vector):
        """Returns radians between -pi and pi.
          In THIS GAME field, POSITIVE number means CLOCKWISE direction from SELF to VECTOR."""
        diff = vector.abs_angle() - self.abs_angle()
        # print("vctr", vector.abs_angle(), "self", self.abs_angle(), file=sys.stderr)
        if abs(diff) < math.pi:
            return diff
        elif self.abs_angle() < math.pi:
            return diff - math.pi * 2
        else:
            return diff + math.pi * 2

    def distance_from(self, point_as_vector):
        """Returns the minimum distance from the point(vector) to the line segment(self)."""
        if np.dot(self, point_as_vector) < 0:
            return point_as_vector.magnitude()
        elif np.dot(-self, point_as_vector - self) < 0:
            return (point_as_vector - self).magnitude()
        else:
            # print(self, point_as_vector, np.cross(self, point_as_vector), self.magnitude(), file=sys.stderr)
            return abs(np.cross(self, point_as_vector) / self.magnitude())

    def rotate(self, r):
        rot = np.matrix(((math.cos(r), math.sin(r)), (-math.sin(r), math.cos(r))))
        # print(np.dot(self, rot), file=sys.stderr)
        # print(np.array(np.dot(self, rot)).ravel(), file=sys.stderr)
        return Vector(*np.array(np.dot(self, rot)).ravel())


# Accept turn_history: [current[turn#, [ally Pod1, ally Pod2, enemy Pod1, enemy Pod2]], previous[...]]
# Reject Pod1.history(0) = previous Pod1
class TurnHistory:
    def __init__(self, turn=0, pods=()):
        self.current_turn = turn
        self.state = [pods]  # type: List[List[Pod]]

    def turn_end(self, pods):
        self.state.append(pods)
        self.current_turn += 1

    def pod_last_state(self, pod_id, i=0):
        i += self.current_turn - 1
        if len(self.state):
            return self.state[max(0, i)][pod_id]
        else:
            return None


def geometric_series(a, r, n):
    # sigma{ar**n}
    if n:
        return sum([a * r ** i for i in range(n)])
    else:
        return a * n


def center_of_three_points(p1, p2, p3):
    x = p1[0] + p1[1] * 1j
    y = p2[0] + p2[1] * 1j
    z = p3[0] + p3[1] * 1j
    w = z - x
    w /= y - x
    c = ((x - y) * (w - abs(w) ** 2) / 2j / w.imag - x) * -1
    return c.real, c.imag


DT = DebugTool()

# Constants
ROTATE_PER_TURN = math.radians(18)
CP_RADIUS = 600
POD_RADIUS = 400
FRICTION = 0.85
MAX_THRUST = 100
MIN_THRUST = 0.1

LAPS_TO_GO = int(DT.input())
NUMBER_OF_CP = int(DT.input())

CP = []  # type: List[Vector]
for i in range(NUMBER_OF_CP):
    cp_x, cp_y = [int(j) for j in DT.input().split()]
    # print(checkpoint_x, checkpoint_y, file=sys.stderr)
    CP.append(Vector(cp_x, cp_y))

# Constants for tactics
OUT_IN_OUT = []
EARLY_PIVOT = []
FAST_EARLY_PIVOT = [0, 1]
CIRCULAR_MOVE = []

history = TurnHistory()

# Game loop
while True:
    DT.start_timer()

    all_pods = []  # type: List[Pod]
    for i in range(2):
        all_pods.append(Pod(i, "ally", *DT.input().split()))
        if history.current_turn > 1:
            if all_pods[i].next_cp_id == 1 and history.pod_last_state(i).next_cp_id == 0:
                all_pods[i].laps_to_go = history.pod_last_state(i).laps_to_go - 1
            else:
                all_pods[i].laps_to_go = history.pod_last_state(i).laps_to_go

    for i in range(2):
        all_pods.append(Pod(i + 2, "enemy", *DT.input().split()))
        if history.current_turn > 1:
            if all_pods[i + 2].next_cp_id == 1 and history.pod_last_state(i + 2).next_cp_id == 0:
                all_pods[i + 2].laps_to_go = history.pod_last_state(i + 2).laps_to_go - 1
            else:
                all_pods[i + 2].laps_to_go = history.pod_last_state(i + 2).laps_to_go

    for p in all_pods[0:2]:  # type: Pod # ally pods
        DT.stderr(history.current_turn, LAPS_TO_GO, p.next_cp_id)

        # BOOST at the first turn
        if history.current_turn == 0 and not DT.debug_mode:
            p.thrust_target = CP[p.next_cp_id]
            p.thrust_power = float("inf")

        # Go straight for the final checkpoint
        elif p.laps_to_go == 1 and p.next_cp_id == 0:
            p.thrust_target = CP[p.next_cp_id] - p.inertia
            p.thrust_power = float("inf")

        # Depending on angle to CP, begin OUT_IN_OUT Move
        elif p.pod_id in OUT_IN_OUT and abs(p.angle_for_location(CP[p.next_cp_id])) < ROTATE_PER_TURN * 2:
            # To determine turns to begin pivot etc, roughly estimate trajectory.
            ttp = int(abs((CP[p.next_cp_id] - p.location).angle_for(
                CP[p.following_cp_id(1)] - CP[p.next_cp_id])) // ROTATE_PER_TURN)
            dst = (CP[p.next_cp_id] - p.location).magnitude()
            inr = p.inertia.magnitude() * math.cos(p.inertia.angle_for(CP[p.next_cp_id] - p.location))

            trj = [0]
            while True:
                inr = inr * .85 + MAX_THRUST
                trj.append(max(trj) + inr)
                if max(trj) > dst:
                    break
            ttr = len(trj) - 1
            DT.stderr("ttr", ttr, "ttp", ttp)

            # Simulation for ttr turns or until having enough distance to reach CP
            trj = [p]  # Type: list[Pod]
            for i in range(ttr + ttp):

                # At first, get close to CP
                if i < ttr - ttp:
                    # Basically, target is next CP with thrust MAX_THRUST.
                    target_vector = (CP[trj[i].next_cp_id] - trj[i].location).as_magnitude(MAX_THRUST)
                    if DT.debug_mode:
                        # todo: refactor plot statements in OO style and remove the use of plt
                        DT.plt.figure("vector clock" + str(i))
                        DT.plot_vector_clock(target_vector, "b", "Straight to CP")
                        DT.plot_vector_clock(trj[i].inertia, "y", "Inertia")
                        DT.plot_vector_clock(trj[i].pod_angle_as_vector, "black", "original angle")
                        DT.plt.ylim(150, -150)
                        DT.plt.xlim(-150, 150)

                    # Rotate target_vector to resist against inertia
                    compo = trj[i].inertia.magnitude() * math.sin(target_vector.angle_for(trj[i].inertia))
                    if target_vector.angle_for(trj[i].inertia) > 0:
                        rot = -math.asin(min(1, abs(compo)))
                    else:
                        rot = math.asin(min(1, abs(compo)))
                    target_vector = target_vector.rotate(rot)
                    if DT.debug_mode:
                        DT.plot_vector_clock(target_vector, "y", "Rotate for compo" + str(rot))

                    # Revise target_vector considering limit of pivot angle (ROTATE_PER_TURN)
                    angle = trj[i].pod_angle_as_vector.angle_for(target_vector)
                    if abs(angle) < ROTATE_PER_TURN:
                        pass
                    elif angle > 0:
                        target_vector = trj[i].pod_angle_as_vector.rotate(ROTATE_PER_TURN)
                    else:
                        target_vector = trj[i].pod_angle_as_vector.rotate(-ROTATE_PER_TURN)
                    if DT.debug_mode:
                        DT.plot_vector_clock(target_vector, "r", "Rotate for compo & abs angle" + str(angle))

                # Begin to pivot for the following CP except for the case CP is in small angle
                else:
                    angle = trj[i].pod_angle_as_vector.angle_for(CP[trj[i].following_cp_id(1)] - CP[trj[i].next_cp_id])
                    if abs(angle) < ROTATE_PER_TURN:
                        target_vector = trj[i].pod_angle_as_vector
                    elif angle > 0:
                        target_vector = trj[i].pod_angle_as_vector.rotate(ROTATE_PER_TURN)
                    else:
                        target_vector = trj[i].pod_angle_as_vector.rotate(-ROTATE_PER_TURN)

                    if DT.debug_mode and p.pod_id == 1:
                        DT.plt.figure("vector clock" + str(i))
                        DT.plot_vector_clock(target_vector, "r", "Rotate RPT for following CP" + str(angle))
                        DT.plot_vector_clock(trj[i].inertia, "y", "Inertia")
                        DT.plot_vector_clock(trj[i].pod_angle_as_vector, "black", "original angle")
                        DT.plt.ylim(150, -150)
                        DT.plt.xlim(-150, 150)

                    # ToDo: Consider right thrust power instead of setting always MAX_THRUST.
                    target_vector = target_vector.as_magnitude(MAX_THRUST)

                # Update Pod instances
                # Set thrust of current Pod instance
                trj[i].thrust_target = trj[i].location + target_vector
                trj[i].thrust_power = target_vector.magnitude()

                # Create next Pod instance and append it to the trajectory list
                # Assuming the angle between target_vector and Pod's face is less than 18 degrees
                # ToDo: Use next_location() not to depend on the assumption
                loc = trj[i].location + trj[i].inertia + target_vector
                inr = loc - trj[i].location
                inr = inr.as_magnitude(inr.magnitude() * .85)
                trj.append(
                    Pod(trj[i].pod_id, "ally", loc[0], loc[1], inr[0], inr[1], math.degrees(target_vector.abs_angle()),
                        trj[i].next_cp_id))

                if DT.debug_mode:
                    tx = [t.location.x() for t in trj]
                    ty = [t.location.y() for t in trj]

                    DT.plt.figure("map")

                    # trj of location by BLUE line
                    DT.plt.plot(tx, ty, lw=2, color="b")

                    # start point by Blue circle
                    if i == 0:
                        DT.plt.plot(tx[0], ty[0], "bo")

                    # CP by RED circle
                    DT.plt.plot(CP[p.next_cp_id][0], CP[p.next_cp_id][1], "ro")
                    DT.plt.plot(CP[p.following_cp_id(1)][0], CP[p.following_cp_id(1)][1], "ro")

                    # angle by RED arrow
                    DT.plt.plot((trj[i].location[0], trj[i].pod_angle_as_vector[0] + trj[i].location[0]),
                                (trj[i].location[1], trj[i].pod_angle_as_vector[1] + trj[i].location[1]),
                                color="black")

                    # inertia by YELLOW arrow
                    DT.plt.plot((trj[i].location[0], trj[i].inertia[0] + trj[i].location[0]),
                                (trj[i].location[1], trj[i].inertia[1] + trj[i].location[1]),
                                color="y")

                    # target by BLUE line
                    DT.plt.plot((trj[i].location[0], trj[i].thrust_target[0]),
                                (trj[i].location[1], trj[i].thrust_target[1]), c="b")

                    DT.plt.xlim(0, 16000)
                    DT.plt.ylim(9000, 0)

                    # DT.plt.figure(2)
                    # DT.plt.plot([math.degrees(tr.abs_angle) for tr in trj], "ro")
                    # DT.plt.ylim(-180, 180)

                # finish simulation when reaching enough distance to CP
                if (loc - p.location).magnitude() > (CP[p.next_cp_id] - p.location).magnitude():
                    break

            p.thrust_target = trj[0].thrust_target
            p.thrust_power = trj[0].thrust_power

            # Before pivoting, rotate target_vector according to the end point of the simulation
            # Todo: maybe inertia needed to be considered.
            if ttr > ttp:
                rot = (loc - p.location).angle_for(CP[p.next_cp_id] - p.location)
                target_vector = (p.thrust_target - p.location).as_magnitude(p.thrust_power).rotate(rot)
                p.thrust_target = target_vector + p.location
                p.thrust_power = target_vector.magnitude()

        elif p.pod_id in EARLY_PIVOT:
            # Early-Pivot Implementation
            # If location_without_thrust is in next_checkpoint within turn_to_pivot, begin early-pivot for the
            # following checkpoint after the next.
            angle_to_pivot = abs(p.angle_for_location(CP[p.following_cp_id(1)]))
            turn_to_pivot = int(angle_to_pivot // ROTATE_PER_TURN)
            inertia_during_pivot = geometric_series(p.inertia, FRICTION, turn_to_pivot)
            DT.stderr("idp", inertia_during_pivot, "inr", p.inertia, "ttp", turn_to_pivot)
            if inertia_during_pivot.distance_from(CP[p.next_cp_id] - p.location) < CP_RADIUS * 0.9:
                # Try Fast-Early-Pivot.
                DT.stderr("Trying Fast-Early-Pivot...", turn_to_pivot, CP[p.following_cp_id(1)])
                p.thrust_target = CP[p.following_cp_id(1)]
                p.thrust_power = MAX_THRUST
                next_inertia = geometric_series(p.next_location() - p.location, FRICTION, turn_to_pivot - 1)
                if next_inertia.distance_from(CP[p.next_cp_id] - p.next_location()) > CP_RADIUS * 0.9:
                    p.thrust_power = MIN_THRUST

                DT.stderr("Early-Pivot...", turn_to_pivot, CP[p.following_cp_id(1)])
            # Set target to the edge of the next checkpoint as usual.
            else:
                target_vector = CP[p.next_cp_id] - p.location
                # Revise target to the closest point from next checkpoint
                target_vector += (CP[p.following_cp_id(1)] - CP[p.next_cp_id]).as_magnitude(CP_RADIUS * 0.5)
                opt_thrust_vector = target_vector - p.inertia
                p.thrust_target = p.location + opt_thrust_vector
                angle_to_pivot = abs(opt_thrust_vector.angle_for(p.pod_angle_as_vector))
                p.thrust_power = min(MAX_THRUST,
                                     max(MIN_THRUST,
                                         opt_thrust_vector.magnitude() * math.cos(angle_to_pivot) / (
                                             1 + angle_to_pivot // ROTATE_PER_TURN)))

        elif p.pod_id in FAST_EARLY_PIVOT:
            pp = p.plan_fep_moves()  # type: List[Pod] # Planned Pod
            p.thrust_target = pp[0].thrust_target
            p.thrust_power = pp[0].thrust_power

        elif p.pod_id in CIRCULAR_MOVE:
            # Circular-Move Implementation
            # In short, the circle of this implementation is TOO BIG !!
            # No chance to have enough speed for the circle-move even if no collision.
            # Speed might be improved by opt_thrust_vector and/or target edge of CP for the circle,
            # but recovering from collisions is difficult

            # Move along with the circle passing through the next 2 CP.
            center_of_circle = Vector(*center_of_three_points(p.location, CP[p.next_cp_id], CP[p.following_cp_id(1)]))
            radius_of_circle = (p.location - center_of_circle).magnitude()
            opt_speed_for_circle = math.pi * radius_of_circle / 10

            # Calculate next_target_vector for optimal speed.
            target_vector = center_of_circle - p.location
            if target_vector.angle_for(CP[p.next_cp_id] - p.location) > 0:
                target_vector = target_vector.rotate(math.pi / 2)
                target_vector = target_vector.rotate(-ROTATE_PER_TURN)
            else:
                target_vector = target_vector.rotate(-math.pi / 2)
                target_vector = target_vector.rotate(ROTATE_PER_TURN)

            # Check Pod speed along with the circle
            inertia_for_circle = p.inertia.magnitude() * math.cos(p.inertia.angle_for(target_vector))
            speed_diff = opt_speed_for_circle - inertia_for_circle * FRICTION

            # If good angle and speed, thrust to keep speed.
            if abs(p.pod_angle_as_vector.angle_for(
                    target_vector)) < math.pi / 5 and MIN_THRUST <= speed_diff <= MAX_THRUST:
                p.thrust_power = speed_diff
                p.thrust_target = p.location + target_vector - p.inertia
            # If too fast, no thrust regardless of angle.
            if speed_diff < MAX_THRUST:
                p.thrust_target = p.location + target_vector - p.inertia
            # If too slow, pivot for shortcut and thrust according to the Pod angle
            else:
                # If Pod speed is not enough, re-calculate target_vector
                target_vector = CP[p.next_cp_id] - p.location
                # Revise target to the closest point from next checkpoint
                target_vector += (CP[p.following_cp_id(1)] - CP[p.next_cp_id]).as_magnitude(CP_RADIUS * 0.5)
                target_vector = target_vector.as_magnitude(opt_speed_for_circle)
                target_vector = target_vector - p.inertia
                if abs(p.angle_for_location(CP[p.next_cp_id])) < math.pi / 2:
                    p.thrust_power = min(MAX_THRUST,
                                         max(MIN_THRUST,
                                             target_vector.magnitude() / math.cos(
                                                 p.angle_for_location(CP[p.next_cp_id]))))
                else:
                    p.thrust_power = MIN_THRUST
                p.thrust_target = p.location + target_vector - p.inertia

        # Shielding Implementation
        for e in all_pods[2:4]:  # Enemy Pods
            positional_vector = e.next_location() - p.next_location()  # type: Vector
            if positional_vector.magnitude() < POD_RADIUS * 2:
                if abs(positional_vector.angle_for(p.pod_angle_as_vector)) < math.pi:
                    p.shield = 1

        # You have to output the target position followed by the power (0 <= thrust <= 100) or "BOOST" or "SHIELD"
        if p.thrust_power == float("inf"):
            print("{0} {1} {2} {2}".format(int(p.thrust_target.x()), int(p.thrust_target.y()), "BOOST"))
        elif p.shield == 1:
            print("{0} {1} {2} {2}".format(int(p.thrust_target.x()), int(p.thrust_target.y()), "SHIELD"))
        else:
            print("{0} {1} {2} {2}".format(int(p.thrust_target.x()), int(p.thrust_target.y()), int(p.thrust_power)))

    history.turn_end(all_pods)

    DT.elapsed_time()

    # To debug: print("Debug messages...", file=sys.stderr)
