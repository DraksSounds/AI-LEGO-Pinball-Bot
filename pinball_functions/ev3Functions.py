import ev3_dc as ev3
from time import sleep, time
import threading
from playsound import playsound

class Robot:
    def __init__(self):
        """
        Initializes the EV3 resources.
        """

        # Connect to both EV3's
        self.ev3_function = ev3.EV3(protocol=ev3.BLUETOOTH, host="00:00:00:00:00:00")       # EV3_FUNCTION
        print(self.ev3_function)
        self.ev3_score = ev3.EV3(protocol=ev3.BLUETOOTH, host="00:00:00:00:00:00")       # EV3 (score)
        print(self.ev3_score)

        # EV3_FUNCTION
        self.m_flip_right = ev3.Motor(ev3.PORT_A, ev3_obj=self.ev3_function)
        self.m_flip_left  = ev3.Motor(ev3.PORT_D, ev3_obj=self.ev3_function)
        self.m_plunger    = ev3.Motor(ev3.PORT_B, ev3_obj=self.ev3_function)

        self.s_right  = ev3.Touch(ev3.PORT_1, ev3_obj=self.ev3_function)
        self.s_left = ev3.Touch(ev3.PORT_4, ev3_obj=self.ev3_function)

        self.m_flip_right.speed = 100
        self.m_flip_left.speed  = 100
        self.m_plunger.speed    = 50

        self.m_flip_left.ramp_up_time = 0.1
        self.m_flip_left.ramp_up = 360
        self.m_flip_right.ramp_up_time = 0.1
        self.m_flip_right.ramp_up = 360

        self.m_flip_left.ramp_down_time = 0.1
        self.m_flip_left.ramp_down = 360
        self.m_flip_right.ramp_down_time = 0.1
        self.m_flip_right.ramp_down = 360

        # EV3_SCORE
        self.m_trapdoor = ev3.Motor(ev3.PORT_A, ev3_obj=self.ev3_score)
        self.m_spinner  = ev3.Motor(ev3.PORT_D, ev3_obj=self.ev3_score)

        self.s_100 = ev3.Color(ev3.PORT_1, ev3_obj=self.ev3_score)
        self.s_200 = ev3.Color(ev3.PORT_2, ev3_obj=self.ev3_score)
        self.s_300 = ev3.Color(ev3.PORT_3, ev3_obj=self.ev3_score)

        self.m_trapdoor.speed = 25
        self.m_spinner.speed  = 10

        self.previous_left_direction = "down"
        self.previous_right_direction = "down"
        self.m_flip_left.start_move_for(0.5, brake=True)
        self.m_flip_right.start_move_for(0.5, brake=True)
        self.m_flip_left.position = 0
        self.m_flip_right.position = 0

        self.s_100_reflected = self.s_100.reflected + 4
        self.s_200_reflected = self.s_200.reflected + 3
        self.s_300_reflected = self.s_300.reflected + 3
        self.score = 0

    def activate_board(self, manual=True):
        """
        Activates the thread that handles the trapdoor and spinner.

        Args:
            manual (bool): If the flipper should be controlled manually (using the buttons).
        """
        self.busy = True
        self.board_thread = threading.Thread(target=self.board_loop, daemon=True)
        self.board_thread.start()
        print("Board thread activated.")
        self.score_thread = threading.Thread(target=self.score_loop, daemon=True)
        self.score_thread.start()
        print("Score thread activated.")
        if manual:
            self.flip_thread = threading.Thread(target=self.flip_loop, daemon=True)
            self.flip_thread.start()
            print("Flip thread activated.")

    def deactivate_board(self):
        """
        Deactivates the thread that handles the trapdoor and spinner.
        """
        try:
            self.busy = False
            self.board_thread.join()
            print("Board thread deactivated.")
            self.score_thread.join()
            print("Score thread deactivated.")
            self.flip_thread.join()
            print("Flip thread deactivated.")
        except:
            print("No other thread to stop.")

    def board_loop(self):
        """
        The loop that handles the trapdoor and spinner.

        Spins the spinner and moves the trapdoor up and down at an interval.
        """

        self.m_spinner.start_move()
        while self.busy:
            for i in range(15):
                sleep(1)
                if not self.busy:
                    break
            self.m_trapdoor.start_move_to(-140, brake=True)
            for i in range(2):
                sleep(0.75)
                if not self.busy:
                    break
            self.m_trapdoor.start_move_to(0, brake=True)

    def flip_loop(self):
        while self.busy:
            if self.s_left.touched:
                self.flip("left", "up")
            else:
                self.flip("left", "down")
            if self.s_right.touched:
                self.flip("right", "up")
            else:
                self.flip("right", "down")

    def fire_plunger(self):
        """
        Fires the plunger.
        """
        duration = 2.0
        speed = 20

        self.m_plunger.start_move_for(duration=duration, speed=speed, brake=True)
        sleep(duration)
        self.m_plunger.start_move_to(0, brake=True)

    def flip(self, side: str, direction: str):
        """
        Flips the flipper in the given direction.

        Args:
            sdide (str): The motor to flip. ("left" or "right")
            direction (str): The direction to flip the motor in. ("up" or "down")
        """
        if side == "left":
            if direction == "up" and self.previous_left_direction == "down":
                self.m_flip_left.start_move_for(0.2, brake=True, direction=-1)
            elif direction == "down" and self.previous_left_direction == "up":
                self.m_flip_left.start_move_for(0.2, brake=True, direction=1)
            self.previous_left_direction = direction
        else:
            if direction == "up" and self.previous_right_direction == "down":
                self.m_flip_right.start_move_for(0.2, brake=True, direction=-1)
            elif direction == "down" and self.previous_right_direction == "up":
                self.m_flip_right.start_move_for(0.2, brake=True, direction=1)
            self.previous_right_direction = direction

    def score_loop(self):
        """The loop that handles the score sensors."""

        self.scored = False
        while self.busy:
            prev_time = time()
            self.score = 0
            # Wait for a sensor to detect the ball
            while self.busy and self.score == 0:

                # 100 points
                if self.s_100.reflected > self.s_100_reflected:
                    time_diff = time() - prev_time
                    if time_diff <= 5.0: # Double if scored within 5 seconds
                        print(f"Double! ({round(time_diff, 1)}/5.0 seconds)                           ")
                        self.score += 200
                        playsound("sounds/100 double.wav", block=False)
                    else:
                        self.score += 100
                        playsound("sounds/100.wav", block=False)

                # 200 points
                elif self.s_200.reflected > self.s_200_reflected:
                    time_diff = time() - prev_time
                    if time_diff <= 5.0: # Double if scored within 5 seconds
                        print(f"Double! ({round(time_diff, 1)}/5.0 seconds)                           ")
                        self.score += 400
                        playsound("sounds/200 double.wav", block=False)
                    else:
                        self.score += 200
                        playsound("sounds/200.wav", block=False)

                # 300 points
                elif self.s_300.reflected > self.s_300_reflected:
                    time_diff = time() - prev_time
                    if time_diff <= 5.0: # Double if scored within 5 seconds
                        print(f"Double! ({round(time_diff, 1)}/5.0 seconds)                           ")
                        self.score += 600
                        playsound("sounds/300 double.wav", block=False)
                    else:
                        self.score += 300
                        playsound("sounds/300.wav", block=False)
                
                # Prevent too high cpu usage
                sleep(0.005)

            self.scored = True
            #  Wait for the score to be checked by the main loop
            while self.scored and self.busy:
                sleep(0.1)

    def check_score(self)->int:
        """
        Checks the score sensors and returns and resets the score.

        Returns:
            int: The score.
        """
        if self.scored:
            self.scored = False
            return self.score
        else: return 0

    def __enter__(self):
        """
        Starts the EV3 resources on enter.
        """
        return self

    def cleanup(self):
        """
        Cleans up the EV3 resources.

        Stops the motors and disconnects from the EV3.
        """
        # Stop the board thread
        self.deactivate_board()

        # Reset the trapdoor position
        while self.m_trapdoor.busy: pass
        if self.m_trapdoor.position < 0:
            print("Resetting motor positions...")
            self.m_trapdoor.start_move_to(0, brake=True)
            while self.m_trapdoor.busy: pass

            # Wait for the motor to be in position
            sleep(1)

        # Stop the motors
        self.m_trapdoor.stop(brake=False)
        self.m_spinner.stop(brake=False)
        self.m_flip_left.stop(brake=False)
        self.m_flip_right.stop(brake=False)
        self.m_plunger.stop(brake=False)

        # Debug
        print("EV3 resources cleaned up.")

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Cleans up the EV3 resources on exit.

        Stops the motors and disconnects from the EV3.
        """
        self.cleanup()

if __name__ == "__main__":
    with Robot() as robot:
        
        robot.activate_board()

        # Test the motors
        robot.m_flip_left.start_move_for(1.0, brake=True)
        sleep(1.0)
        robot.m_flip_right.start_move_for(1.0, brake=True)
        sleep(1.0)
        robot.fire_plunger()
        sleep(1.0)