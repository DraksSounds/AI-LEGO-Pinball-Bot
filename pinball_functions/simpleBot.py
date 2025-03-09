import threading
import time
from random import randint

class simpleBot:
    def __init__(self, robot, tracker):
        self.robot = robot
        self.tracker = tracker
        self.width = tracker.width
        self.height = tracker.height
        self.busy = True
        
        # Initialize velocity tracking variables
        self.prev_x = self.tracker.ball_x
        self.prev_y = self.tracker.ball_y
        self.prev_time = time.time()
        self.ball_vx = 0
        self.ball_vy = 0
        self.just_stuck = False

        self.bot_thread = threading.Thread(target=self.bot_loop, daemon=True)
        self.bot_thread.start()

    def update_velocity(self):
        """Calculates the ball's velocity based on position changes over time."""
        current_time = time.time()
        dt = current_time - self.prev_time

        if dt > 0:  # Prevent division by zero
            self.ball_vx = (self.tracker.ball_x - self.prev_x) / dt
            self.ball_vy = (self.tracker.ball_y - self.prev_y) / dt

        # Update previous values
        self.prev_x = self.tracker.ball_x
        self.prev_y = self.tracker.ball_y
        self.prev_time = current_time

    def predict_future_position(self, time_ahead=0.07):
        """Predicts where the ball will be in a short time based on its velocity."""
        predicted_x = self.tracker.ball_x + self.ball_vx * time_ahead
        predicted_y = self.tracker.ball_y + self.ball_vy * time_ahead
        return predicted_x, predicted_y

    def make_decision(self):
        """Makes a decision on whether to flip the flippers based on ball position and movement."""
        self.update_velocity()  # Update velocity before making a decision

        ball_vx = self.ball_vx
        ball_vy = self.ball_vy

        # Predict where the ball will be in the near future
        predicted_x, predicted_y = self.predict_future_position()

        # Adjust threshold dynamically based on velocity
        base_threshold = 0.77 * self.height
        speed_factor = min(abs(ball_vy) / 300, 0.15)  # Adjust threshold for speed, capping at 15%
        dynamic_threshold = base_threshold - (speed_factor * self.height)

        stuck_threshold = 20  # Velocity threshold to detect a stuck ball

        right_flipper_up = False
        left_flipper_up = False

        if predicted_y > dynamic_threshold and not self.just_stuck:  # Use predicted position
            if predicted_x > 0.55 * self.width:
                self.robot.flip("right", "up")
                right_flipper_up = True
            elif predicted_x < 0.45 * self.width:
                self.robot.flip("left", "up")
                left_flipper_up = True
            else:
                self.robot.flip("right", "up")
                self.robot.flip("left", "up")
                right_flipper_up = left_flipper_up = True
        else:
            self.robot.flip("left", "down")
            self.robot.flip("right", "down")

        # Prevent the ball from getting stuck
        if abs(ball_vy) < stuck_threshold and abs(ball_vx) < stuck_threshold:
            self.just_stuck = True

            # Have a 2.5% chance to flip both flippers up when the ball is stuck
            if randint(1, 40) == 1 and predicted_y > dynamic_threshold:
                self.robot.flip("right", "up")
                self.robot.flip("left", "up")
                right_flipper_up = left_flipper_up = True
                
            # In all other cases, flip the flippers down, to allow the ball to roll off
            else:
                if right_flipper_up:
                    self.robot.flip("right", "down")
                if left_flipper_up:
                    self.robot.flip("left", "down")
        else:
            self.just_stuck = False

    def bot_loop(self):
        """Continuously makes decisions for the bot in a loop."""
        while self.busy:
            self.make_decision()
            time.sleep(0.1)  # Adjusted for better prediction timing
    
    def end_thread(self):
        """Stops the bot loop safely."""
        self.busy = False
        try:
            self.bot_thread.join()
        except:
            pass
