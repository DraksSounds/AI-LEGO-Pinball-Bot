from pinball_functions.scoreFunctions import FlaskGameServer
from pinball_functions.ev3Functions import Robot
from pinball_functions.cameraFunctions import BallTracker
from pinball_functions.simpleBot import simpleBot
import threading
import requests
import time
from playsound import playsound

def main():

    # Initialize the robot and the simple bot
    with Robot() as robot:
        simple_bot = simpleBot(robot, tracker)

        # Wait for the game to start
        while server.game_state == "Starting": time.sleep(0.1)
        print("Game Started")

        # Activate the board and fire the plunger
        robot.activate_board(manual=False)
        robot.fire_plunger()
        robot.check_score()
        time.sleep(0.8)

        while True:

            # Main game loop
            while server.game_state == "Running":
                # Check if the ball is out of the field
                if tracker.ball_in_square:
                    break

                # Check the score and update the server
                score = robot.check_score()
                if score > 0:
                    server.score += score
                    print(f"Score: {server.score} | High Score: {server.highscore}")
                    requests.get(f'http://localhost:5000/update_score/{server.score}')

                time.sleep(0.1) # Prevent high cpu usage

            # Check if the game is ending
            if server.game_state == "Ending":
                break

            # Visual and audio feedback
            print("Game Over!")
            server.game_state = "Game Over"
            playsound("sounds/game over.wav", block=False)
            requests.get(f'http://localhost:5000/game_over')
            
            #  Waiting for the user to press the "Restart Game" button
            while server.game_state == "Game Over":
                time.sleep(0.1) # Prevent high cpu usage

            # Check if the game is ending again
            if server.game_state == "Ending":
                break

            # Start a new game
            print("Starting a new game...")
            requests.get(f'http://localhost:5000/new_game')
            robot.fire_plunger()
            time.sleep(0.8)
        print("Ending the game...")
        tracker.ending = True
        simple_bot.end_thread()


if __name__ == "__main__":
    server = FlaskGameServer()
    tracker = BallTracker(rewarp=False, recolor=False, resquare=False)
    flask_thread = threading.Thread(target=server.run, daemon=True)
    flask_thread.start()
    tracker_thread = threading.Thread(target=tracker.track_object, daemon=True)
    tracker_thread.start()
    time.sleep(1)
    main()
