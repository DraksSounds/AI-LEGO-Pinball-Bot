from pinball_functions.scoreFunctions import FlaskGameServer
from pinball_functions.ev3Functions import Robot
from pinball_functions.cameraFunctions import BallTracker
from pinball_functions.AIBot import AIBot
import threading
import requests
import time
from playsound import playsound
import json
import matplotlib.pyplot as plt

def main():
    # Load statistics
    try:
        with open("models/data last action.json", "r") as f:
            lists = json.load(f)
        games_played, scores, durations = lists["games_played"], lists["scores"], lists["durations"]
    except:
        games_played = []
        scores = []
        durations = []
    try: game = games_played[-1]
    except: game = 0

    # Initialize the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    mng = plt.get_current_fig_manager()
    mng.window.geometry("+50+195")  # Moves window
    ax1.set_title("Game Score Progression")
    ax2.set_title("Game Duration Progression")
    ax1.set_xlabel("Game Number")
    ax2.set_xlabel("Game Number")
    ax1.set_ylabel("Score")
    ax2.set_ylabel("Duration (seconds)")
    ax1.grid(True)
    ax2.grid(True)
    line1, = ax1.plot(games_played, scores, marker="o", linestyle="-", color="b", label="Score")
    line2, = ax2.plot(games_played, durations, marker="o", linestyle="-", color="r", label="Duration")
    ax1.legend()
    ax2.legend()
    plt.ion()  # Turn on interactive mode for live updates
    plt.show(block=False)
    plt.pause(0.2)  # Allow time to render


    def update_graph():
        """Update and refresh the graphs."""
        line1.set_data(games_played, scores)
        ax1.relim()
        ax1.autoscale_view()

        line2.set_data(games_played, durations)
        ax2.relim()
        ax2.autoscale_view()

        plt.draw()
        plt.pause(0.2)  # Give time to refresh the plot

    # Initialize the robot and AI bot
    with Robot() as robot:
        AI_bot = AIBot(robot, tracker, server.read_score)

        # Wait for the start button to be pressed
        while server.game_state == "Starting": time.sleep(0.1)
        print("Game Started")

        # Activate the board and fire the plunger
        robot.activate_board(manual=False)
        robot.fire_plunger()
        robot.check_score()
        time.sleep(0.8)

        while True:
            game += 1
            time_start = time.time()

            # Main game loop
            while server.game_state == "Running":
                # Check if the ball is out of the field
                if tracker.ball_in_square:
                    break

                # Get the score and update the server
                score = robot.check_score()
                if score > 0:
                    server.score += score
                    print(f"Score: {server.score} | High Score: {server.highscore}                           ")
                    requests.get(f'http://localhost:5000/update_score/{server.score}')

                time.sleep(0.1) # Prevent high cpu usage

            # Game Over

            # Exit if the game is ending
            if server.game_state == "Ending":
                break
            time_end = time.time()

            # Visual feedback
            print("Game Over!                                            ")
            server.game_state = "Game Over"
            playsound("sounds/game over.wav", block=False)
            requests.get(f'http://localhost:5000/game_over')
            
            # Update statistics
            if time_end-time_start > 0.2:
                games_played.append(game)
                durations.append(time_end-time_start)
                scores.append(server.score)

            # Update the graphs
            update_graph()

            # Check if the game is ending again
            if server.game_state == "Ending":
                break

            # Start a new game
            print("Starting a new game...")
            requests.get(f'http://localhost:5000/new_game')
            robot.fire_plunger()
            time.sleep(0.8)

        # End the game
        print("Ending the game...                           ")
        tracker.ending = True
        AI_bot.end_thread()
        
    # Save statistics
    lists = {
        "games_played": games_played,
        "scores": scores,
        "durations": durations
    }
    
    with open("models/data last action.json", "w") as f:
        json.dump(lists, f)


if __name__ == "__main__":
    server = FlaskGameServer()
    tracker = BallTracker(rewarp=False, recolor=False, resquare=False)
    flask_thread = threading.Thread(target=server.run, daemon=True)
    flask_thread.start()
    tracker_thread = threading.Thread(target=tracker.track_object, daemon=True)
    tracker_thread.start()
    time.sleep(1)
    main()
