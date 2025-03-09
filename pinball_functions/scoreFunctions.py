import time
import threading
import logging
import requests
import socket
from flask import Flask, render_template, jsonify

class FlaskGameServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.score = 0
        self.highscore = 0
        self.game_state = "Starting"
        self._setup_routes()
        
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    def _setup_routes(self):
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/update_score/<int:new_score>', 'update_score', self.update_score)
        self.app.add_url_rule('/get_score', 'get_score', self.get_score)
        self.app.add_url_rule('/game_over', 'game_over', self.game_over)
        self.app.add_url_rule('/new_game', 'new_game', self.new_game)
        self.app.add_url_rule('/end_game', 'end_game', self.end_game)

    def index(self):
        return render_template('scoreboard.html')

    def update_score(self, new_score):
        self.score = new_score
        if self.score > self.highscore:
            self.highscore = self.score
        return jsonify({'status': 'success', 'score': self.score, 'highscore': self.highscore})

    def get_score(self):
        return jsonify({'score': self.score, 'highscore': self.highscore, 'gameState': self.game_state})

    def game_over(self):
        self.game_state = "Game Over"
        return jsonify({'score': self.score, 'highscore': self.highscore, 'gameState': self.game_state})

    def new_game(self):
        self.score = 0
        self.game_state = "Running"
        return jsonify({'score': self.score, 'highscore': self.highscore, 'gameState': self.game_state})

    def end_game(self):
        self.game_state = "Ending"
        return jsonify({'score': self.score, 'highscore': self.highscore, 'gameState': self.game_state})

    def run(self):
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Flask server running at http://{local_ip}:5000")
        self.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    
    def read_score(self):
        return self.score

# Example usage
def game_loop(server):
    print("Game Started")
    
    while True:
        if server.game_state == "Running":
            # Increment the score by a random value between 50 and 100
            server.score = 1200
            print(f"Score: {server.score} | High Score: {server.highscore}")
            # Update the score on the server
            requests.get(f'http://localhost:5000/update_score/{server.score}')
            time.sleep(1)

            # Check if the score has reached or exceeded 500
            if server.score >= 500:
                print("Game Over!")
                server.game_state = "Game Over"
                # Notify the server that the game is over
                requests.get(f'http://localhost:5000/game_over')
                
                # Wait for a while before starting a new game
                for _ in range(10):
                    if server.game_state == "Running":
                        break
                    time.sleep(0.5)
                
                print("Starting a new game...")
                # Start a new game if the game state is still "Game Over"
                if server.game_state == "Game Over":
                    requests.get(f'http://localhost:5000/new_game')
                time.sleep(1)

        time.sleep(1)

if __name__ == "__main__":
    server = FlaskGameServer()
    flask_thread = threading.Thread(target=server.run, daemon=True)
    flask_thread.start()
    time.sleep(1)
    game_loop(server)
