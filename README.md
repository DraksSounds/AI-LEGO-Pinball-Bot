# AI-LEGO-Pinball-Bot
Python code that runs a full PinBall experience, with 3 different ways to control the PinBall machine.

This is the code for my AI Lego Pinball machine, as seen on my YouTube channel:
<div align="left">
  <a href="https://youtu.be/do4EYq6oEQg"><img src="https://img.youtube.com/vi/do4EYq6oEQg/0.jpg" alt="AI LEGO Pinball machine video"></a>
</div>

## Table of Contents
1. [Introduction](#introduction)
2. [Credits](#credits)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [About The Machine](#about-the-machine)
7. [License](#license)
8. [Contributing](#contributing)
9. [Disclaimer](#disclaimer)

## Introduction
The code involves 3 main codes for 3 different ways to control a Pinball machine. This includes manual, with a bot, and with AI. Furthermore, there are some modules to handle the robot and an HTML code for the scoreboard that is hosted locally.

## Credits
This project contains sounds that I did not produce.

The "game over.wav" file was downloaded from this YouTube video by "Sumit - FOXITOGO" (@foxitogoextra9418):\
Arcade Retro Game Over - Sound Effect (Final Cut) | https://www.youtube.com/watch?v=FVJJKIJWKdc \
I cannot find it elsewhere, but I'm not certain this was the original creator. Do notify me if you know the original creator.

The point scoring sounds are from Epidemic Sound.\
Game, Points, Single | https://www.epidemicsound.com/sound-effects/tracks/21e648ed-3c9e-45e1-bd14-da5623019081/

## Features
The project includes multiple features that work simultaneously.
- **Motorized Plunger:** Returns the ball to the field after draining.
- **Motorized Flippers:** Controlled with touch sensors.
- **Gameplay Mechanics:** Moving elements (spinning circle, trap door).
- **Scoring:** Color sensors sense the ball, scoring points and applying bonuses.
- **Tablet Score Display:** Displays scores and tracks high scores, also starts new games.
- **Vision System:** Camera tracks the ball and triggers game over when needed. Currently only works with color isolation.
- **Control:** There are three ways to control the machine:
  - **Manual Control:** Buttons control flippers for simple manual gameplay.
  - **Simple Bot:** Predicts ball movement and flips based on its position.
  - **Neural Network AI:** Learns to play in real-time by adjusting strategies using rewards and punishments (Reinforcement Learning).

## Installation
### Requirements
- Lego PinBall machine containing two LEGO Mindstorms EV3 kits
- Python 3 with libraries installed (see requirements.txt)
- Camera (e.g., phone with DroidCam or a webcam)
- Tablet / iPad with internet access
- Speaker
- Tested on a Windows 10 operating system.

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/CreativeMindstorms/AI-LEGO-Pinball-Bot.git
2. Create a virtual environment and install the requirements.
3. Adjust variables inside the main code and modules, especially:
	`src/pinball_functions/ev3Functions.py` and `src/pinball_functions/cameraFunctions.py`
4. Warp your camera and adjust the hsv values for the ball by running `src/pinball_functions/cameraFunctions.py`

## Usage
Run any `src/main_*.py` after installation and configuration.

Visit the website that will be printed in the console to see the scoreboard.

## About The Machine
The machine uses two Mindstorms EV3 Bricks, of which you need the Bluetooth address.
### Connections

**ev3_function**:
| PORT | DESCRIPTION |
|------|-------------|
| A    | Right flipper motor. Stalls at limits. |
| B    | Plunger motor, controlled in `pinball_functions.ev3Functions.Robot.fire_plunger`. |
| D    | Left flipper motor. Stalls at limits. |
| 1    | Right touch sensor, to control flippers manually. |
| 4    | Lefttouch sensor, to control flippers manually. |

**ev3_score**:
| PORT | DESCRIPTION |
|------|-------------|
| A    | Trapdoor motor. Opens and closes to predefined positions in `pinball_functions.ev3Functions.Robot.board_loop`. |
| D    | Spinner motor. Spins at constant speed in `pinball_functions.ev3Functions.Robot.board_loop`. |
| 1    | Color sensor to detect 100 points scored. |
| 2    | Color sensor to detect 200 points scored. |
| 3    | Color sensor to detect 300 points scored. |


## License
This project is licensed under the [GPLv3 License](LICENSE). Contributions and modifications are welcome, but they must remain open-source and credit the original author.

## Contributing

Contributions are welcome, and I appreciate your interest in improving this project! However, I want to keep this easy to manage, as it is a personal project and a learning experience for me.

If you’d like to suggest changes or improvements, here’s how you can contribute:

1.  **Fork the Repository:** Create a personal copy of the repository by clicking the "Fork" button at the top.
2.  **Make Changes:** Implement your changes in your forked repository. Please keep changes focused and well-documented.
3.  **Submit a Pull Request (PR):** Open a pull request with a clear explanation of your changes. Include why the change is beneficial and how it affects the project.

## Disclaimer

This project is a hobby, and while I enjoy working on it, I can’t provide consistent support or assistance. Please feel free to reach out via email for questions or feedback, but responses may be delayed depending on my availability.
