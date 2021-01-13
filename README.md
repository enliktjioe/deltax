# DeltaX Competition 2020-2021

## Prerequisites

- Supported OS: 
	- Windows (7/10)
	- Ubuntu (preferable 18.04+)
- Currently, macOS doesn't have RCS connector / agent in purpose to connect the RC car from computer
- Create RCSnail API account from [here](https://api.rcsnail.com/signin)
- Python 3.7+ recommended
- Required Python libraries in `requirements.txt`


## How to do Data Recording
- Currently, it's possible only in Windows
- Launch `RCSAgent/RCSnail.exe`, make surce AI checkbox was unticked.
- When the RC Car already connected, control it using joystick controller or other controllers.
- Sample public recording data were accessible from [here](https://owncloud.ut.ee/owncloud/index.php/s/FjqqdgPd4yaF36k)

## How to Run in Windows
- Connect to `rscgx` wifi hotspot at DeltaX Track in 2nd floor of Delta Building University of Tartu. 
- Launch `RCSAgent/RCSnail.exe`, login using RCSnail API account
- To able make the car running *autonomously* based on our AI model, tick the checkbox AI
- Update `RCSnail-AI-lite/config/configuration.yml`:
	- Change the `model_name` with the AI model that you want to run
	- Change the `control_mode` to `full_model` for autonomous mode. The other is `full_expert` for steering with controller
- Go to `RCSnail-AI-lite/src` and run `main_windows.py`


## How to Run in Ubuntu
- Connect to `rscgx` wifi hotspot at DeltaX Track in 2nd floor of Delta Building University of Tartu.
- For Ubuntu it seems that connection to the car works better in eduroam network [[source]](https://courses.cs.ut.ee/t/DeltaX2021SelfDriving/Main/OS)
- Update config `RCSnail-Connector/config/configuration.yml`:
	-  `track` to track id (for example: `deltax` or `eeden`)
	-  `car` to car id (for example: `deltax_i8_01` or `eeden_i8_04`
- Update RCSnail API username and password from function `sign_in_with_email_and_password` in `RCSnail-Connector/src/main.py`
- Before running the RC connector, we need to run the AI model first from `RCSnail-AI-lite/src/main_ubuntu.py`
- Update `RCSnail-AI-lite/config/configuration.yml`:
	- Change the `model_name` with the AI model that you want to run
	- Change the `control_mode` to `full_model` for autonomous mode.
- Run Ubuntu RC Connector using `RCSnail-Connector/src/main.py`
- It seems that the RCSnail-Connector works fine only if launched from its root folder, not from inside the src/ folder (so you have to do src/main.py) [[source]](https://courses.cs.ut.ee/t/DeltaX2021SelfDriving/Main/OS)

## Reference(s)
- [DeltaX Self-driving competition](https://courses.cs.ut.ee/t/DeltaX2021SelfDriving/Main/HomePage)
- [RCSnail API](https://api.rcsnail.com/)

