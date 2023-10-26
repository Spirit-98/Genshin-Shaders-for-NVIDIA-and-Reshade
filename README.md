# Genshin-Shaders-for-NVIDIA-and-Reshade
You NEED a Nvidia GPU (with Geforce Experience) to use these filters with Game filters, otherwise you should use reshade.

Step 1 - Goto 'C:\Program Files\NVIDIA Corporation\' (or where ever you have installed windows)

Step 2 - Download the zip file in the repo and extract the 'Ansel' folder to the 'C:\Program Files\NVIDIA Corporation\' directory.

Step 3 - Head over to where you installed Genshin impact, and inside that folder there should be another folder called 'Genshin Impact game': 'C:\Program Files\Genshin Impact\Genshin Impact game'

Step 4: We need to rename two things within the 'Genshin Impact game ' folder 
1. GenshinImpact_data and 2. GenshinImpact.exe

rename GenshinImpact_data to eurotrucks2_data

and

rename GenshinImpact.exe to eurotrucks2.exe

From now on, if you want to use the filters, you must start the game from eurotrucks2.exe NOT from the launcher.

Step 5 IMPORTANT!!. Start the game from eurotrucks2.exe, and immediately hit CTRL-Z or manually rename eurotrucks2.exe back to GenshinImpact.exe to login (sometimes this step isnt needed but like 99% of the time it is needed)

Once in game you can hit ALT-F3 or ALT-Z to bring up the Game Filters or geforce experience UI

from there you can mess around with all the reshade filters you want that I included, but some dont work unless you have them added to a .ini file and use that .ini file as a preset

You could also use the filters I made and worked hard on for 1+ years

To update the game to a newer verison: rename eurotrucks2_data and eurotrucks2.exe BACK to GenshinImpact_data and GenshinImpact.exe respectively.


IF THE FILTERS STOP WORKING/game filters doesnt work anymore/Isnt showing up when you hit ALT-F3:
rename last used .ini config file for filters (will force stop them from automatically being on the next time) - THIS STEP WILL FIX THE PROBLEM almost 100% of the time
delete tempfiles %temp%
turn off image sharpening from Nvidia control panel
delete AppData/Local/NVIDIA/GLCache
delete /NVIDIA CORP/nvidia geforce and share deleting share isnt neccessary

UPDATE 26/10/2023:
Guys I have some really bad news, as of October 23, 2023, Miyhoyo patched the method I used to get these filters working.
The filters work fine, but you can no longer login to the game. when renaming back to Genshinimpact.exe - the game auto creates a new file called eurotrucks2.exe stopping you from logging in.
I sent in a ticket to mihoyo over this issue so maybe they revert it, but I'm doubtful 😞
I've been using the filters since June 27, 2022 soo I'm kinda sad about this.
