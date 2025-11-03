•	Tizen Doctor

1.	Verify VS Command
•	Implement a class to check the installation and version of Visual Studio extension for Tizen.
•	Implement a class to check the installation of Visual Studio.
•	Implement a class to check if all the required VS Workload/toolsets is installed.
2.	Check Chrome browser and default browser installation status for verify Web Commad.
•	Check if chrome is installed or not.
•	Report Chrome path if installed.
•	Check which is default browser set in the system.
3.	http proxy checker and set-http-proxy command
•	Check whether user’s system network needs HTTP proxy setting.
•	Provide information regarding command to set HTTP Proxy in Tizen doctor tool.
4.	Implement a class to check and report whether host system supports HW Acceleration for Emulator execution 
•	Check if system supports CPU virtualization for emulator execution.
•	Check if it's Windows or Linux, to decide whether to use
a.	KVM (for Linux)
b.	HAXM/WHPX(Hyper-V)(Windows)
•	Verify if the CPU is of Intel or AMD, to decide whether to use
o	WHPX (Hyper-V)(AMD)
o	HAXM/WHPX (Windows)
•	Verify if HAXM/Hyper-V/KVM is installed / enabled or not.
•	Report the install status and help guide.

5.	Tizen Doctor Diagnostic Tool Enhancements
•	Set command: It been added to create presets for external tools.
•	List command: Displays all stored diagnostic tools and their conditions
•	Show Command: Displays the stored conditions of a specific tool in JSON format
•	Remove Command: Deletes a specific condition of a tool.
•	Update Command: Updates a specific condition of a tool with a new JSON file.
•	verify command: produces clear and script-friendly results.
•	Preset security: 
o	Protection via File System Permissions(hard Task)
o	Implement a Preset "Lock" Mechanism
o	Verify Integrity with a Checksum
•	Improve summary section of the tizen doctor Command
•	Fix Sam/CQM Score
•	Documentaion: Write the whole Documentation for tizen Doctor(give importance)
•	TastCase:  Write the all test case for Tizen Doctor(give importance) 
