python main.py
#!/bin/bash
# OK, just shutdown all ... applications after n minutes
sudo shutdown -h +2 &
# Try normal shutdown in the meantime
osascript -e 'tell application "System Events" to shut down'