# EMG Pacman

How to:
1. Place three electrodes on three of your muscles (we used left and right arms and cheek)
2. Connect electrodes to arduino. Make sure that channels in left_arm_channel, right_arm_channel and third_channel variables are the same as port you connected electrodes to!
3. Upload .ino script to Arduino
4. Move the pacman! Use right arm to move right, left hand to move left, both to move up, and third muscle to move down!
(you can change this logic in arm_status() function)
5. Adjust the thresholds. If the pacman is too sensitive, raise the threshold variables, otherwise, lower it