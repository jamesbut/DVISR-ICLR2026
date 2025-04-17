#!/bin/bash

# This script transfers run scripts to a server and executes those
# scripts on that server.
# The script and commands in question are different depending on whether
# they are being ran on the myriad system or a custom server.
# This script should be ran from this directory.

server_name="myriad"
server_path="~/BayesianSymbolicRegression"
exp_args_file="exp_config_args.txt"

# TODO: Add case where exp_config_args.txt is empty

# Read config modification args from exp_args_file
exp_config_args=()
while IFS= read -r line; do
    exp_config_args+=("$line")
done < "$exp_args_file"

# Create jobscripts folder and copy myriad_jobscript.sh in there to use as
# a template
mkdir -p jobscripts
cp myriad_jobscript.sh jobscripts

for ((i = 0; i < ${#exp_config_args[@]}; i++)); do

    # Create new file for each config args
    file_name="jobscripts/myriad_jobscript_$i.sh"
    cp jobscripts/myriad_jobscript.sh $file_name

    # Read last line of file
    last_line=$(tail -n 1 "$file_name")

    # Create new line
    new_line="$last_line ${exp_config_args[$i]} < outs/out$i.txt"

    # Modify last line in file
    { sed '$d' $file_name; echo $new_line; } > tmp && mv tmp $file_name

done

# Remove template jobscript
rm jobscripts/myriad_jobscript.sh

exit

# Transfer myriad_jobscript.sh file to myriad server.
# This way, one can make changes to the script here and those changes
# are reflected on the server
scp myriad_jobscript.sh $server_name:$server_path"/scripts"

# Run myriad job using qsub
ssh $server_name << EOF
    cd $server_path
    qsub scripts/myriad_jobscript.sh
EOF
